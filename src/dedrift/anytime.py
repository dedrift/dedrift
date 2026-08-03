"""The anytime-valid check: stateful counterpart to :mod:`dedrift.check`.

Same logs, same canary design, same channels — a different inference layer.
The fixed-sample path answers "is this cycle different?" and controls error
*per check*; this one answers "has anything changed since the epoch began?"
and controls error over the whole trajectory. Both run on identical inputs
so the comparison is reproducible from the CLI rather than from a study
script.

The budget, and the part that is easy to get wrong
--------------------------------------------------
``alpha = alpha_prime + gamma_total``, where ``alpha_prime`` is the e-BH
level and ``gamma_total`` buys nuisance-parameter coverage for the *whole
battery*. e-BH needs every input to be a valid e-value, and a process whose
interval misses the truth is not one, so coverage failures union-bound
across processes:

    P(ever a false alert) <= alpha_prime + sum_i gamma_i

Hence ``gamma_i = gamma_total / K``, computed from the live process count
rather than typed in. At 24 processes, using ``gamma_i = gamma_total``
directly would state 0.05 while delivering 0.28.

What is and is not established
------------------------------
* Per process, Ville's inequality bounds the probability of *ever* crossing.
* Across the battery at a single cycle, e-BH controls FDR under arbitrary
  dependence.
* Across the battery *over the trajectory*, anytime-valid FDR requires the
  causal condition of the stopped-e-BH result (see
  :mod:`dedrift.evalues.ebh`), which our dependent streams plausibly but
  not provably satisfy. Reports say "measured" for that, never "proven".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from dedrift.check import DEGRADED_ERROR_FRACTION, get_golden_baseline
from dedrift.config import ProjectConfig
from dedrift.evalues import (
    EProcessState,
    PriorState,
    epoch_fingerprint,
    rate_evalue,
    symmetric_grid,
    update_process,
    ville_threshold,
)
from dedrift.evalues.ebh import ebh_from_logs
from dedrift.evalues.process import state_to_row
from dedrift.evalues.rates import per_process_gamma
from dedrift.signatures import signatures_frame
from dedrift.signatures.structural import RATE_SIGNATURES
from dedrift.store import Store

#: Bumped when signature extraction changes meaning; part of the epoch
#: fingerprint, because evidence gathered with a different instrument is
#: not evidence about the same null.
EXTRACTOR_VERSION = "1"


@dataclass(frozen=True)
class ProcessReport:
    """One e-process as the report renders it."""

    key: tuple[str, str, str, str]
    log_wealth: float
    evidence: float
    epoch: int
    cycles: int
    bets_placed: int
    rise_cycle: int | None
    crossed_at: int | None
    rejected: bool
    detail: str

    @property
    def label(self) -> str:
        """Report-friendly identity, e.g. ``[golden] edge_case/refusal (rate)``."""
        b, f, s, c = self.key
        return f"[{b}] {f}/{s} ({c})"


@dataclass(frozen=True)
class AnytimeCheckResult:
    """Outcome of one anytime-valid check.

    Attributes:
        ts: Check timestamp (UTC, ISO).
        current_cycle: Cycle folded in by this check.
        fingerprint: Epoch fingerprint in force.
        alpha: Lifetime battery-wide budget.
        alpha_prime: e-BH level.
        gamma_total: Total coverage budget.
        gamma_per_process: ``gamma_total / K``, the level each interval uses.
        n_processes: Processes in the pool.
        verdict: ``DRIFT DETECTED`` / ``OK`` / ``DEGRADED DATA``.
        degraded: Whether drift conclusions were suppressed.
        processes: Per-process report rows.
        resets: Human-readable epoch-reset notices.
        n_alerts: Convenience count.
    """

    ts: str
    current_cycle: str
    fingerprint: str
    alpha: float
    alpha_prime: float
    gamma_total: float
    gamma_per_process: float
    n_processes: int
    verdict: str
    degraded: bool
    processes: list[ProcessReport] = field(default_factory=list)
    resets: list[str] = field(default_factory=list)

    @property
    def n_alerts(self) -> int:
        """Processes rejected by e-BH at this check."""
        return sum(1 for p in self.processes if p.rejected)

    def alerts(self) -> list[ProcessReport]:
        """Rejected processes, deterministically ordered."""
        return sorted((p for p in self.processes if p.rejected), key=lambda p: p.key)


def load_states(store: Store) -> dict[tuple[str, str, str, str], EProcessState]:
    """Read persisted e-process state.

    Columns are named explicitly rather than relying on ``SELECT *`` and a
    row factory: the schema will grow, and positional unpacking would break
    silently at exactly the moment state stops meaning what it says.
    """
    conn = store.connect()
    out: dict[tuple[str, str, str, str], EProcessState] = {}
    query = (
        "SELECT baseline, family, signature, channel, log_wealth, cycles, "
        "bets_placed, epoch, fingerprint, peak_log_wealth, rise_cycle, "
        "crossed_at, prior_successes, prior_trials, reference_successes, "
        "reference_trials FROM eprocess_state"
    )
    for row in conn.execute(query):
        (
            baseline,
            family,
            signature,
            channel,
            log_wealth,
            cycles,
            bets_placed,
            epoch,
            fingerprint,
            peak,
            rise_cycle,
            crossed_at,
            prior_s,
            prior_t,
            ref_s,
            ref_t,
        ) = row
        key = (baseline, family, signature, channel)
        out[key] = EProcessState(
            key=key,
            log_wealth=log_wealth,
            cycles=cycles,
            bets_placed=bets_placed,
            epoch=epoch,
            fingerprint=fingerprint,
            prior=PriorState(
                n_cycles=cycles,
                successes=prior_s,
                trials=prior_t,
                reference_successes=ref_s,
                reference_trials=ref_t,
            ),
            peak_log_wealth=peak,
            rise_cycle=rise_cycle,
            crossed_at=crossed_at,
        )
    return out


def save_states(store: Store, states: list[EProcessState], ts: str) -> None:
    """Upsert e-process state."""
    conn = store.connect()
    for st in states:
        row = state_to_row(st)
        row["updated_ts"] = ts
        cols = ", ".join(row)
        marks = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT OR REPLACE INTO eprocess_state ({cols}) VALUES ({marks})",
            tuple(row.values()),
        )
    conn.commit()


def run_anytime_check(store: Store, config: ProjectConfig | None = None) -> AnytimeCheckResult:
    """Fold the latest cycle into the rate-channel e-processes and adjudicate.

    Only the rate channel is implemented: a verified single channel is worth
    more than four unverified ones, and the scalar and semantic channels
    need constructions of their own rather than a reuse of this one.

    Args:
        store: Project store containing canary records with cycle IDs.
        config: Project configuration; loaded from the store when omitted.

    Returns:
        The check result. State is persisted before returning.

    Raises:
        ValueError: If no canary cycles exist.
    """
    cfg = config or ProjectConfig.load(store.project_dir)
    ac = cfg.anytime
    records = [r for r in store.read_records() if r.cycle_id is not None]
    if not records:
        msg = "no canary records with cycle IDs; run canaries first"
        raise ValueError(msg)

    frame = signatures_frame(records)
    cycles = list(dict.fromkeys(frame["cycle_id"]))
    current = cycles[-1]
    golden = [c for c in get_golden_baseline(store) if c in cycles and c != current]
    rolling = [c for c in cycles[:-1] if c not in golden][-cfg.rolling_window_cycles :]

    fingerprint = epoch_fingerprint(
        suite_version=str(len({r.canary_id for r in records if r.canary_id})),
        embedder=cfg.embedder,
        golden_cycles=tuple(golden),
        extractor_version=EXTRACTOR_VERSION,
    )

    cur_frame = frame[frame["cycle_id"] == current]
    degraded = bool(cur_frame["had_error"].mean() > DEGRADED_ERROR_FRACTION)
    families = sorted(frame["family"].unique())

    # The pool must be enumerated before any bet is placed: the coverage
    # budget per process depends on the pool size, and the interval is part
    # of the bet, so a pool that changed mid-check would break predictability.
    pool: list[tuple[str, str, list[str]]] = [
        (name, sig, ref)
        for name, ref in (("rolling", rolling), ("golden", golden))
        if ref
        for sig in RATE_SIGNATURES
    ]
    n_pool = len(pool) * len(families)
    gamma_i = per_process_gamma(ac.gamma_total, max(n_pool, 1))
    grid = symmetric_grid(ac.tilts)

    states = load_states(store)
    updated: list[EProcessState] = []
    log_evalues: list[float] = []
    rows: list[dict[str, Any]] = []
    resets: list[str] = []

    for baseline_name, sig, ref_cycles in pool:
        ref_frame = frame[frame["cycle_id"].isin(ref_cycles)]
        for family in families:
            key = (baseline_name, family, sig, "rate")
            state = states.get(key, EProcessState(key=key))
            ref_fam = ref_frame[ref_frame["family"] == family][sig].dropna()
            cur_fam = cur_frame[cur_frame["family"] == family][sig].dropna()

            prior = PriorState(
                n_cycles=state.prior.n_cycles,
                successes=state.prior.successes,
                trials=state.prior.trials,
                reference_successes=int(ref_fam.sum()),
                reference_trials=len(ref_fam),
            )
            successes, trials = int(cur_fam.sum()), len(cur_fam)
            outcome = rate_evalue(
                successes,
                trials,
                prior,
                gamma=gamma_i,
                grid=grid,
                frozen_reference=(baseline_name == "golden"),
            )
            upd = update_process(
                state,
                outcome,
                fingerprint=fingerprint,
                alpha_prime=ac.alpha_prime,
                successes=successes,
                trials=trials,
                reference=(prior.reference_successes, prior.reference_trials),
            )
            if upd.reset_notice:
                resets.append(f"{key}: {upd.reset_notice}")
            updated.append(upd.state)
            log_evalues.append(upd.state.log_wealth)
            rows.append({"key": key, "state": upd.state, "detail": outcome.detail})

    decision = ebh_from_logs(log_evalues, alpha=ac.alpha_prime)
    processes = [
        ProcessReport(
            key=r["key"],
            log_wealth=r["state"].log_wealth,
            evidence=float(min(r["state"].log_wealth, 700.0)),
            epoch=r["state"].epoch,
            cycles=r["state"].cycles,
            bets_placed=r["state"].bets_placed,
            rise_cycle=r["state"].rise_cycle,
            crossed_at=r["state"].crossed_at,
            rejected=bool(decision.rejected[i] and not degraded),
            detail=r["detail"],
        )
        for i, r in enumerate(rows)
    ]

    ts = datetime.now(timezone.utc).isoformat()
    save_states(store, updated, ts)
    n_alerts = sum(1 for p in processes if p.rejected)
    verdict = "DEGRADED DATA" if degraded else ("DRIFT DETECTED" if n_alerts else "OK")

    result = AnytimeCheckResult(
        ts=ts,
        current_cycle=current,
        fingerprint=fingerprint,
        alpha=ac.alpha,
        alpha_prime=ac.alpha_prime,
        gamma_total=ac.gamma_total,
        gamma_per_process=gamma_i,
        n_processes=len(processes),
        verdict=verdict,
        degraded=degraded,
        processes=processes,
        resets=resets,
    )
    _persist(store, result)
    return result


def _persist(store: Store, result: AnytimeCheckResult) -> None:
    conn = store.connect()
    conn.execute(
        "INSERT INTO checks (ts, baseline_kind, params_json, verdict) VALUES (?, ?, ?, ?)",
        (
            result.ts,
            "anytime",
            json.dumps(
                {
                    "current_cycle": result.current_cycle,
                    "fingerprint": result.fingerprint,
                    "alpha": result.alpha,
                    "alpha_prime": result.alpha_prime,
                    "gamma_total": result.gamma_total,
                    "gamma_per_process": result.gamma_per_process,
                    "n_processes": result.n_processes,
                    "ville_threshold_log": ville_threshold(result.alpha_prime),
                }
            ),
            result.verdict,
        ),
    )
    conn.commit()


def wealth_table(result: AnytimeCheckResult) -> pd.DataFrame:
    """Per-process wealth as a frame, for reports and notebooks."""
    return pd.DataFrame(
        [
            {
                "process": p.label,
                "log_wealth": round(p.log_wealth, 3),
                "epoch": p.epoch,
                "cycles": p.cycles,
                "bets": p.bets_placed,
                "onset_cycle": p.rise_cycle,
                "crossed_at": p.crossed_at,
                "alert": p.rejected,
            }
            for p in sorted(result.processes, key=lambda x: -x.log_wealth)
        ]
    )
