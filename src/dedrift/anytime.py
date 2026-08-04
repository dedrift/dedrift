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

Why only the golden baseline
----------------------------
The bound above needs each process's nuisance interval to be a *single
fixed event*, settled once and then held. A frozen golden baseline gives
exactly that: the reference counts never change, so one Clopper-Pearson
interval at level ``gamma_i`` covers or does not, once, for the whole
epoch.

A *rolling* reference is recomputed every cycle, so its interval is a new
event each cycle and the union bound over a horizon of ``T`` cycles is
``T * gamma_i``, not ``gamma_i``. At the shipped default and ``T = 2000``
that is not a guarantee at all. Honouring it needs a time-uniform interval
-- a confidence sequence -- which this package does not yet have. Until it
does, rolling processes are **excluded from the anytime pool** and
:func:`dedrift.evalues.rates.rate_evalue` refuses to build a bet on a
non-frozen reference rather than labelling one as if it were covered.
Rolling comparisons remain fully supported on the fixed-sample path, which
is where they were always adjudicated.

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

import numpy as np
import pandas as pd

from dedrift.check import DEGRADED_ERROR_FRACTION, get_golden_baseline
from dedrift.config import ProjectConfig
from dedrift.evalues import (
    EProcessState,
    PriorState,
    epoch_fingerprint,
    geometric_allocation,
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

#: The only baseline the anytime path admits. A rolling reference lacks a
#: time-uniform interval (module docstring); until a confidence sequence
#: exists, betting against it would state a guarantee we cannot deliver.
ANYTIME_BASELINE = "golden"

#: Verdict when the epoch has no frozen baseline to bet against. Distinct
#: from ``OK`` on purpose: an empty pool means nothing was monitored, and
#: an operator reading ``OK`` would conclude the opposite.
NO_BASELINE_VERDICT = "NO GOLDEN BASELINE"


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

    @property
    def evalue_capped(self) -> float:
        """``E`` on the natural scale, capped so reports never print ``inf``.

        Wealth overflows float64 well before it stops being interesting, so
        the decision rule runs in logs; this exists only for display.
        """
        return float(np.exp(min(self.log_wealth, 700.0)))


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
        n_processes: Processes in the epoch's declared pool.
        pool_declared_now: True when this check declared the pool, i.e. it is
            the epoch's first. Worth surfacing: membership is frozen here, so
            a signature with no reference data at this moment waits for the
            next epoch to join.
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
    pool_declared_now: bool
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


Key = tuple[str, str, str, str]


def load_pool(store: Store, fingerprint: str) -> list[Key]:
    """Return the pool declared for this epoch, or empty if undeclared."""
    conn = store.connect()
    rows = conn.execute(
        "SELECT baseline, family, signature, channel FROM epoch_pool "
        "WHERE fingerprint = ? ORDER BY baseline, family, signature, channel",
        (fingerprint,),
    )
    return [(b, f, s, c) for b, f, s, c in rows]


def declare_pool(store: Store, fingerprint: str, keys: list[Key], ts: str) -> None:
    """Freeze the pool for this epoch. Called once, on the epoch's first check."""
    conn = store.connect()
    conn.executemany(
        "INSERT OR REPLACE INTO epoch_pool "
        "(fingerprint, baseline, family, signature, channel, declared_ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(fingerprint, *k, ts) for k in keys],
    )
    conn.commit()


def live_keys(
    frame: pd.DataFrame,
    families: list[str],
    baselines: list[tuple[str, list[str]]],
) -> list[Key]:
    """Combinations with usable REFERENCE data — the admissible pool.

    A signature with no reference observations (an ``exact_match`` column
    that is entirely absent because the suite declares no expected answers,
    say) cannot ever produce evidence, yet including it shrinks every other
    process's coverage budget and raises the e-BH threshold. Excluding it is
    pure gain.

    The test uses reference data only. Current-cycle availability is *not*
    consulted: that would make pool membership depend on the very cycle
    being bet on, which is the predictability violation this package is
    built to prevent. The consequence is stated rather than hidden — a
    signature that gains data mid-epoch waits until the next epoch to join,
    and one that loses data stays in the pool contributing ``E_t = 1``.
    """
    keys: list[Key] = []
    for baseline_name, ref_cycles in baselines:
        if not ref_cycles:
            continue
        ref_frame = frame[frame["cycle_id"].isin(ref_cycles)]
        for family in families:
            fam = ref_frame[ref_frame["family"] == family]
            for sig in RATE_SIGNATURES:
                if len(fam[sig].dropna()) > 0:
                    keys.append((baseline_name, family, sig, "rate"))
    return sorted(keys)


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

    fingerprint = epoch_fingerprint(
        suite_version=str(len({r.canary_id for r in records if r.canary_id})),
        embedder=cfg.embedder,
        golden_cycles=tuple(golden),
        extractor_version=EXTRACTOR_VERSION,
    )

    cur_frame = frame[frame["cycle_id"] == current]
    degraded = bool(cur_frame["had_error"].mean() > DEGRADED_ERROR_FRACTION)
    families = sorted(frame["family"].unique())

    # The pool is declared ONCE per epoch and then frozen. Its size sets the
    # per-process coverage budget, which sets the nuisance interval, which is
    # part of the bet — and for a frozen baseline the guarantee needs that
    # interval to be a single fixed event settled at epoch start. A pool
    # recomputed per cycle would quietly turn it into a sequence of
    # different events.
    ts = datetime.now(timezone.utc).isoformat()
    # Golden only: a rolling reference has no time-uniform interval, so its
    # coverage event is fresh every cycle and the budget arithmetic above
    # does not hold for it. See the module docstring.
    baselines = [(ANYTIME_BASELINE, golden)]
    pool_keys = load_pool(store, fingerprint)
    pool_declared_now = False
    if not pool_keys:
        pool_keys = live_keys(frame, families, baselines)
        declare_pool(store, fingerprint, pool_keys, ts)
        pool_declared_now = True
    ref_by_baseline = dict(baselines)
    gamma_i = per_process_gamma(ac.gamma_total, max(len(pool_keys), 1))
    grid = symmetric_grid(ac.tilts)

    states = load_states(store)
    updated: list[EProcessState] = []
    log_evalues: list[float] = []
    rows: list[dict[str, Any]] = []
    resets: list[str] = []

    for key in pool_keys:
        baseline_name, family, sig, _channel = key
        ref_cycles = ref_by_baseline.get(baseline_name, [])
        ref_frame = frame[frame["cycle_id"].isin(ref_cycles)]
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
            frozen_reference=(baseline_name == ANYTIME_BASELINE),
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

    # Epoch allocation. "per_epoch" spends alpha' afresh in every epoch --
    # the honest default, since a fingerprint change makes it a different
    # null. "geometric" is for operators who want a bound over an unbounded
    # number of epochs: alpha'_e = alpha' * 2^-(e+1), which is summable.
    # Validated rather than silently ignored: this key was parsed and never
    # read for a release, so setting it did nothing.
    epoch_now = max((s.epoch for s in updated), default=0)
    if ac.epoch_allocation == "geometric":
        effective_alpha_prime = geometric_allocation(ac.alpha_prime, epoch_now)
    elif ac.epoch_allocation == "per_epoch":
        effective_alpha_prime = ac.alpha_prime
    else:
        msg = (
            f"unknown anytime.epoch_allocation {ac.epoch_allocation!r}; "
            "expected 'per_epoch' or 'geometric'"
        )
        raise ValueError(msg)

    decision = ebh_from_logs(log_evalues, alpha=effective_alpha_prime)
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

    save_states(store, updated, ts)
    n_alerts = sum(1 for p in processes if p.rejected)
    if not pool_keys:
        verdict = NO_BASELINE_VERDICT
    elif degraded:
        verdict = "DEGRADED DATA"
    else:
        verdict = "DRIFT DETECTED" if n_alerts else "OK"

    result = AnytimeCheckResult(
        ts=ts,
        current_cycle=current,
        fingerprint=fingerprint,
        alpha=ac.alpha,
        alpha_prime=effective_alpha_prime,
        gamma_total=ac.gamma_total,
        gamma_per_process=gamma_i,
        n_processes=len(processes),
        pool_declared_now=pool_declared_now,
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
                    "pool_declared_now": result.pool_declared_now,
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
