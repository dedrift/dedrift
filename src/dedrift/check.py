"""The drift check pipeline: dual baselines, test battery, FDR, materiality.

Gating order (SPEC.md §6 — order matters):

1. Run all tests, collect p-values (PSI and Page-Hinkley produce flags,
   never p-values).
2. Benjamini-Hochberg FDR at ``q`` across ALL p-valued tests in the check
   (both baselines together — one multiplicity family per check).
3. Survivors pass the materiality gate (per-channel effect thresholds).
4. Only tests passing BOTH become alerts. Everything else is reported as
   "observed, below materiality" or "not significant".

Every check compares the current cycle against BOTH the rolling reference
(sudden breaks) and the frozen golden baseline (boiling-frog drift) and
reports both verdicts (principle 6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dedrift.config import ProjectConfig
from dedrift.detectors import (
    PageHinkleyResult,
    TestOutcome,
    ad_test,
    benjamini_hochberg,
    bootstrap_p95_test,
    ks_test,
    levene_test,
    page_hinkley,
    psi,
    two_proportion_z_test,
    welch_t_test,
)
from dedrift.signatures import signatures_frame
from dedrift.signatures.structural import RATE_SIGNATURES, SCALAR_SIGNATURES
from dedrift.store import Store

BASELINE_FILE = "baseline.json"

#: Fraction of errored records in the current cycle above which the check
#: refuses to interpret drift and reports DEGRADED DATA instead.
DEGRADED_ERROR_FRACTION = 0.2


@dataclass(frozen=True)
class TestRecord:
    """One executed test with its gating outcome.

    Attributes:
        baseline: ``"rolling"`` (sudden) or ``"golden"`` (cumulative).
        family: Canary family.
        signature: Signature name.
        outcome: The statistical outcome.
        p_adjusted: BH-adjusted p-value (NaN if the raw p was NaN).
        significant: True if BH rejected at q.
        material: True if the effect exceeds the materiality gate.
        alert: significant AND material.
    """

    baseline: str
    family: str
    signature: str
    outcome: TestOutcome
    p_adjusted: float = float("nan")
    significant: bool = False
    material: bool = False
    alert: bool = False


@dataclass(frozen=True)
class FlagRecord:
    """A non-p-valued indicator (PSI or Page-Hinkley), reported as a flag.

    Attributes:
        kind: ``"psi"`` or ``"page_hinkley"``.
        family: Canary family.
        signature: Signature name.
        value: PSI value or PH statistic.
        label: PSI label or PH direction.
        change_cycle_id: PH onset estimate (cycle ID), if alarmed.
        material: True if the corresponding effect passes materiality.
    """

    kind: str
    family: str
    signature: str
    value: float
    label: str
    change_cycle_id: str | None = None
    material: bool = False


@dataclass(frozen=True)
class CheckResult:
    """Everything one check produced (input for attribution and the report).

    Attributes:
        ts: Check timestamp (UTC, ISO).
        current_cycle: Cycle under test.
        rolling_cycles: Cycle IDs in the rolling reference.
        golden_cycles: Cycle IDs in the golden baseline.
        fdr_q: FDR level used.
        seed: Seed used for all resampling.
        verdict_sudden: ``OK`` / ``DRIFT DETECTED`` / ``NO REFERENCE``.
        verdict_cumulative: Same, against the golden baseline.
        degraded: True if the current cycle's error fraction was too high
            for drift interpretation (overall verdict DEGRADED DATA).
        tests: All executed tests with gating outcomes.
        flags: PSI / Page-Hinkley indicators.
        n_alerts: Convenience count of alerting tests.
    """

    ts: str
    current_cycle: str
    rolling_cycles: list[str]
    golden_cycles: list[str]
    fdr_q: float
    seed: int
    verdict_sudden: str
    verdict_cumulative: str
    degraded: bool
    tests: list[TestRecord] = field(default_factory=list)
    flags: list[FlagRecord] = field(default_factory=list)

    @property
    def n_alerts(self) -> int:
        """Number of alerting tests."""
        return sum(1 for t in self.tests if t.alert)

    def alerts(self) -> list[TestRecord]:
        """Alerting tests, deterministically ordered."""
        return sorted(
            (t for t in self.tests if t.alert),
            key=lambda t: (t.baseline, t.family, t.signature, t.outcome.test),
        )


# -- baseline management -------------------------------------------------------


def set_golden_baseline(store: Store, cycle_ids: list[str]) -> None:
    """Freeze the golden baseline to the given cycles (never auto-updated).

    Args:
        store: The project store.
        cycle_ids: Cycles the owner declares known-good.
    """
    path = store.project_dir / BASELINE_FILE
    path.write_text(json.dumps({"golden_cycles": sorted(cycle_ids)}, indent=2), encoding="utf-8")


def get_golden_baseline(store: Store) -> list[str]:
    """Return the frozen golden cycle IDs ([] if never set)."""
    path = store.project_dir / BASELINE_FILE
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("golden_cycles", []))


# -- pipeline ------------------------------------------------------------------


def _ordered_cycles(frame: pd.DataFrame) -> list[str]:
    """Cycle IDs ordered by each cycle's first record (execution order)."""
    firsts = frame.groupby("cycle_id")["record_id"].min()
    order = frame.drop_duplicates("cycle_id").set_index("cycle_id").index
    # Order by appearance in the frame (records are stored in append order).
    seen: list[str] = []
    for cid in frame["cycle_id"]:
        if cid not in seen:
            seen.append(cid)
    del firsts, order
    return seen


def _material_scalar(test: TestOutcome, cfg: ProjectConfig) -> bool:
    m = cfg.materiality
    if test.test == "levene":
        r = test.effect_size
        return bool(r >= m.variance_ratio or (r > 0 and r <= 1 / m.variance_ratio))
    if test.test == "p95_boot":
        return bool(abs(test.effect_size) >= m.p95_relative)
    return bool(abs(test.effect_size) >= m.scalar_cohen_d)


def run_check(store: Store, config: ProjectConfig | None = None) -> CheckResult:
    """Run the full gated drift check for the latest cycle.

    Args:
        store: The project store (must contain canary records with cycles).
        config: Project configuration; loaded from the store when omitted.

    Returns:
        The complete check result. Also persisted to the ``checks`` and
        ``alerts`` tables.

    Raises:
        ValueError: If no canary cycles exist.
    """
    cfg = config or ProjectConfig.load(store.project_dir)
    records = [r for r in store.read_records() if r.cycle_id is not None]
    if not records:
        msg = "no canary records with cycle IDs; run canaries first"
        raise ValueError(msg)
    frame = signatures_frame(records)
    cycles = _ordered_cycles(frame)
    current = cycles[-1]
    history = cycles[:-1]
    golden = [c for c in get_golden_baseline(store) if c in cycles and c != current]
    rolling = [c for c in history if c not in golden][-cfg.rolling_window_cycles :]

    cur_frame = frame[frame["cycle_id"] == current]
    degraded = bool(cur_frame["had_error"].mean() > DEGRADED_ERROR_FRACTION)

    tests: list[TestRecord] = []
    flags: list[FlagRecord] = []
    families = sorted(frame["family"].unique())

    def battery(baseline_name: str, ref_frame: pd.DataFrame) -> list[TestRecord]:
        out: list[TestRecord] = []
        for family in families:
            ref_fam = ref_frame[ref_frame["family"] == family]
            cur_fam = cur_frame[cur_frame["family"] == family]
            if ref_fam.empty or cur_fam.empty:
                continue
            for sig in SCALAR_SIGNATURES:
                ref = ref_fam[sig].to_numpy(dtype=float)
                cur = cur_fam[sig].to_numpy(dtype=float)
                for fn in (ks_test, ad_test, welch_t_test, levene_test):
                    out.append(TestRecord(baseline_name, family, sig, fn(ref, cur)))
                out.append(
                    TestRecord(
                        baseline_name,
                        family,
                        sig,
                        bootstrap_p95_test(ref, cur, n_boot=cfg.permutations, seed=cfg.seed),
                    )
                )
            for sig in RATE_SIGNATURES:
                ref_series = ref_fam[sig].dropna()
                cur_series = cur_fam[sig].dropna()
                if ref_series.empty or cur_series.empty:
                    continue
                out.append(
                    TestRecord(
                        baseline_name,
                        family,
                        sig,
                        two_proportion_z_test(
                            int(ref_series.sum()),
                            len(ref_series),
                            int(cur_series.sum()),
                            len(cur_series),
                        ),
                    )
                )
        return out

    verdicts: dict[str, str] = {}
    for name, ref_cycles in (("rolling", rolling), ("golden", golden)):
        if not ref_cycles:
            verdicts[name] = "NO REFERENCE"
            continue
        ref_frame = frame[frame["cycle_id"].isin(ref_cycles)]
        tests.extend(battery(name, ref_frame))
        verdicts[name] = "pending"

    # Gate 2: one BH family across every p-valued test in this check.
    rejected, adjusted = benjamini_hochberg([t.outcome.p_value for t in tests], q=cfg.fdr_q)

    # Gates 3-4: materiality, then alert = significant AND material.
    gated: list[TestRecord] = []
    for t, rej, adj in zip(tests, rejected, adjusted, strict=True):
        if t.signature in RATE_SIGNATURES:
            material = abs(t.outcome.effect_raw) >= cfg.materiality.rate_threshold(t.signature)
        else:
            material = _material_scalar(t.outcome, cfg)
        gated.append(
            TestRecord(
                baseline=t.baseline,
                family=t.family,
                signature=t.signature,
                outcome=t.outcome,
                p_adjusted=adj,
                significant=rej,
                material=material,
                alert=bool(rej and material and not degraded),
            )
        )

    # Flags: PSI (golden bins) and Page-Hinkley (per-cycle means over history).
    if golden:
        golden_frame = frame[frame["cycle_id"].isin(golden)]
        for family in families:
            g_fam = golden_frame[golden_frame["family"] == family]
            c_fam = cur_frame[cur_frame["family"] == family]
            if g_fam.empty or c_fam.empty:
                continue
            for sig in SCALAR_SIGNATURES:
                res = psi(g_fam[sig].to_numpy(dtype=float), c_fam[sig].to_numpy(dtype=float))
                if res.label != "stable":
                    flags.append(FlagRecord("psi", family, sig, res.value, res.label, None, True))
    for family in families:
        fam_frame = frame[frame["family"] == family]
        for sig in SCALAR_SIGNATURES:
            means = fam_frame.groupby("cycle_id")[sig].mean().reindex(cycles).to_numpy()
            ph: PageHinkleyResult = page_hinkley(means, lambda_=cfg.ph_lambda, delta=cfg.ph_delta)
            if ph.alarm:
                onset = cycles[ph.change_index] if ph.change_index is not None else None
                material = any(
                    g.material for g in gated if g.family == family and g.signature == sig
                )
                flags.append(
                    FlagRecord(
                        "page_hinkley", family, sig, ph.statistic, ph.direction, onset, material
                    )
                )

    for name in ("rolling", "golden"):
        if verdicts.get(name) == "pending":
            any_alert = any(t.alert for t in gated if t.baseline == name)
            verdicts[name] = "DRIFT DETECTED" if any_alert else "OK"

    result = CheckResult(
        ts=datetime.now(timezone.utc).isoformat(),
        current_cycle=current,
        rolling_cycles=rolling,
        golden_cycles=golden,
        fdr_q=cfg.fdr_q,
        seed=cfg.seed,
        verdict_sudden=verdicts["rolling"],
        verdict_cumulative=verdicts["golden"],
        degraded=degraded,
        tests=gated,
        flags=sorted(flags, key=lambda f: (f.kind, f.family, f.signature)),
    )
    _persist(store, result)
    return result


def _persist(store: Store, result: CheckResult) -> None:
    conn = store.connect()
    cursor = conn.execute(
        "INSERT INTO checks (ts, baseline_kind, params_json, verdict) VALUES (?, ?, ?, ?)",
        (
            result.ts,
            "dual",
            json.dumps(
                {
                    "current_cycle": result.current_cycle,
                    "rolling": result.rolling_cycles,
                    "golden": result.golden_cycles,
                    "fdr_q": result.fdr_q,
                    "seed": result.seed,
                }
            ),
            f"sudden={result.verdict_sudden};cumulative={result.verdict_cumulative}"
            + (";DEGRADED" if result.degraded else ""),
        ),
    )
    check_id = cursor.lastrowid
    for t in result.alerts():
        conn.execute(
            "INSERT INTO alerts (check_id, signature, family, test, p_adjusted,"
            " effect_size, effect_units, details_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                check_id,
                t.signature,
                t.family,
                t.outcome.test,
                t.p_adjusted,
                t.outcome.effect_size,
                "cohen_d" if t.signature in SCALAR_SIGNATURES else "rate_diff",
                json.dumps(
                    {
                        "baseline": t.baseline,
                        "statistic": t.outcome.statistic,
                        "effect_raw": t.outcome.effect_raw,
                        "n_ref": t.outcome.n_ref,
                        "n_cur": t.outcome.n_cur,
                    }
                ),
            ),
        )
    conn.commit()


def export_baseline_path(store: Store) -> Path:
    """Path of the golden-baseline file (for docs/tests)."""
    return store.project_dir / BASELINE_FILE
