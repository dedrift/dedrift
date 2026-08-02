"""The drift check pipeline: dual baselines, test battery, FDR, materiality.

Gating order (SPEC.md §6 — order matters):

1. Run all tests, collect p-values (PSI and Page-Hinkley produce flags,
   never p-values).
2. Benjamini-Hochberg FDR at ``q`` across the PRIMARY tests in the check
   (both baselines together — one multiplicity family per check). One
   primary per channel: KS for location/shape, Levene for dispersion, the
   P95 permutation test for tails, two-proportion z for rates, MMD for
   semantics. Anderson-Darling and Welch run as CORROBORATION only — they
   ask the same question as KS on the same data, so admitting them to the
   pool would roughly double m (and halve every BH threshold) at no
   informational gain. Corroboration tests never alert.
3. Survivors pass the materiality gate (per-channel effect thresholds; KS
   gates on the KS statistic D — the sup-norm CDF distance — because a
   shape change with equal means has Cohen's d ~ 0 and is still real drift).
4. Only tests passing BOTH become alerts. Everything else is reported as
   "observed, below materiality", "not significant", or "corroboration".

Every check compares the current cycle against BOTH the rolling reference
(sudden breaks) and the frozen golden baseline (boiling-frog drift) and
reports both verdicts (principle 6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from dedrift.config import ProjectConfig
from dedrift.detectors import (
    PageHinkleyResult,
    TestOutcome,
    ad_test,
    benjamini_hochberg,
    calibrate_mmd_floor,
    ks_test,
    levene_test,
    mmd_rbf_test,
    p95_permutation_test,
    page_hinkley,
    psi,
    two_proportion_z_test,
    welch_t_test,
)
from dedrift.embeddings import embed_records, get_pinned_embedder
from dedrift.schema import InteractionRecord
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
        primary: True if this test enters the BH pool and can alert;
            False for corroboration tests (AD, Welch).
        p_adjusted: BH-adjusted p-value (NaN if the raw p was NaN or the
            test is corroboration-only).
        significant: True if BH rejected at q (primaries only).
        material: True if the effect exceeds the materiality gate.
        alert: significant AND material (primaries only).
    """

    baseline: str
    family: str
    signature: str
    outcome: TestOutcome
    primary: bool = True
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
    """Cycle IDs by first appearance in the frame (records are append-ordered)."""
    seen: list[str] = []
    for cid in frame["cycle_id"]:
        if cid not in seen:
            seen.append(cid)
    return seen


def _material_scalar(test: TestOutcome, cfg: ProjectConfig) -> bool:
    m = cfg.materiality
    if test.test == "levene":
        r = test.effect_size
        return bool(r >= m.variance_ratio or (r > 0 and r <= 1 / m.variance_ratio))
    if test.test == "p95_perm":
        return bool(abs(test.effect_size) >= m.p95_relative)
    if test.test in ("ks", "ad"):
        # KS detects ANY distributional change; a shape shift with equal
        # means has Cohen's d ~ 0 and is still real drift. Gate on the KS
        # statistic D (sup-norm CDF distance); d stays a reported diagnostic.
        return bool(test.statistic >= m.ks_distance)
    return bool(abs(test.effect_size) >= m.scalar_cohen_d)


def _add_semantic_displacement(
    frame: pd.DataFrame,
    records: list[InteractionRecord],
    embeddings: dict[str, npt.NDArray[np.float64]],
    golden_cycles: list[str],
) -> None:
    """Add a per-record semantic_displacement column to the signature frame.

    Displacement = cosine distance from each record's embedding to the
    centroid of the SAME canary's reference outputs (golden cycles when set,
    else all cycles — mildly contaminated by the current cycle, which the
    docs note as a reason to set a golden baseline). A scalar per record, so
    it flows through the ordinary scalar battery — KS/AD/Levene/P95 apply.
    """
    ref_cycle_set = set(golden_cycles)
    by_canary: dict[str, list[npt.NDArray[np.float64]]] = {}
    for r in records:
        in_ref = r.cycle_id in ref_cycle_set if ref_cycle_set else True
        if r.canary_id is not None and in_ref:
            by_canary.setdefault(r.canary_id, []).append(embeddings[r.id])
    centroids = {c: np.mean(np.stack(v), axis=0) for c, v in by_canary.items() if v}

    def displacement(record_id: str, canary_id: str | None) -> float:
        centroid = centroids.get(canary_id or "")
        if centroid is None:
            return float("nan")
        vec = embeddings[record_id]
        denom = float(np.linalg.norm(vec) * np.linalg.norm(centroid))
        if denom == 0:
            return 1.0
        return float(1.0 - float(vec @ centroid) / denom)

    id_to_canary = {r.id: r.canary_id for r in records}
    frame["semantic_displacement"] = [
        displacement(rid, id_to_canary.get(rid)) for rid in frame["record_id"]
    ]


def _mmd_battery(
    baseline_name: str,
    ref_cycles: list[str],
    current: str,
    frame: pd.DataFrame,
    embeddings: dict[str, npt.NDArray[np.float64]],
    families: list[str],
    cfg: ProjectConfig,
    mmd_floors: dict[tuple[str, str], float],
) -> list[TestRecord]:
    """Run MMD per family against one baseline; record calibrated floors.

    The materiality floor per (baseline, family) is the config override when
    non-negative, else the 95th percentile of MMD^2 between pairs of the
    baseline's own cycles (known-same distribution). With fewer than three
    reference cycles the floor is 0.0 (uncalibratable; reported as such).
    """
    out: list[TestRecord] = []
    for family in families:
        fam = frame[frame["family"] == family]
        ref_ids = fam[fam["cycle_id"].isin(ref_cycles)]["record_id"]
        cur_ids = fam[fam["cycle_id"] == current]["record_id"]
        if len(ref_ids) < 2 or len(cur_ids) < 2:
            continue
        ref_emb = np.stack([embeddings[i] for i in ref_ids])
        cur_emb = np.stack([embeddings[i] for i in cur_ids])
        # One bandwidth per (family, baseline): pooled median heuristic over
        # ref+cur, used for BOTH the observed statistic and the floor so the
        # two are commensurable under one kernel.
        from dedrift.detectors.mmd import median_heuristic_bandwidth

        sigma = median_heuristic_bandwidth(np.vstack([ref_emb, cur_emb]))
        outcome = mmd_rbf_test(
            ref_emb,
            cur_emb,
            n_permutations=max(cfg.permutations, 500),
            seed=cfg.seed,
            sigma=sigma,
        )
        if cfg.materiality.embedding_mmd2_floor >= 0:
            floor = cfg.materiality.embedding_mmd2_floor
        else:
            per_cycle = [
                np.stack([embeddings[i] for i in fam[fam["cycle_id"] == cycle]["record_id"]])
                for cycle in ref_cycles
            ]
            floor = calibrate_mmd_floor([p for p in per_cycle if len(p) >= 2], sigma=sigma)
        mmd_floors[(baseline_name, family)] = floor
        out.append(TestRecord(baseline_name, family, "embedding", outcome))
    return out


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

    # Tier 2 (optional): embeddings via the pinned embedder. When pinned, a
    # per-record semantic_displacement scalar joins the ordinary scalar
    # battery, and MMD tests per family join the same BH pool. The MMD^2
    # materiality floor is auto-calibrated per (family, baseline) from
    # reference-cycle pairs (known-same distribution) unless overridden.
    embeddings: dict[str, Any] | None = None
    if get_pinned_embedder(store) is not None:
        embeddings = embed_records(store, records)
        _add_semantic_displacement(frame, records, embeddings, get_golden_baseline(store))

    scalar_signatures = tuple(SCALAR_SIGNATURES) + (
        ("semantic_displacement",) if embeddings is not None else ()
    )

    cur_frame = frame[frame["cycle_id"] == current]
    degraded = bool(cur_frame["had_error"].mean() > DEGRADED_ERROR_FRACTION)

    tests: list[TestRecord] = []
    flags: list[FlagRecord] = []
    mmd_floors: dict[tuple[str, str], float] = {}
    families = sorted(frame["family"].unique())

    def battery(baseline_name: str, ref_frame: pd.DataFrame) -> list[TestRecord]:
        out: list[TestRecord] = []
        for family in families:
            ref_fam = ref_frame[ref_frame["family"] == family]
            cur_fam = cur_frame[cur_frame["family"] == family]
            if ref_fam.empty or cur_fam.empty:
                continue
            for sig in scalar_signatures:
                ref = ref_fam[sig].dropna().to_numpy(dtype=float)
                cur = cur_fam[sig].dropna().to_numpy(dtype=float)
                # Primaries (enter BH, may alert): KS, Levene, P95 permutation.
                out.append(TestRecord(baseline_name, family, sig, ks_test(ref, cur)))
                out.append(TestRecord(baseline_name, family, sig, levene_test(ref, cur)))
                out.append(
                    TestRecord(
                        baseline_name,
                        family,
                        sig,
                        p95_permutation_test(
                            ref, cur, n_permutations=cfg.permutations, seed=cfg.seed
                        ),
                    )
                )
                # Corroboration only (same location hypothesis as KS on the
                # same data; admitting them would inflate m at no gain).
                out.append(TestRecord(baseline_name, family, sig, ad_test(ref, cur), primary=False))
                out.append(
                    TestRecord(baseline_name, family, sig, welch_t_test(ref, cur), primary=False)
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
        if embeddings is not None:
            tests.extend(
                _mmd_battery(
                    name, ref_cycles, current, frame, embeddings, families, cfg, mmd_floors
                )
            )
        verdicts[name] = "pending"

    # Gate 2: one BH family across the PRIMARY tests in this check.
    primary_idx = [i for i, t in enumerate(tests) if t.primary]
    rejected_p, adjusted_p = benjamini_hochberg(
        [tests[i].outcome.p_value for i in primary_idx], q=cfg.fdr_q
    )
    rejected = [False] * len(tests)
    adjusted = [float("nan")] * len(tests)
    for pos, i in enumerate(primary_idx):
        rejected[i] = rejected_p[pos]
        adjusted[i] = adjusted_p[pos]

    # Gates 3-4: materiality, then alert = primary AND significant AND material.
    gated: list[TestRecord] = []
    for t, rej, adj in zip(tests, rejected, adjusted, strict=True):
        if t.outcome.test == "mmd":
            material = t.outcome.effect_size >= mmd_floors.get((t.baseline, t.family), 0.0)
        elif t.signature in RATE_SIGNATURES:
            material = abs(t.outcome.effect_raw) >= cfg.materiality.rate_threshold(t.signature)
        else:
            material = _material_scalar(t.outcome, cfg)
        gated.append(
            TestRecord(
                baseline=t.baseline,
                family=t.family,
                signature=t.signature,
                outcome=t.outcome,
                primary=t.primary,
                p_adjusted=adj,
                significant=rej,
                material=material,
                alert=bool(t.primary and rej and material and not degraded),
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
