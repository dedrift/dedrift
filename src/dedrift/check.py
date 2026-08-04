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

import hashlib
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
    psi_null_expectation,
    two_proportion_z_test,
    welch_t_test,
)
from dedrift.detectors.heuristic import PSI_MODERATE
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
        material: True if the effect exceeds the materiality gate; None for
            corroboration tests — materiality gates are defined on primary
            effect scales only (the AD statistic, for instance, is not a
            sup-norm distance and has no meaningful threshold here).
        alert: significant AND material (primaries only).
    """

    baseline: str
    family: str
    signature: str
    outcome: TestOutcome
    primary: bool = True
    p_adjusted: float = float("nan")
    significant: bool = False
    material: bool | None = False
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
class CompositionIssue:
    """A violated balanced-design assumption for one (baseline, family).

    The two-sample tests require both windows to contain the SAME canaries
    at uniform repetition counts. That is necessary for the comparison to
    mean anything; it is not sufficient for exchangeability (see
    :mod:`dedrift.detectors.scalar`). If a
    canary's records vanish from one window — a timeout, a partial run, a
    suite edit — the family's mixture shifts and KS would fire on a
    missing-data artifact, not drift. The check therefore SUPPRESSES the
    family's comparison against that baseline and reports the issue instead.

    Attributes:
        baseline: ``"rolling"`` or ``"golden"``.
        family: Canary family whose comparison was suppressed.
        missing_canaries: In the reference window but not the current cycle.
        extra_canaries: In the current cycle but not the reference window.
        unbalanced: True if repetition counts differ across canaries within
            a window (mixture weights shifted even with identical sets).
        detail: Human-readable explanation for the report.
    """

    baseline: str
    family: str
    missing_canaries: tuple[str, ...]
    extra_canaries: tuple[str, ...]
    unbalanced: bool
    detail: str


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
        composition_issues: Suppressed (baseline, family) comparisons whose
            windows were not composition-comparable.
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
    composition_issues: list[CompositionIssue] = field(default_factory=list)

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


def _test_seed(base_seed: int, *parts: str) -> int:
    """Deterministic per-test seed derived from the project seed and identity.

    Every permutation test in a check used to receive ``cfg.seed`` verbatim,
    so the ~170 permutation nulls in one BH pool were coupled by common
    random numbers. Each p-value was individually exact, but their joint
    behaviour carried a dependence that came from our seeding rather than
    from the data -- an avoidable contribution to exactly the multiplicity
    question the paper is careful about elsewhere.

    Deriving the seed from ``(project seed, baseline, family, signature,
    test)`` keeps the run bit-for-bit reproducible while decoupling the
    nulls. BLAKE2b is used rather than :func:`hash` because the latter is
    salted per process and would break reproducibility.

    Args:
        base_seed: The project-level seed.
        *parts: Identity of the test (baseline, family, signature, name).

    Returns:
        A seed in ``[0, 2**31)``.
    """
    key = "|".join((str(base_seed), *parts)).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") % (2**31)


def _material_scalar(test: TestOutcome, cfg: ProjectConfig) -> bool:
    m = cfg.materiality
    if test.test == "levene":
        r = test.effect_size
        return bool(r >= m.dispersion_ratio or (r > 0 and r <= 1 / m.dispersion_ratio))
    if test.test == "p95_perm":
        return bool(abs(test.effect_size) >= m.p95_relative)
    if test.test == "ks":
        # KS detects ANY distributional change; a shape shift with equal
        # means has Cohen's d ~ 0 and is still real drift. Gate on the KS
        # statistic D (sup-norm CDF distance), which is also the reported
        # effect for this channel. NOTE on binding scale: the raw-alpha=0.05
        # critical D is ~1.36*sqrt((n+m)/(n*m)), which for equal arms drops
        # below the 0.15 default only at n >~ 165 per arm — below that (and
        # always after BH tightening) significance is the stricter filter
        # and this gate is a large-n guard against trivially significant D,
        # not the operative filter. Documented in docs/statistics.md.
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

    Reference records are scored **leave-one-out**
    ---------------------------------------------
    A reference record helped define its own centroid, so scoring it against
    the full mean measures an in-sample distance while a current-cycle
    record is measured out-of-sample. With ``n`` reference embeddings per
    canary the null expectations differ by a factor of roughly
    ``(1 + 1/n) / (1 - 1/n)`` — about 1.2x at the default ``n = 12`` — which
    inflates the current window's location *and* dispersion on completely
    unchanged data. KS, Brown-Forsythe and the P95 permutation test on this
    column would all be anticonservative by construction.

    Removing the record from its own centroid restores the comparison: both
    windows are then scored against a centroid that excludes them. For a
    reference record ``i`` of canary ``c``, the leave-one-out centroid is
    ``(n * mean_c - x_i) / (n - 1)``, computed in closed form rather than by
    recomputing ``n`` means. Canaries with a single reference record cannot
    be scored leave-one-out and yield NaN, which the battery already drops.
    """
    ref_cycle_set = set(golden_cycles)
    by_canary: dict[str, list[npt.NDArray[np.float64]]] = {}
    for r in records:
        in_ref = r.cycle_id in ref_cycle_set if ref_cycle_set else True
        if r.canary_id is not None and in_ref:
            by_canary.setdefault(r.canary_id, []).append(embeddings[r.id])
    centroids = {c: np.mean(np.stack(v), axis=0) for c, v in by_canary.items() if v}
    counts = {c: len(v) for c, v in by_canary.items() if v}
    reference_ids = {
        r.id
        for r in records
        if r.canary_id is not None and ((r.cycle_id in ref_cycle_set) if ref_cycle_set else True)
    }

    def _cosine_distance(vec: npt.NDArray[np.float64], centroid: npt.NDArray[np.float64]) -> float:
        denom = float(np.linalg.norm(vec) * np.linalg.norm(centroid))
        if denom == 0:
            return 1.0
        return float(1.0 - float(vec @ centroid) / denom)

    def displacement(record_id: str, canary_id: str | None) -> float:
        key = canary_id or ""
        centroid = centroids.get(key)
        if centroid is None:
            return float("nan")
        vec = embeddings[record_id]
        if record_id in reference_ids:
            n = counts[key]
            if n < 2:
                return float("nan")
            centroid = (n * centroid - vec) / (n - 1)
        return _cosine_distance(vec, centroid)

    id_to_canary = {r.id: r.canary_id for r in records}
    frame["semantic_displacement"] = [
        displacement(rid, id_to_canary.get(rid)) for rid in frame["record_id"]
    ]


def _composition_issue(
    baseline: str, family: str, ref_fam: pd.DataFrame, cur_fam: pd.DataFrame
) -> CompositionIssue | None:
    """Check the balanced-design assumption for one (baseline, family).

    Returns an issue when the two windows do not contain the same canaries,
    or when repetition counts are non-uniform across canaries within a
    window. Either way the family mixture differs between windows, so a
    two-sample test would measure missing data, not drift.
    """
    ref_counts = ref_fam["canary_id"].value_counts()
    cur_counts = cur_fam["canary_id"].value_counts()
    missing = tuple(sorted(set(ref_counts.index) - set(cur_counts.index)))
    extra = tuple(sorted(set(cur_counts.index) - set(ref_counts.index)))
    unbalanced = bool(
        (len(cur_counts) > 0 and cur_counts.min() != cur_counts.max())
        or (len(ref_counts) > 0 and ref_counts.min() != ref_counts.max())
    )
    if not missing and not extra and not unbalanced:
        return None
    parts: list[str] = []
    if missing:
        parts.append(f"canaries in reference but absent from current cycle: {', '.join(missing)}")
    if extra:
        parts.append(f"canaries in current cycle but absent from reference: {', '.join(extra)}")
    if unbalanced:
        parts.append(
            "repetition counts differ across canaries within a window "
            f"(current min/max {int(cur_counts.min()) if len(cur_counts) else 0}/"
            f"{int(cur_counts.max()) if len(cur_counts) else 0}, "
            f"reference min/max {int(ref_counts.min()) if len(ref_counts) else 0}/"
            f"{int(ref_counts.max()) if len(ref_counts) else 0})"
        )
    return CompositionIssue(
        baseline=baseline,
        family=family,
        missing_canaries=missing,
        extra_canaries=extra,
        unbalanced=unbalanced,
        detail="; ".join(parts),
    )


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
            seed=_test_seed(cfg.seed, baseline_name, family, "embedding", "mmd"),
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
    issues: list[CompositionIssue] = []
    mmd_floors: dict[tuple[str, str], float] = {}
    families = sorted(frame["family"].unique())

    def battery(
        baseline_name: str, ref_frame: pd.DataFrame, skip: frozenset[str]
    ) -> list[TestRecord]:
        out: list[TestRecord] = []
        for family in families:
            if family in skip:
                continue
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
                            ref,
                            cur,
                            n_permutations=cfg.permutations,
                            seed=_test_seed(cfg.seed, baseline_name, family, sig, "p95_perm"),
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
        # Composition guard (balanced-design exchangeability is CHECKED, not
        # assumed): a family whose windows differ in canary membership or
        # repetition balance is suppressed for this baseline — its
        # two-sample tests would measure missing data, not drift.
        skip: set[str] = set()
        for family in families:
            ref_fam = ref_frame[ref_frame["family"] == family]
            cur_fam = cur_frame[cur_frame["family"] == family]
            if not ref_fam.empty and cur_fam.empty:
                # An entirely vanished family would otherwise be skipped
                # silently; say why nothing was compared.
                issues.append(
                    CompositionIssue(
                        baseline=name,
                        family=family,
                        missing_canaries=tuple(sorted(ref_fam["canary_id"].dropna().unique())),
                        extra_canaries=(),
                        unbalanced=False,
                        detail="family entirely absent from the current cycle",
                    )
                )
                skip.add(family)
                continue
            if ref_fam.empty or cur_fam.empty:
                continue
            issue = _composition_issue(name, family, ref_fam, cur_fam)
            if issue is not None:
                issues.append(issue)
                skip.add(family)
        tests.extend(battery(name, ref_frame, frozenset(skip)))
        if embeddings is not None:
            ok_families = [f for f in families if f not in skip]
            tests.extend(
                _mmd_battery(
                    name, ref_cycles, current, frame, embeddings, ok_families, cfg, mmd_floors
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
    # Corroboration tests carry material=None: the gates are defined on the
    # primary effect scales, and e.g. the AD statistic is not commensurable
    # with a sup-norm CDF distance.
    gated: list[TestRecord] = []
    for t, rej, adj in zip(tests, rejected, adjusted, strict=True):
        material: bool | None
        if not t.primary:
            material = None
        elif t.outcome.test == "mmd":
            material = bool(t.outcome.effect_size >= mmd_floors.get((t.baseline, t.family), 0.0))
        elif t.signature in RATE_SIGNATURES:
            material = bool(
                abs(t.outcome.effect_raw) >= cfg.materiality.rate_threshold(t.signature)
            )
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
    # Families with a composition issue are skipped here too — diagnostics
    # computed on a known-broken design would only manufacture noise.
    composition_bad = {i.family for i in issues}
    if golden:
        golden_frame = frame[frame["cycle_id"].isin(golden)]
        for family in families:
            if family in composition_bad:
                continue
            g_fam = golden_frame[golden_frame["family"] == family]
            c_fam = cur_frame[cur_frame["family"] == family]
            if g_fam.empty or c_fam.empty:
                continue
            for sig in SCALAR_SIGNATURES:
                g_vals = g_fam[sig].to_numpy(dtype=float)
                c_vals = c_fam[sig].to_numpy(dtype=float)
                # Domain-of-validity guard: PSI's null expectation is
                # ~(B-1)*(1/n_ref + 1/n_cur); at canary scale this alone can
                # exceed the folk thresholds (measured: PSI flagged 100% of
                # stable runs before this guard). Emit PSI only where
                # sampling noise cannot produce a "moderate" label by
                # itself; it remains available for production-scale windows.
                if psi_null_expectation(len(g_vals), len(c_vals)) > PSI_MODERATE / 2:
                    continue
                res = psi(g_vals, c_vals)
                if res.label != "stable":
                    # PSI's 0.1/0.25 thresholds are folk conventions, not a
                    # materiality judgment; borrow the gated battery's
                    # materiality for this (family, signature) instead of
                    # hardcoding True.
                    psi_material = any(
                        bool(g.material) for g in gated if g.family == family and g.signature == sig
                    )
                    flags.append(
                        FlagRecord("psi", family, sig, res.value, res.label, None, psi_material)
                    )
    for family in families:
        if family in composition_bad:
            continue
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
        composition_issues=sorted(issues, key=lambda i: (i.baseline, i.family)),
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
                    "composition_issues": [
                        {"baseline": i.baseline, "family": i.family, "detail": i.detail}
                        for i in result.composition_issues
                    ],
                }
            ),
            f"sudden={result.verdict_sudden};cumulative={result.verdict_cumulative}"
            + (";DEGRADED" if result.degraded else ""),
        ),
    )
    check_id = cursor.lastrowid
    # Effect units match the scale each test is gated AND reported on.
    effect_units = {
        "ks": "ks_D",
        "levene": "mad_ratio",
        "p95_perm": "relative_p95_shift",
        "two_proportion_z": "rate_diff",
        "mmd": "mmd2",
    }
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
                effect_units.get(t.outcome.test, "cohen_d"),
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
