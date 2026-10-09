"""The drift check pipeline: dual baselines, test battery, FDR, materiality.

Gating order (SPEC.md §6 — order matters):

1. Run all tests, collect p-values (PSI and Page-Hinkley produce flags,
   never p-values).
2. Benjamini-Hochberg adjustment at ``q`` across the PRIMARY tests in the check
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
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import wasserstein_distance

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
from dedrift.detectors.cyclefx import (
    CycleEffectEstimate,
    cycle_level_pvalue,
    estimate_icc,
    per_cycle_statistic,
    rate_z_pvalue_clustered,
    standardize_within_cycle,
    welch_pvalue_clustered,
)
from dedrift.detectors.heuristic import PSI_MODERATE
from dedrift.detectors.mmd import MIN_FLOOR_CYCLES
from dedrift.embeddings import embed_records, get_pinned_embedder
from dedrift.schema import InteractionRecord
from dedrift.signatures import signatures_frame
from dedrift.signatures.structural import RATE_SIGNATURES, SCALAR_SIGNATURES
from dedrift.store import (
    FinalizedSnapshotChangedError,
    Store,
    _atomic_private_writer,
    _harden_permissions,
)

BASELINE_FILE = "baseline.json"

#: Cap on the per-channel cycle-effect estimate: beyond this the estimate is
#: treated as drift-contaminated and the correction is bounded (a larger rho
#: would inflate every corrected p enough to hide the drift that produced it).
RHO_CAP = 0.15

#: Fraction of errored records in the current cycle above which the check
#: refuses to interpret drift and reports DEGRADED DATA instead.
DEGRADED_ERROR_FRACTION = 0.2

#: Bump whenever the fixed-path candidate/persistence semantics change. Old
#: check rows then fail closed instead of completing a streak under new rules.
FIXED_PERSISTENCE_VERSION = "4"


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
        absolute_distance: Wasserstein-1 distance between the reference and
            current scalar samples in the signature's native units. ``None``
            for rate/semantic channels. A configured native-unit floor gates
            every primary scalar test on the channel.
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
    absolute_distance: float | None = None


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
        changed_canaries: Canary IDs whose input, expectation, predicate, or
            rubric identity differs between the reference and current window.
        detail: Human-readable explanation for the report.
    """

    baseline: str
    family: str
    missing_canaries: tuple[str, ...]
    extra_canaries: tuple[str, ...]
    unbalanced: bool
    detail: str
    changed_canaries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonAssessment:
    """Machine-readable evidence coverage for one baseline comparison.

    ``power_status`` is intentionally ``NOT ASSESSED``. A finite p-value
    establishes that a test was defined; it does not establish useful power,
    and the current configuration does not declare the alternative and target
    power needed to make that claim.
    """

    baseline: str
    coverage_status: str
    power_status: str
    n_families_total: int
    n_families_tested: int
    n_primary_tests: int
    n_valid_primary_tests: int
    n_undefined_primary_tests: int
    suppressed_families: tuple[str, ...] = ()
    n_channels_total: int = 0
    n_channels_tested: int = 0
    missing_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class MmdFloorAssessment:
    """Materiality-floor provenance for one semantic comparison.

    ``value`` is ``None`` when auto-calibration had too few known-good
    cycles.  In that state the semantic test remains visible as diagnostic
    evidence but cannot alert: silently substituting a zero floor would turn
    "uncalibrated" into "every significant effect is material".
    """

    baseline: str
    family: str
    value: float | None
    status: str
    n_reference_cycles: int


@dataclass(frozen=True)
class CheckResult:
    """Everything one check produced (input for attribution and the report).

    Attributes:
        ts: Check timestamp (UTC, ISO).
        current_cycle: Cycle under test.
        rolling_cycles: Cycle IDs in the rolling reference.
        golden_cycles: Cycle IDs in the golden baseline.
        declared_golden_cycles: Exact file-backed baseline declaration read
            before computation. Persistence verifies it is still current.
        fdr_q: FDR level used.
        seed: Seed used for all resampling.
        permutations_requested: Configured Monte Carlo permutation count.
        permutations_effective: Count actually used, automatically raised
            when needed so an isolated permutation test can reach the first
            BH threshold.
        primary_family_upper_bound: Conservative size used for that
            permutation-resolution calculation.
        verdict_sudden: ``OK`` / ``DRIFT DETECTED`` / ``NO REFERENCE`` /
            ``NO VALID COMPARISON`` / ``PARTIAL COVERAGE``.
        verdict_cumulative: Same, against the golden baseline.
        degraded: True if the current cycle's error fraction was too high
            for drift interpretation (overall verdict DEGRADED DATA).
        tests: All executed tests with gating outcomes.
        flags: PSI / Page-Hinkley indicators.
        composition_issues: Suppressed (baseline, family) comparisons whose
            windows were not composition-comparable.
        assessments: Machine-readable coverage and power status per baseline.
        mmd_floors: Semantic materiality-floor values and calibration status.
        cycle_effects: Per-channel cycle-effect engagements (baseline,
            family, signature, estimated ICC, reference-cycle count) whose
            p-values were replaced by the cluster-aware correction.
        persistence_demoted: Significant+material first-time alerts held by
            the alert_persistence gate at this check (0 when the gate is
            off).
        persistence_fingerprint: Identity of the detector configuration,
            golden baseline, suite, embedder, and active agent configuration.
            Candidate streaks never cross a fingerprint boundary.
        pinned_embedder: Exact embedder identity read for this computation.
            Persistence verifies the pin is still identical while holding the
            writer lock, so a concurrent first pin cannot silently produce a
            check with the semantic battery omitted.
        overall_verdict: Conservative aggregate status for CLI/report use.
        n_alerts: Convenience count of alerting tests.
        families_present: Families this project actually populates. Reported
            against the fixed vocabulary so a reader can tell a deliberately
            unused family from an accidentally empty one -- an agent that
            cannot refuse has no honest ``refusal_boundary`` canaries, and
            leaving it empty is correct rather than a gap in coverage.
    """

    ts: str
    current_cycle: str
    rolling_cycles: list[str]
    golden_cycles: list[str]
    declared_golden_cycles: list[str]
    fdr_q: float
    seed: int
    permutations_requested: int
    permutations_effective: int
    primary_family_upper_bound: int
    families_present: tuple[str, ...]
    verdict_sudden: str
    verdict_cumulative: str
    degraded: bool
    tests: list[TestRecord] = field(default_factory=list)
    flags: list[FlagRecord] = field(default_factory=list)
    composition_issues: list[CompositionIssue] = field(default_factory=list)
    assessments: list[ComparisonAssessment] = field(default_factory=list)
    mmd_floors: list[MmdFloorAssessment] = field(default_factory=list)
    cycle_effects: list[dict[str, Any]] = field(default_factory=list)
    persistence_demoted: int = 0
    persistence_fingerprint: str = ""
    pinned_embedder: str | None = None
    snapshot_log_offset: int = 0
    snapshot_finalized_revision: int = 0
    snapshot_record_ids: tuple[str, ...] = field(default_factory=tuple)

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

    @property
    def overall_verdict(self) -> str:
        """Conservative aggregate which never presents missing evidence as OK."""
        if self.degraded:
            return "DEGRADED DATA"
        if self.n_alerts:
            return "DRIFT DETECTED"
        verdicts = (self.verdict_sudden, self.verdict_cumulative)
        if "NO VALID COMPARISON" in verdicts:
            return "NO VALID COMPARISON"
        if "PARTIAL COVERAGE" in verdicts:
            return "PARTIAL COVERAGE"
        if "NO REFERENCE" in verdicts:
            return "PARTIAL COVERAGE" if "OK" in verdicts else "NO REFERENCE"
        return "OK"


# -- baseline management -------------------------------------------------------


def set_golden_baseline(store: Store, cycle_ids: list[str]) -> None:
    """Freeze the golden baseline to the given cycles (never auto-updated).

    Args:
        store: The project store.
        cycle_ids: Cycles the owner declares known-good.
    """
    conn = store.connect()
    if conn.in_transaction:
        raise RuntimeError("set_golden_baseline requires ownership of the SQLite transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        if not cycle_ids:
            raise ValueError("golden baseline requires at least one finalized cycle")
        if any(not isinstance(cycle_id, str) or not cycle_id for cycle_id in cycle_ids):
            raise ValueError("golden baseline cycle IDs must be non-empty strings")
        if len(set(cycle_ids)) != len(cycle_ids):
            raise ValueError("golden baseline cycle IDs must not contain duplicates")
        available = set(store.finalized_cycle_ids())
        missing = sorted(set(cycle_ids) - available)
        if missing:
            raise ValueError(
                "golden baseline contains unknown or open canary cycle(s): " + ", ".join(missing)
            )
        path = store.project_dir / BASELINE_FILE
        payload = json.dumps({"golden_cycles": sorted(cycle_ids)}, indent=2).encode("utf-8")
        with _atomic_private_writer(path) as stream:
            stream.write(payload)
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


def get_golden_baseline(store: Store) -> list[str]:
    """Return the frozen golden cycle IDs ([] if never set)."""
    path = store.project_dir / BASELINE_FILE
    if not path.exists():
        return []
    _harden_permissions(path, 0o600)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"golden_cycles"}:
        raise ValueError("baseline.json must contain only a golden_cycles array")
    cycles = data["golden_cycles"]
    if (
        not isinstance(cycles, list)
        or any(not isinstance(cycle, str) or not cycle for cycle in cycles)
        or len(set(cycles)) != len(cycles)
    ):
        raise ValueError("baseline.json golden_cycles must be unique non-empty strings")
    return cycles


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


def _material_scalar(
    test: TestOutcome,
    cfg: ProjectConfig,
    signature: str,
    absolute_distance: float | None,
) -> bool:
    m = cfg.materiality
    # Gates resolve per signature: the defaults are shaped for text channels,
    # and a project whose only live scalar is latency on shared hardware has
    # to raise them without desensitising every other channel at once.
    native_floor = m.scalar_threshold("absolute_distance", signature)
    if native_floor > 0 and (
        absolute_distance is None
        or not math.isfinite(absolute_distance)
        or absolute_distance < native_floor
    ):
        return False
    if test.test == "levene":
        ratio = m.scalar_threshold("dispersion_ratio", signature)
        r = test.effect_size
        return bool(r >= ratio or r <= 1 / ratio)
    if test.test == "p95_perm":
        return bool(abs(test.effect_size) >= m.scalar_threshold("p95_relative", signature))
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
        return bool(test.statistic >= m.scalar_threshold("ks_distance", signature))
    return bool(abs(test.effect_size) >= m.scalar_cohen_d)


def _persistence_fingerprint(
    cfg: ProjectConfig,
    declared_golden: list[str],
    current_records: list[InteractionRecord],
    pinned_embedder: str | None,
) -> str:
    """Hash every stable input that gives a candidate channel its meaning."""
    suite = sorted(
        {
            (
                record.canary_id,
                json.dumps(
                    record.input.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
            )
            for record in current_records
        }
    )
    payload = json.dumps(
        {
            "version": FIXED_PERSISTENCE_VERSION,
            "detector_config": asdict(cfg),
            "golden_cycles": sorted(declared_golden),
            "pinned_embedder": pinned_embedder or "",
            "agent_configs": sorted({record.config_fingerprint for record in current_records}),
            "suite": suite,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _add_semantic_displacement(
    frame: pd.DataFrame,
    records: list[InteractionRecord],
    embeddings: dict[str, npt.NDArray[np.float64]],
    reference_cycles: list[str],
) -> None:
    """Add a per-record semantic_displacement column to the signature frame.

    Displacement = cosine distance from each record's embedding to the
    centroid of the SAME canary's explicitly selected reference outputs. The
    caller selects the filtered golden window when one was declared, or a
    prior rolling/history window otherwise. The current cycle must never be
    part of this centroid: doing so lets the sample under test move its own
    null target and attenuates real semantic drift. An empty reference list
    deliberately produces only NaNs rather than silently falling back to all
    records. A scalar per record, so it flows through the ordinary scalar
    battery — KS/AD/Levene/P95 apply.

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
    ref_cycle_set = set(reference_cycles)
    by_canary: dict[str, list[npt.NDArray[np.float64]]] = {}
    for r in records:
        in_ref = r.cycle_id in ref_cycle_set
        if r.canary_id is not None and in_ref:
            by_canary.setdefault(r.canary_id, []).append(embeddings[r.id])
    centroids = {c: np.mean(np.stack(v), axis=0) for c, v in by_canary.items() if v}
    counts = {c: len(v) for c, v in by_canary.items() if v}
    reference_ids = {
        r.id for r in records if r.canary_id is not None and r.cycle_id in ref_cycle_set
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
    their input/evaluation identities changed, or repetition counts are not
    uniform across canaries within a window. In each case a two-sample test
    would mix a collection/suite change into the behavior comparison.
    """
    ref_counts = ref_fam["canary_id"].value_counts()
    cur_counts = cur_fam["canary_id"].value_counts()
    missing = tuple(sorted(set(ref_counts.index) - set(cur_counts.index)))
    extra = tuple(sorted(set(cur_counts.index) - set(ref_counts.index)))
    unbalanced = bool(
        (len(cur_counts) > 0 and cur_counts.min() != cur_counts.max())
        or (len(ref_counts) > 0 and ref_counts.min() != ref_counts.max())
    )

    identity_columns = (
        "suite_fingerprint",
        "canary_fingerprint",
        "correctness_predicate_id",
        "expectation_fingerprint",
        "rubric_id",
    )

    def identities(data: pd.DataFrame, canary_id: str) -> set[tuple[str | None, ...]]:
        rows = data[data["canary_id"] == canary_id]
        found: set[tuple[str | None, ...]] = set()
        for values in rows[list(identity_columns)].itertuples(index=False, name=None):
            found.add(
                tuple(None if value is None or pd.isna(value) else str(value) for value in values)
            )
        return found

    changed = tuple(
        sorted(
            canary_id
            for canary_id in set(ref_counts.index) & set(cur_counts.index)
            if identities(ref_fam, str(canary_id)) != identities(cur_fam, str(canary_id))
            or len(identities(ref_fam, str(canary_id))) != 1
            or len(identities(cur_fam, str(canary_id))) != 1
        )
    )
    if not missing and not extra and not unbalanced and not changed:
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
    if changed:
        parts.append(
            "canary input/evaluation identity changed or varied within a window: "
            + ", ".join(changed)
        )
    return CompositionIssue(
        baseline=baseline,
        family=family,
        missing_canaries=missing,
        extra_canaries=extra,
        unbalanced=unbalanced,
        detail="; ".join(parts),
        changed_canaries=changed,
    )


def _mmd_battery(
    baseline_name: str,
    ref_cycles: list[str],
    current: str,
    frame: pd.DataFrame,
    embeddings: dict[str, npt.NDArray[np.float64]],
    families: list[str],
    cfg: ProjectConfig,
    n_permutations: int,
    mmd_floors: dict[tuple[str, str], MmdFloorAssessment],
) -> list[TestRecord]:
    """Run MMD per family against one baseline; record calibrated floors.

    The materiality floor per (baseline, family) is the config override when
    non-negative, else the 95th percentile of MMD^2 between pairs of the
    baseline's own cycles (known-same distribution). With fewer than five
    reference cycles the floor is unavailable and MMD cannot alert unless an
    explicit non-negative floor was configured.
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
            n_permutations=n_permutations,
            seed=_test_seed(cfg.seed, baseline_name, family, "embedding", "mmd"),
            sigma=sigma,
        )
        usable_cycles = 0
        if cfg.materiality.embedding_mmd2_floor >= 0:
            floor: float | None = cfg.materiality.embedding_mmd2_floor
            status = "CONFIGURED"
        else:
            per_cycle = [
                np.stack([embeddings[i] for i in fam[fam["cycle_id"] == cycle]["record_id"]])
                for cycle in ref_cycles
            ]
            usable = [p for p in per_cycle if len(p) >= 2]
            usable_cycles = len(usable)
            if usable_cycles < MIN_FLOOR_CYCLES:
                floor = None
                status = "UNCALIBRATED"
            else:
                floor = calibrate_mmd_floor(usable, sigma=sigma)
                status = "AUTO-CALIBRATED"
        mmd_floors[(baseline_name, family)] = MmdFloorAssessment(
            baseline=baseline_name,
            family=family,
            value=floor,
            status=status,
            n_reference_cycles=usable_cycles if status != "CONFIGURED" else len(ref_cycles),
        )
        out.append(
            TestRecord(
                baseline_name,
                family,
                "embedding",
                outcome,
                primary=floor is not None,
            )
        )
    return out


def run_check(store: Store, config: ProjectConfig | None = None) -> CheckResult:
    """Run one fixed-horizon check against finalized, immutable cycles.

    The indexed record IDs and committed log offset are captured in a brief
    WAL read transaction. Expensive embeddings and resampling then run with
    no writer lock; finalization makes the selected cycle contents immutable.
    """
    cfg = config or ProjectConfig.load(store.project_dir)
    for attempt in range(3):
        records, snapshot_offset, snapshot_revision = (
            store.read_finalized_canary_snapshot_with_revision()
        )
        try:
            return _run_check_snapshot(
                store,
                cfg,
                records,
                snapshot_offset,
                snapshot_revision,
            )
        except FinalizedSnapshotChangedError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def _run_check_snapshot(
    store: Store,
    cfg: ProjectConfig,
    records: list[InteractionRecord],
    snapshot_offset: int,
    snapshot_revision: int,
) -> CheckResult:
    """Run the full gated drift check for the latest cycle.

    Args:
        store: The project store (must contain finalized canary cycles).
        cfg: Validated project configuration.
        records: Immutable finalized-record snapshot.
        snapshot_offset: Committed JSONL offset recorded with the check.
        snapshot_revision: Finalized-cycle watermark captured with the records.

    Returns:
        The complete check result. Also persisted to the ``checks`` and
        ``alerts`` tables.

    Raises:
        ValueError: If no canary cycles exist.
    """
    if not records:
        msg = "no finalized canary cycles; finish and finalize a canary cycle first"
        raise ValueError(msg)
    frame = signatures_frame(records, custom_scalars=tuple(cfg.custom_scalars))
    cycles = _ordered_cycles(frame)
    current = cycles[-1]
    history = cycles[:-1]
    declared_golden = get_golden_baseline(store)
    golden = [c for c in declared_golden if c in cycles and c != current]
    rolling = [c for c in history if c not in golden][-cfg.rolling_window_cycles :]
    pinned_embedder = get_pinned_embedder(store)
    if cfg.embedder and cfg.embedder != pinned_embedder:
        raise ValueError(
            f"embeddings.model expects {cfg.embedder!r}, but the authoritative pin is "
            f"{pinned_embedder!r}; use `dedrift embedder pin` or restore the project state"
        )
    persistence_fingerprint = _persistence_fingerprint(
        cfg,
        declared_golden,
        [record for record in records if record.cycle_id == current],
        pinned_embedder,
    )

    # Tier 2 (optional): embeddings via the pinned embedder. When pinned, a
    # per-record semantic_displacement scalar joins the ordinary scalar
    # battery, and MMD tests per family join the same BH pool. The MMD^2
    # materiality floor is auto-calibrated per (family, baseline) from
    # reference-cycle pairs (known-same distribution) unless overridden.
    embeddings: dict[str, Any] | None = None
    if pinned_embedder is not None:
        embeddings = embed_records(store, records)
        # A declared baseline is authoritative even when none of its cycle
        # IDs survive snapshot filtering (for example, it names only the
        # current cycle). Falling back in that case would silently change the
        # reference contract. With no declared golden baseline, use only
        # prior rolling/history cycles; current is never admitted.
        semantic_reference = golden if declared_golden else (rolling or history)
        _add_semantic_displacement(frame, records, embeddings, semantic_reference)

    built_in_scalars = tuple(
        signature
        for signature in SCALAR_SIGNATURES
        if cfg.monitor_latency or signature != "latency_ms"
    )
    scalar_signatures = (
        built_in_scalars
        + tuple(cfg.custom_scalars)
        + (("semantic_displacement",) if embeddings is not None else ())
    )

    cur_frame = frame[frame["cycle_id"] == current]
    degraded = bool(cur_frame["had_error"].mean() > DEGRADED_ERROR_FRACTION)

    tests: list[TestRecord] = []
    flags: list[FlagRecord] = []
    issues: list[CompositionIssue] = []
    cycle_notes: list[dict[str, Any]] = []
    mmd_floors: dict[tuple[str, str], MmdFloorAssessment] = {}
    families = sorted(frame["family"].unique())
    n_reference_baselines = int(bool(rolling)) + int(bool(golden))
    primary_family_upper_bound = (
        n_reference_baselines
        * len(families)
        * (3 * len(scalar_signatures) + len(RATE_SIGNATURES) + (1 if embeddings is not None else 0))
    )
    resolution_floor = (
        math.ceil(primary_family_upper_bound / cfg.fdr_q)
        if primary_family_upper_bound
        else cfg.permutations
    )
    if resolution_floor > 250_000:
        raise ValueError(
            "detection.fdr_q would require more than 250,000 permutations per test "
            "at this battery size; raise fdr_q or reduce the declared test family"
        )
    effective_permutations = max(cfg.permutations, resolution_floor)

    # Cycle-effect engagement: per-channel ICC from ALL history cycles
    # (never the one under test), log scale for positive channels. Two
    # measured failure modes shaped this:
    #  * golden-window-only estimation (K=3) cannot separate wobble from
    #    noise at canary within-cycle overdispersion (under-engagement left
    #    the sigma=0.15 audit null at 67%);
    #  * history-wide estimation is contaminated by a drift already in
    #    history, disarming the correction at post-onset checks.
    # The compromise: history-wide estimation with rho CAPPED at
    # RHO_CAP — beyond the cap the estimate is treated as drift-
    # contaminated and the correction is bounded rather than allowed to
    # disarm detection. semantic_displacement is excluded: its leave-one-out
    # reference construction already spans multiple cycles.
    hist_frame = frame[frame["cycle_id"] != current]
    golden_frame = frame[frame["cycle_id"].isin(golden)]
    icc_by_channel: dict[tuple[str, str], Any] = {}
    if cfg.cycle_effect == "auto":
        for family in families:
            fam_hist = hist_frame[hist_frame["family"] == family]
            for sig in tuple(scalar_signatures) + tuple(RATE_SIGNATURES):
                if sig == "semantic_displacement":
                    continue
                pair = fam_hist[[sig, "cycle_id"]].dropna()
                vals = pair[sig].to_numpy(dtype=float)
                if not len(vals):
                    continue
                if sig not in RATE_SIGNATURES and float(np.min(vals)) > 0.0:
                    vals = np.log(vals)
                ice = estimate_icc(
                    vals,
                    pair["cycle_id"].to_numpy(),
                    threshold=cfg.cycle_effect_icc,
                )
                if ice.rho > RHO_CAP:
                    ice = CycleEffectEstimate(RHO_CAP, ice.n_cycles, True)
                icc_by_channel[(family, sig)] = ice

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
                ref_pair = ref_fam[[sig, "cycle_id"]].dropna()
                ref = ref_pair[sig].to_numpy(dtype=float)
                cur = cur_fam[sig].dropna().to_numpy(dtype=float)
                ice = icc_by_channel.get((family, sig))
                # Primaries (enter BH, may alert): KS, Levene, P95 permutation.
                ks_out = ks_test(ref, cur)
                lev_out = levene_test(ref, cur)
                p95_out = p95_permutation_test(
                    ref,
                    cur,
                    n_permutations=effective_permutations,
                    seed=_test_seed(cfg.seed, baseline_name, family, sig, "p95_perm"),
                )
                native_distance = (
                    float(wasserstein_distance(ref, cur))
                    if len(ref) and len(cur)
                    else float("nan")
                )
                if ice is not None and ice.engaged:
                    # Cycle-level null stats come from THIS baseline's
                    # reference cycles when there are enough of them (they
                    # define the null level; all-history stats would absorb
                    # a persistent drift and dilute it). Early rolling
                    # windows (<3 cycles) fall back to all history.
                    base_cyc = ref_pair["cycle_id"].to_numpy()
                    hist_pair = hist_frame.loc[
                        hist_frame["family"] == family, [sig, "cycle_id"]
                    ].dropna()
                    hist_vals = hist_pair[sig].to_numpy(dtype=float)
                    hist_cyc = hist_pair["cycle_id"].to_numpy()
                    if len(set(base_cyc.tolist())) >= 3:
                        null_vals, null_cyc = ref, base_cyc
                    else:
                        null_vals, null_cyc = hist_vals, hist_cyc
                    # Work in log space for strictly positive channels:
                    # provider wobble is multiplicative (exp offsets), so
                    # logged offsets are additive and the Gaussian/t
                    # approximations behind the cycle-level summaries and
                    # Welch hold far better than on skewed raw statistics.
                    positive = (
                        len(ref) > 0
                        and len(cur) > 0
                        and float(np.min(ref)) > 0.0
                        and float(np.min(cur)) > 0.0
                    )
                    wref = np.log(ref) if positive else ref
                    wcur = np.log(cur) if positive else cur
                    wnull = np.log(null_vals) if positive else null_vals
                    if np.isfinite(ks_out.p_value):
                        # Engaged KS is a disjunctive composite: KS on
                        # within-cycle-standardized values (shape; cycle
                        # offsets cancel exactly, so record-level power
                        # survives) OR design-effect Welch (location; the
                        # Kish correction IS the right one for means under
                        # clustering). Bonferroni-min is valid under any
                        # dependence of the two members.
                        #
                        # The shape member requires quasi-continuous data:
                        # on few-support channels (retries, tool counts) the
                        # per-cycle z-grid itself shifts with the cycle's
                        # rate estimate and KS fires on the grid mismatch —
                        # measured 90% null alerts before this guard.
                        continuous = len(np.unique(ref)) >= 10 and len(np.unique(cur)) >= 10
                        ks_std = None
                        if continuous:
                            ref_std = standardize_within_cycle(wref, base_cyc)
                            cur_std = standardize_within_cycle(wcur, np.full(len(wcur), "__cur__"))
                            ref_std = ref_std[np.isfinite(ref_std)]
                            cur_std = cur_std[np.isfinite(cur_std)]
                            if len(ref_std) >= 2 and len(cur_std) >= 2:
                                ks_std = ks_test(ref_std, cur_std)
                        p_mean = welch_pvalue_clustered(
                            wref,
                            wcur,
                            ice.rho,
                            len(wref) / max(1, len(set(base_cyc.tolist()))),
                            float(len(wcur)),
                        )
                        parts = [p_mean] if np.isfinite(p_mean) else []
                        if ks_std is not None and np.isfinite(ks_std.p_value):
                            parts.append(ks_std.p_value)
                        p_comp = (
                            min(1.0, 2.0 * min(parts))
                            if len(parts) == 2
                            else (parts[0] if parts else ks_out.p_value)
                        )
                        # ``auto`` is a robustness correction, never a way
                        # to manufacture stronger evidence than the exact
                        # record-level path. The approximate cluster-aware
                        # member can occasionally be numerically smaller;
                        # retain the larger (more conservative) p-value.
                        p_comp = max(ks_out.p_value, p_comp)
                        # Gating needs both members' effects in D units:
                        # D_shape misses pure location shifts, so add the
                        # KS distance that a pure location shift of the
                        # observed size would produce on the reference law.
                        delta = float(np.mean(wcur) - np.mean(wnull)) if len(wcur) else 0.0
                        d_loc = (
                            ks_test(wref, wref + delta).statistic
                            if len(wref) and np.isfinite(delta)
                            else 0.0
                        )
                        d_shape = ks_std.statistic if ks_std is not None else 0.0
                        d_corrected = max(d_shape, d_loc)
                        ks_out = replace(
                            ks_out,
                            statistic=d_corrected,
                            p_value=p_comp,
                            effect_size=d_corrected,
                            effect_raw=math.exp(delta) - 1.0 if positive else delta,
                        )
                    ref_mads = per_cycle_statistic(wnull, null_cyc, "mad")
                    cur_mad = (
                        float(np.mean(np.abs(wcur - np.median(wcur))))
                        if len(wcur)
                        else float("nan")
                    )
                    # Near-discrete cycle statistics (a 0/1 P95 flipping by
                    # sampling noise) make the t summary meaningless: too few
                    # distinct reference values => keep the record-level p.
                    p_lev = (
                        cycle_level_pvalue(ref_mads, cur_mad)
                        if len(np.unique(ref_mads)) >= 4
                        else float("nan")
                    )
                    if np.isfinite(p_lev):
                        mean_ref_mad = float(np.mean(ref_mads))
                        if positive:
                            # ratio of log-scale dispersions, on the gated scale
                            ratio = (
                                math.exp(cur_mad - mean_ref_mad)
                                if np.isfinite(cur_mad)
                                else float("nan")
                            )
                        else:
                            ratio = cur_mad / mean_ref_mad if mean_ref_mad > 0 else float("inf")
                        lev_out = replace(
                            lev_out,
                            p_value=(
                                max(lev_out.p_value, p_lev)
                                if np.isfinite(lev_out.p_value)
                                else p_lev
                            ),
                            effect_size=ratio,
                        )
                    ref_p95s = per_cycle_statistic(wnull, null_cyc, "p95")
                    cur_p95 = float(np.percentile(wcur, 95)) if len(wcur) else float("nan")
                    p_p95 = (
                        cycle_level_pvalue(ref_p95s, cur_p95)
                        if len(np.unique(ref_p95s)) >= 4
                        else float("nan")
                    )
                    if np.isfinite(p_p95):
                        mean_ref_p95 = float(np.mean(ref_p95s))
                        if positive:
                            rel = (
                                math.exp(cur_p95 - mean_ref_p95) - 1.0
                                if np.isfinite(cur_p95)
                                else float("nan")
                            )
                        else:
                            rel = (
                                (cur_p95 - mean_ref_p95) / abs(mean_ref_p95)
                                if mean_ref_p95 != 0
                                else (0.0 if cur_p95 == 0 else float("inf"))
                            )
                        p95_out = replace(
                            p95_out,
                            p_value=(
                                max(p95_out.p_value, p_p95)
                                if np.isfinite(p95_out.p_value)
                                else p_p95
                            ),
                            effect_size=rel,
                        )
                    cycle_notes.append(
                        {
                            "baseline": baseline_name,
                            "family": family,
                            "signature": sig,
                            "rho": round(ice.rho, 4),
                            "n_reference_cycles": ice.n_cycles,
                        }
                    )
                out.append(
                    TestRecord(
                        baseline_name,
                        family,
                        sig,
                        ks_out,
                        absolute_distance=native_distance,
                    )
                )
                out.append(
                    TestRecord(
                        baseline_name,
                        family,
                        sig,
                        lev_out,
                        absolute_distance=native_distance,
                    )
                )
                out.append(
                    TestRecord(
                        baseline_name,
                        family,
                        sig,
                        p95_out,
                        absolute_distance=native_distance,
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
                z_out = two_proportion_z_test(
                    int(ref_series.sum()),
                    len(ref_series),
                    int(cur_series.sum()),
                    len(cur_series),
                )
                ice = icc_by_channel.get((family, sig))
                if ice is not None and ice.engaged and np.isfinite(z_out.p_value):
                    m_ref = len(ref_series) / max(
                        1, ref_fam.loc[ref_series.index, "cycle_id"].nunique()
                    )
                    z_out = replace(
                        z_out,
                        p_value=rate_z_pvalue_clustered(
                            int(ref_series.sum()),
                            len(ref_series),
                            int(cur_series.sum()),
                            len(cur_series),
                            ice.rho,
                            m_ref,
                            float(len(cur_series)),
                        ),
                    )
                    cycle_notes.append(
                        {
                            "baseline": baseline_name,
                            "family": family,
                            "signature": sig,
                            "rho": round(ice.rho, 4),
                            "n_reference_cycles": ice.n_cycles,
                        }
                    )
                out.append(TestRecord(baseline_name, family, sig, z_out))
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
            if ref_fam.empty and not cur_fam.empty:
                issues.append(
                    CompositionIssue(
                        baseline=name,
                        family=family,
                        missing_canaries=(),
                        extra_canaries=tuple(sorted(cur_fam["canary_id"].dropna().unique())),
                        unbalanced=False,
                        detail="family absent from the reference window",
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
                    name,
                    ref_cycles,
                    current,
                    frame,
                    embeddings,
                    ok_families,
                    cfg,
                    effective_permutations,
                    mmd_floors,
                )
            )
        verdicts[name] = "pending"

    # Gate 2: one BH family across the PRIMARY tests in this check.
    primary_idx = [i for i, t in enumerate(tests) if t.primary]
    rejected_p, adjusted_p = benjamini_hochberg(
        [tests[i].outcome.p_value for i in primary_idx],
        q=cfg.fdr_q,
        m_total=primary_family_upper_bound,
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
    #
    # Degradation suppresses alerts EXCEPT had_error: a check that refuses
    # to interpret drift while the error rate surges, and stays silent about
    # the surge itself, makes an error storm a perfect evasion. The error
    # rate remains interpretable under degradation (it is the degradation
    # criterion), so it alerts; everything else stays suppressed and the
    # overall verdict remains DEGRADED DATA.
    gated: list[TestRecord] = []
    for t, rej, adj in zip(tests, rejected, adjusted, strict=True):
        material: bool | None
        if not t.primary:
            material = None
        elif t.outcome.test == "mmd":
            floor = mmd_floors.get((t.baseline, t.family))
            material = bool(
                floor is not None
                and floor.value is not None
                and t.outcome.effect_size >= floor.value
            )
        elif t.signature in RATE_SIGNATURES:
            material = bool(
                abs(t.outcome.effect_raw) >= cfg.materiality.rate_threshold(t.signature)
            )
        else:
            material = _material_scalar(
                t.outcome,
                cfg,
                t.signature,
                t.absolute_distance,
            )
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
                alert=bool(
                    t.primary and rej and material and (not degraded or t.signature == "had_error")
                ),
                absolute_distance=t.absolute_distance,
            )
        )

    # Alert persistence gate: with alert_persistence=N, a channel alerts
    # only after it is a candidate on N distinct consecutive checked cycles
    # (current plus N-1 prior cycles). Re-running a check on the same current
    # cycle cannot advance the streak. Wobble-induced alerts are transient;
    # drift persists. Candidates held by the gate remain visible in the
    # appendix. Measured numbers and the trade-off are in
    # docs/statistics.md#cycle-effects.
    demoted = 0
    if cfg.alert_persistence > 1:
        required_prior = cfg.alert_persistence - 1
        # These are the actual cycles immediately before current, not merely
        # the latest cycles on which somebody happened to run a check. A
        # skipped cycle breaks the streak because its candidacy is unknown.
        expected_prior_cycles = list(reversed(history[-required_prior:]))
        expected_prior_set = set(expected_prior_cycles)
        rows = store.connect().execute(
            "SELECT params_json FROM checks WHERE baseline_kind = 'dual' ORDER BY id DESC"
        )
        latest_candidates: dict[str, set[tuple[str, str, str, str]]] = {}
        for row in rows:
            params = json.loads(row[0])
            if params.get("persistence_fingerprint") != persistence_fingerprint:
                continue
            prior_cycle = params.get("current_cycle")
            if (
                not isinstance(prior_cycle, str)
                or prior_cycle not in expected_prior_set
                or prior_cycle in latest_candidates
            ):
                continue
            parsed_candidates: set[tuple[str, str, str, str]] = set()
            for candidate in params.get("candidate_channels", []):
                if isinstance(candidate, list) and len(candidate) == 4:
                    parsed_candidates.add(
                        (
                            str(candidate[0]),
                            str(candidate[1]),
                            str(candidate[2]),
                            str(candidate[3]),
                        )
                    )
            latest_candidates[prior_cycle] = parsed_candidates
            if len(latest_candidates) == required_prior:
                break

        # The gate compares against CANDIDATE channels (primary &
        # significant & material), persisted in params_json. Referencing
        # only fired alerts would self-extinguish a streak before it reached
        # N. Missing prior distinct cycles make the eligible set empty.
        persistent_channels: set[tuple[str, str, str, str]] = set()
        if len(expected_prior_cycles) == required_prior and all(
            cycle in latest_candidates for cycle in expected_prior_cycles
        ):
            prior_candidate_sets = [latest_candidates[cycle] for cycle in expected_prior_cycles]
            persistent_channels = set.intersection(*prior_candidate_sets)
        new_gated: list[TestRecord] = []
        for t in gated:
            identity = (t.baseline, t.family, t.signature, t.outcome.test)
            if t.alert and identity not in persistent_channels:
                demoted += 1
                new_gated.append(replace(t, alert=False))
            else:
                new_gated.append(t)
        gated = new_gated

    # Flags: PSI (golden bins) and Page-Hinkley (per-cycle means over history).
    # Families with a composition issue are skipped here too — diagnostics
    # computed on a known-broken design would only manufacture noise.
    composition_bad = {i.family for i in issues}
    # Declared numeric channels are scalars like any other and get the same
    # diagnostics. The semantic channel stays out, as it always has: its
    # per-record displacement is defined against a reference centroid, so a
    # PSI bin table or a Page-Hinkley running mean over it would be tracking
    # the reference as much as the agent.
    flag_signatures = tuple(SCALAR_SIGNATURES) + tuple(cfg.custom_scalars)
    if golden:
        golden_frame = frame[frame["cycle_id"].isin(golden)]
        for family in families:
            if family in composition_bad:
                continue
            g_fam = golden_frame[golden_frame["family"] == family]
            c_fam = cur_frame[cur_frame["family"] == family]
            if g_fam.empty or c_fam.empty:
                continue
            for sig in flag_signatures:
                g_vals = g_fam[sig].to_numpy(dtype=float)
                c_vals = c_fam[sig].to_numpy(dtype=float)
                # A declared channel the agent did not emit is NaN, not zero.
                if not (np.isfinite(g_vals).all() and np.isfinite(c_vals).all()):
                    continue
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
        # Per-cycle means over the cycles where THIS family has data, in
        # global cycle order. Reindexing onto all cycles instead inserts NaN
        # for mid-history gaps, and NaN propagates through the running mean
        # and both accumulators: the stream could never alarm again after a
        # single missed cycle, silently. Masking to observed cycles shortens
        # the stream (onset precision degrades honestly) instead of
        # poisoning it.
        present = set(fam_frame["cycle_id"].unique())
        fam_cycles = [c for c in cycles if c in present]
        fam_frame_by_cycle = fam_frame.groupby("cycle_id")
        for sig in flag_signatures:
            means = fam_frame_by_cycle[sig].mean().reindex(fam_cycles).to_numpy()
            if not np.isfinite(means).all():
                continue
            ph: PageHinkleyResult = page_hinkley(means, lambda_=cfg.ph_lambda, delta=cfg.ph_delta)
            if ph.alarm:
                onset = fam_cycles[ph.change_index] if ph.change_index is not None else None
                material = any(
                    g.material for g in gated if g.family == family and g.signature == sig
                )
                flags.append(
                    FlagRecord(
                        "page_hinkley", family, sig, ph.statistic, ph.direction, onset, material
                    )
                )

    assessments: list[ComparisonAssessment] = []
    for name, ref_cycles in (("rolling", rolling), ("golden", golden)):
        primary = [t for t in gated if t.baseline == name and t.primary]
        valid = [t for t in primary if np.isfinite(t.outcome.p_value)]
        tested_families = {t.family for t in valid}
        observed_channels = {(t.family, t.signature) for t in valid}
        expected_channels: set[tuple[str, str]] = set()
        ref_frame = frame[frame["cycle_id"].isin(ref_cycles)]
        for family in families:
            expected_channels.update((family, signature) for signature in scalar_signatures)
            ref_fam = ref_frame[ref_frame["family"] == family]
            cur_fam = cur_frame[cur_frame["family"] == family]
            for signature in RATE_SIGNATURES:
                # Correctness is opt-in: when no record in either window has
                # an executable expectation contract, an absent exact-match
                # channel is honest rather than missing evidence. Once used
                # on either side, however, it is part of the declared health
                # manifest and cannot silently disappear.
                if signature == "exact_match" and not (
                    ref_fam[signature].notna().any() or cur_fam[signature].notna().any()
                ):
                    continue
                expected_channels.add((family, signature))
            if embeddings is not None:
                expected_channels.add((family, "embedding"))
        missing_channel_pairs = sorted(expected_channels - observed_channels)
        missing_channels = tuple(
            f"{family}/{signature}" for family, signature in missing_channel_pairs
        )
        suppressed = tuple(sorted({i.family for i in issues if i.baseline == name}))
        if not ref_cycles:
            coverage = "NO REFERENCE"
        elif not valid:
            coverage = "NONE"
        elif suppressed or missing_channels or len(tested_families) < len(families):
            coverage = "PARTIAL"
        else:
            coverage = "FULL"
        assessments.append(
            ComparisonAssessment(
                baseline=name,
                coverage_status=coverage,
                power_status="NOT ASSESSED",
                n_families_total=len(families),
                n_families_tested=len(tested_families),
                n_primary_tests=len(primary),
                n_valid_primary_tests=len(valid),
                n_undefined_primary_tests=len(primary) - len(valid),
                suppressed_families=suppressed,
                n_channels_total=len(expected_channels),
                n_channels_tested=len(expected_channels & observed_channels),
                missing_channels=missing_channels,
            )
        )
        if verdicts.get(name) != "pending":
            continue
        if not valid:
            verdicts[name] = "NO VALID COMPARISON"
        elif any(t.alert for t in gated if t.baseline == name):
            verdicts[name] = "DRIFT DETECTED"
        elif coverage == "PARTIAL":
            verdicts[name] = "PARTIAL COVERAGE"
        else:
            verdicts[name] = "OK"

    result = CheckResult(
        ts=datetime.now(timezone.utc).isoformat(),
        current_cycle=current,
        rolling_cycles=rolling,
        golden_cycles=golden,
        declared_golden_cycles=declared_golden,
        fdr_q=cfg.fdr_q,
        seed=cfg.seed,
        permutations_requested=cfg.permutations,
        permutations_effective=effective_permutations,
        primary_family_upper_bound=primary_family_upper_bound,
        families_present=tuple(families),
        verdict_sudden=verdicts["rolling"],
        verdict_cumulative=verdicts["golden"],
        degraded=degraded,
        tests=gated,
        flags=sorted(flags, key=lambda f: (f.kind, f.family, f.signature)),
        composition_issues=sorted(issues, key=lambda i: (i.baseline, i.family)),
        assessments=assessments,
        mmd_floors=sorted(mmd_floors.values(), key=lambda item: (item.baseline, item.family)),
        cycle_effects=cycle_notes,
        persistence_demoted=demoted,
        persistence_fingerprint=persistence_fingerprint,
        pinned_embedder=pinned_embedder,
        snapshot_log_offset=snapshot_offset,
        snapshot_finalized_revision=snapshot_revision,
        snapshot_record_ids=tuple(record.id for record in records),
    )
    _persist(store, result)
    return result


def _persist(store: Store, result: CheckResult) -> None:
    conn = store.connect()
    if conn.in_transaction:
        raise RuntimeError("check persistence requires ownership of the SQLite transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        store.assert_finalized_revision(result.snapshot_finalized_revision)
        if get_golden_baseline(store) != result.declared_golden_cycles:
            raise FinalizedSnapshotChangedError(
                "golden baseline changed during the check snapshot; retry"
            )
        if get_pinned_embedder(store) != result.pinned_embedder:
            raise FinalizedSnapshotChangedError(
                "pinned embedder changed during the check snapshot; retry"
            )
        cursor = conn.execute(
            "INSERT INTO checks "
            "(ts, baseline_kind, params_json, verdict, snapshot_offset, "
            "snapshot_record_ids_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.ts,
                "dual",
                json.dumps(
                    {
                        "current_cycle": result.current_cycle,
                        "rolling": result.rolling_cycles,
                        "golden": result.golden_cycles,
                        "declared_golden": result.declared_golden_cycles,
                        "fdr_q": result.fdr_q,
                        "seed": result.seed,
                        "permutations_requested": result.permutations_requested,
                        "permutations_effective": result.permutations_effective,
                        "primary_family_upper_bound": result.primary_family_upper_bound,
                        "overall_verdict": result.overall_verdict,
                        "assessments": [asdict(a) for a in result.assessments],
                        "mmd_floors": [asdict(item) for item in result.mmd_floors],
                        "cycle_effects": result.cycle_effects,
                        "persistence_demoted": result.persistence_demoted,
                        "persistence_fingerprint": result.persistence_fingerprint,
                        "pinned_embedder": result.pinned_embedder,
                        # Channels that earned an alert this check, whether
                        # or not the persistence gate fired them; the gate
                        # reads this at the next check.
                        "candidate_channels": [
                            [t.baseline, t.family, t.signature, t.outcome.test]
                            for t in result.tests
                            if t.primary
                            and t.significant
                            and t.material
                            and (not result.degraded or t.signature == "had_error")
                        ],
                        "composition_issues": [
                            {
                                "baseline": i.baseline,
                                "family": i.family,
                                "detail": i.detail,
                                "missing_canaries": i.missing_canaries,
                                "extra_canaries": i.extra_canaries,
                                "changed_canaries": i.changed_canaries,
                                "unbalanced": i.unbalanced,
                            }
                            for i in result.composition_issues
                        ],
                        "snapshot_log_offset": result.snapshot_log_offset,
                        "snapshot_finalized_revision": result.snapshot_finalized_revision,
                        "snapshot_record_count": len(result.snapshot_record_ids),
                    }
                ),
                f"sudden={result.verdict_sudden};cumulative={result.verdict_cumulative}"
                + f";overall={result.overall_verdict}"
                + (";DEGRADED" if result.degraded else ""),
                result.snapshot_log_offset,
                None,
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
                " effect_size, effect_units, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
                            "absolute_distance": t.absolute_distance,
                            "n_ref": t.outcome.n_ref,
                            "n_cur": t.outcome.n_cur,
                        }
                    ),
                ),
            )
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


def export_baseline_path(store: Store) -> Path:
    """Path of the golden-baseline file (for docs/tests)."""
    return store.project_dir / BASELINE_FILE
