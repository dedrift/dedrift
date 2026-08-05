"""Batch two-sample tests for scalar and rate signatures (SPEC.md §6).

Each test returns a :class:`TestOutcome` carrying the statistic, the p-value,
and effect sizes in both standardized and original units.

Validity note, and it is a caveat rather than a reassurance. The canary
design is balanced by construction (both windows contain the same canaries
at the same repetition count), which buys equal *composition*. It does NOT
buy exchangeability: that additionally requires the per-record law to be
constant across cycles, which is strictly stronger than "no change anywhere
in the configured stack" and is false whenever a hosted model varies within
a version. The hypothesis these tests actually address is the distributional
one; what a violation costs is measured, not assumed -- see
``TestCycleEffectRobustness`` in tests/test_calibration.py and the
statistics page in the docs. Detection power for shifts confined to a few
canaries is also lower than for family-wide shifts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats


@dataclass(frozen=True)
class TestOutcome:
    """Outcome of one hypothesis test on one signature.

    Attributes:
        test: Test identifier (e.g. ``"ks"``, ``"two_proportion_z"``).
        statistic: The test statistic.
        p_value: The (raw, pre-FDR) p-value; NaN if the test was undefined
            (e.g. degenerate samples).
        effect_size: Standardized effect on the scale the test is gated on:
            the KS statistic D for KS (a sup-norm CDF distance — the honest
            effect for a test that detects any distributional change),
            Cohen's d for the location corroboration tests (AD, Welch),
            percentage-point shift/100 for rates, variance ratio for Levene,
            relative P95 shift for the permutation test.
        effect_raw: Effect in original units (mean/rate/P95 difference,
            current minus reference).
        n_ref: Reference-window sample size.
        n_cur: Current-window sample size.
    """

    test: str
    statistic: float
    p_value: float
    effect_size: float
    effect_raw: float
    n_ref: int
    n_cur: int


def _cohens_d(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> float:
    n1, n2 = len(ref), len(cur)
    v1 = float(np.var(ref, ddof=1)) if n1 > 1 else 0.0
    v2 = float(np.var(cur, ddof=1)) if n2 > 1 else 0.0
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((np.mean(cur) - np.mean(ref)) / np.sqrt(pooled))


def _too_small(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> bool:
    return len(ref) < 2 or len(cur) < 2


def _same_constant(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> bool:
    """Whether both windows are constant at the same value."""
    return bool(np.ptp(ref) == 0 and np.ptp(cur) == 0 and ref[0] == cur[0])


def _different_constants(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> bool:
    """Whether both windows are constant but at different values."""
    return bool(np.ptp(ref) == 0 and np.ptp(cur) == 0 and ref[0] != cur[0])


def ks_test(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> TestOutcome:
    """Two-sample Kolmogorov-Smirnov test.

    The reported effect is the KS statistic D itself, matching the scale the
    materiality gate uses. KS detects ANY distributional change, so reporting
    Cohen's d as its effect would make a genuine shape-change alert (equal
    means, d ~ 0) look immaterial — the exact case the D gate exists to
    catch. Cohen's d for the same comparison appears on the Welch
    corroboration row.

    Args:
        ref: Reference-window values.
        cur: Current-window values.

    Returns:
        Outcome with D as the effect and the raw mean shift in original
        units.
    """
    if _too_small(ref, cur):
        return TestOutcome("ks", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
    if _same_constant(ref, cur):
        return TestOutcome("ks", 0.0, 1.0, 0.0, 0.0, len(ref), len(cur))
    res = stats.ks_2samp(ref, cur, method="auto")
    return TestOutcome(
        test="ks",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=float(res.statistic),
        effect_raw=float(np.mean(cur) - np.mean(ref)),
        n_ref=len(ref),
        n_cur=len(cur),
    )


def ad_test(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> TestOutcome:
    """Two-sample Anderson-Darling test (k-sample form, k=2).

    Uses SciPy's deterministic asymptotic approximation, which caps the
    returned p-value to [0.001, 0.25]. The cap is stated in the report; a
    consequence (documented) is that AD p-values can never be the sole
    survivor of FDR in large test batteries — AD serves as corroboration
    for KS, which is uncapped.

    Args:
        ref: Reference-window values.
        cur: Current-window values.

    Returns:
        Outcome with Cohen's d and raw mean shift as effects.
    """
    if _too_small(ref, cur):
        return TestOutcome("ad", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
    if _same_constant(ref, cur):
        return TestOutcome("ad", 0.0, 1.0, 0.0, 0.0, len(ref), len(cur))
    if _different_constants(ref, cur):
        delta = float(np.mean(cur) - np.mean(ref))
        return TestOutcome("ad", float("inf"), 0.0, float("inf"), delta, len(ref), len(cur))
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*p-value capped.*")
        warnings.filterwarnings("ignore", message=".*p-value floored.*")
        warnings.filterwarnings("ignore", message=".*midrank.*")  # SciPy 1.19 rename
        res = stats.anderson_ksamp([ref, cur])
    # Effect on the scale of the test, as everywhere else: AD is a
    # distributional statistic, so a standardized MEAN difference beside it
    # invites exactly the misreading the KS gate was fixed to prevent (a
    # pure shape change has d ~ 0). Cohen's d lives on the Welch row only.
    return TestOutcome(
        test="ad",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=float(res.statistic),
        effect_raw=float(np.mean(cur) - np.mean(ref)),
        n_ref=len(ref),
        n_cur=len(cur),
    )


def welch_t_test(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> TestOutcome:
    """Welch's t-test on means — secondary, only ever run alongside KS.

    Args:
        ref: Reference-window values.
        cur: Current-window values.

    Returns:
        Outcome with Cohen's d and raw mean shift as effects.
    """
    if _too_small(ref, cur):
        return TestOutcome("welch_t", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
    if _same_constant(ref, cur):
        return TestOutcome("welch_t", 0.0, 1.0, 0.0, 0.0, len(ref), len(cur))
    if _different_constants(ref, cur):
        delta = float(np.mean(cur) - np.mean(ref))
        direction = float(np.sign(delta))
        return TestOutcome(
            "welch_t",
            direction * float("inf"),
            0.0,
            direction * float("inf"),
            delta,
            len(ref),
            len(cur),
        )
    res = stats.ttest_ind(cur, ref, equal_var=False)
    return TestOutcome(
        test="welch_t",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=_cohens_d(ref, cur),
        effect_raw=float(np.mean(cur) - np.mean(ref)),
        n_ref=len(ref),
        n_cur=len(cur),
    )


def levene_test(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> TestOutcome:
    """Levene test for variance shift (median-centered / Brown-Forsythe).

    Effect is reported and gated on the scale the test is computed on.
    Brown--Forsythe forms an F statistic on absolute deviations from the
    median precisely to avoid the sample variance, which is unstable under
    the heavy tails this test exists to survive. Reporting (and gating on)
    ``var(cur)/var(ref)`` would therefore choose a robust test and then
    hand the decision back to the non-robust quantity it was chosen to
    avoid -- the same mistake as gating KS on Cohen's d, one channel over.

    ``effect_size`` is the **robust dispersion ratio**: the ratio of mean
    absolute deviations from each window's median, which is the quantity
    the F statistic is built from. ``effect_raw`` is the difference of
    those mean absolute deviations, in the signature's own units. The raw
    The sample-variance ratio is deliberately not reported here: two
    dispersion numbers on one row is how the wrong one gets quoted.

    Args:
        ref: Reference-window values.
        cur: Current-window values.

    Returns:
        Outcome for the dispersion channel.
    """
    if _too_small(ref, cur):
        return TestOutcome("levene", float("nan"), float("nan"), 1.0, 0.0, len(ref), len(cur))
    if _same_constant(ref, cur) or _different_constants(ref, cur):
        return TestOutcome("levene", 0.0, 1.0, 1.0, 0.0, len(ref), len(cur))
    res = stats.levene(ref, cur, center="median")
    z_ref = float(np.mean(np.abs(ref - np.median(ref))))
    z_cur = float(np.mean(np.abs(cur - np.median(cur))))
    ratio = z_cur / z_ref if z_ref > 0 else float("inf")
    return TestOutcome(
        test="levene",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=ratio,
        effect_raw=z_cur - z_ref,
        n_ref=len(ref),
        n_cur=len(cur),
    )


def p95_permutation_test(
    ref: npt.NDArray[np.float64],
    cur: npt.NDArray[np.float64],
    n_permutations: int = 500,
    seed: int = 0,
) -> TestOutcome:
    """Permutation test for a shift in the 95th percentile.

    Pools both windows, permutes the window labels, and recomputes the P95
    difference; the two-sided p-value uses the add-one convention
    ``(b+1)/(B+1)``, which is valid at every level under exchangeability of
    the pooled sample -- and conservative rather than exact, for three
    reasons worth naming: Monte Carlo sampling of the permutation set, the
    add-one convention itself, and ties counted as exceedances below. A
    further conservatism applies when canaries within a family have
    different laws: the pooled vector is then exchangeable only under
    canary-preserving permutations, and we permute unrestrictedly, which
    widens the permutation null relative to the truth. Stratifying by canary
    would remove that and is not yet done.

    This replaces an earlier centered percentile bootstrap: bootstrap
    approximations of extreme-quantile nulls are unreliable at these sample
    sizes (the P95 rests on a handful of order statistics). Effect size is
    the relative P95 shift.

    Args:
        ref: Reference-window values.
        cur: Current-window values.
        n_permutations: Number of label permutations (seeded).
        seed: RNG seed (recorded in the report).

    Returns:
        Outcome for the tail-behavior channel.
    """
    if _too_small(ref, cur):
        return TestOutcome("p95_perm", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
    if _same_constant(ref, cur):
        return TestOutcome("p95_perm", 0.0, 1.0, 0.0, 0.0, len(ref), len(cur))
    rng = np.random.default_rng(seed)
    n, m = len(ref), len(cur)
    p95_ref = float(np.percentile(ref, 95))
    p95_cur = float(np.percentile(cur, 95))
    observed = p95_cur - p95_ref
    pooled = np.concatenate([ref, cur])
    # Stream vectorized batches. The pipeline raises B to the resolution
    # required by its full BH family (often 6k+); allocating one B x N random,
    # index, and shuffled matrix can otherwise exhaust memory on production
    # windows. Chunking preserves the exact seeded permutation sequence.
    max_batch_elements = 1_000_000
    batch_size = max(1, min(n_permutations, max_batch_elements // (n + m)))
    exceedances = 0
    remaining = n_permutations
    while remaining:
        batch = min(batch_size, remaining)
        order = np.argsort(rng.random((batch, n + m)), axis=1)
        shuffled = pooled[order]
        diffs = np.percentile(shuffled[:, n:], 95, axis=1) - np.percentile(
            shuffled[:, :n], 95, axis=1
        )
        exceedances += int(np.sum(np.abs(diffs) >= abs(observed)))
        remaining -= batch
    p = float((exceedances + 1) / (n_permutations + 1))
    rel = observed / p95_ref if p95_ref != 0 else float("inf")
    return TestOutcome(
        test="p95_perm",
        statistic=observed,
        p_value=p,
        effect_size=float(rel),
        effect_raw=observed,
        n_ref=n,
        n_cur=m,
    )


def two_proportion_z_test(
    successes_ref: int,
    n_ref: int,
    successes_cur: int,
    n_cur: int,
) -> TestOutcome:
    """Two-proportion z-test with Yates continuity correction.

    Effect is the percentage-point shift: ``effect_raw`` is the rate
    difference (current minus reference, in [0, 1] units); ``effect_size``
    equals ``effect_raw`` (rates are already on a standardized scale).

    Args:
        successes_ref: Successes in the reference window.
        n_ref: Reference-window trials.
        successes_cur: Successes in the current window.
        n_cur: Current-window trials.

    Returns:
        Outcome for a rate signature.
    """
    if n_ref == 0 or n_cur == 0:
        return TestOutcome("two_proportion_z", float("nan"), float("nan"), 0.0, 0.0, n_ref, n_cur)
    p1 = successes_ref / n_ref
    p2 = successes_cur / n_cur
    diff = p2 - p1
    pooled = (successes_ref + successes_cur) / (n_ref + n_cur)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_ref + 1 / n_cur))
    if se == 0:
        return TestOutcome("two_proportion_z", 0.0, 1.0, diff, diff, n_ref, n_cur)
    correction = 0.5 * (1 / n_ref + 1 / n_cur)
    z = (abs(diff) - min(correction, abs(diff))) / se
    p = float(2 * stats.norm.sf(z))
    return TestOutcome(
        test="two_proportion_z",
        statistic=float(np.sign(diff) * z),
        p_value=p,
        effect_size=diff,
        effect_raw=diff,
        n_ref=n_ref,
        n_cur=n_cur,
    )
