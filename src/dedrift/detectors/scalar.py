"""Batch two-sample tests for scalar and rate signatures (SPEC.md §6).

Each test returns a :class:`TestOutcome` carrying the statistic, the p-value,
and effect sizes in both standardized and original units. Validity note: the
canary design is balanced by construction (both windows contain the same
canaries with the same repetition count), so under the strong null hypothesis
of "no change anywhere in the stack" the pooled per-family samples are
exchangeable across windows and the two-sample tests below apply. Detection
power for shifts confined to a few canaries is correspondingly lower than for
family-wide shifts; the docs state this rather than hiding it.
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


def _degenerate(ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64]) -> bool:
    return len(ref) < 2 or len(cur) < 2 or (np.ptp(ref) == 0 and np.ptp(cur) == 0)


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
    if _degenerate(ref, cur):
        return TestOutcome("ks", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
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
    if _degenerate(ref, cur):
        return TestOutcome("ad", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*p-value capped.*")
        warnings.filterwarnings("ignore", message=".*p-value floored.*")
        warnings.filterwarnings("ignore", message=".*midrank.*")  # SciPy 1.19 rename
        res = stats.anderson_ksamp([ref, cur])
    return TestOutcome(
        test="ad",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=_cohens_d(ref, cur),
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
    if _degenerate(ref, cur):
        return TestOutcome("welch_t", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
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

    Effect size is the variance ratio ``var(cur)/var(ref)`` (1 = no change);
    ``effect_raw`` is the variance difference.

    Args:
        ref: Reference-window values.
        cur: Current-window values.

    Returns:
        Outcome for the dispersion channel.
    """
    if _degenerate(ref, cur):
        return TestOutcome("levene", float("nan"), float("nan"), 1.0, 0.0, len(ref), len(cur))
    res = stats.levene(ref, cur, center="median")
    v_ref = float(np.var(ref, ddof=1))
    v_cur = float(np.var(cur, ddof=1))
    ratio = v_cur / v_ref if v_ref > 0 else float("inf")
    return TestOutcome(
        test="levene",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size=ratio,
        effect_raw=v_cur - v_ref,
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
    ``(b+1)/(B+1)``, which is exact-level under exchangeability of the pooled
    sample. This replaces an earlier centered percentile bootstrap: bootstrap
    approximations of extreme-quantile nulls are unreliable at these sample
    sizes (the P95 rests on a handful of order statistics), while the
    permutation null is exact. Effect size is the relative P95 shift.

    Args:
        ref: Reference-window values.
        cur: Current-window values.
        n_permutations: Number of label permutations (seeded).
        seed: RNG seed (recorded in the report).

    Returns:
        Outcome for the tail-behavior channel.
    """
    if _degenerate(ref, cur):
        return TestOutcome("p95_perm", float("nan"), float("nan"), 0.0, 0.0, len(ref), len(cur))
    rng = np.random.default_rng(seed)
    n, m = len(ref), len(cur)
    p95_ref = float(np.percentile(ref, 95))
    p95_cur = float(np.percentile(cur, 95))
    observed = p95_cur - p95_ref
    pooled = np.concatenate([ref, cur])
    # Vectorized label permutations: each row is a random ordering of pooled.
    order = np.argsort(rng.random((n_permutations, n + m)), axis=1)
    shuffled = pooled[order]
    diffs = np.percentile(shuffled[:, n:], 95, axis=1) - np.percentile(shuffled[:, :n], 95, axis=1)
    p = float((np.sum(np.abs(diffs) >= abs(observed)) + 1) / (n_permutations + 1))
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
