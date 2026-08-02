"""Population Stability Index — an industry heuristic, NOT a hypothesis test.

PSI is included because practitioners know it, but dedrift never presents it
as statistical evidence: it carries no p-value, does not enter FDR, and is
labeled a heuristic in every report. Conventional reading: PSI < 0.1 stable,
0.1-0.25 moderate shift, > 0.25 major shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25


def psi_null_expectation(n_ref: int, n_cur: int, n_bins: int = 10) -> float:
    """Approximate E[PSI] under the null, from finite-sample noise alone.

    PSI between two samples of the SAME distribution is not zero: to first
    order it behaves like a scaled chi-square with
    ``E[PSI] ~ (B - 1) * (1/n_ref + 1/n_cur)``. At canary scale (tens of
    records per window) this expectation alone can exceed the 0.1 / 0.25
    folk thresholds — e.g. B=10, n_ref=30, n_cur=10 gives ~1.2, guaranteed
    "major" on unchanged data. PSI is a large-sample production-traffic
    index; the check pipeline uses this function to refuse to emit PSI
    flags where the index is dominated by its own sampling noise.

    Args:
        n_ref: Reference-window sample size.
        n_cur: Current-window sample size.
        n_bins: Number of bins used.

    Returns:
        The first-order null expectation of PSI.
    """
    if n_ref <= 0 or n_cur <= 0:
        return float("inf")
    return (n_bins - 1) * (1.0 / n_ref + 1.0 / n_cur)


@dataclass(frozen=True)
class PsiResult:
    """PSI of a current window against golden-baseline bins.

    Attributes:
        value: The index value.
        label: ``"stable"`` / ``"moderate"`` / ``"major"`` per convention.
        bins: The bin edges used (fixed from the golden baseline).
    """

    value: float
    label: str
    bins: tuple[float, ...]


def psi(
    golden: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    n_bins: int = 10,
    bins: tuple[float, ...] | None = None,
) -> PsiResult:
    """Compute PSI with bins fixed from the golden baseline (SPEC.md §6).

    Deciles of the golden window define the bins; pass ``bins`` to reuse
    previously frozen edges. Empty bins are smoothed with a small epsilon.

    Args:
        golden: Golden-baseline values (defines bins when ``bins`` is None).
        current: Current-window values.
        n_bins: Number of quantile bins (default 10, the standard).
        bins: Frozen bin edges from an earlier call, if any.

    Returns:
        The heuristic index with its conventional label.
    """
    if bins is None:
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(golden, quantiles)
        edges = np.unique(edges)
        if len(edges) < 3:  # degenerate distribution; single wide bin pair
            edges = np.array([-np.inf, float(np.median(golden)), np.inf])
        edges[0] = -np.inf
        edges[-1] = np.inf
    else:
        edges = np.asarray(bins)

    eps = 1e-4
    g_frac = np.histogram(golden, bins=edges)[0] / max(len(golden), 1)
    c_frac = np.histogram(current, bins=edges)[0] / max(len(current), 1)
    g_frac = np.clip(g_frac, eps, None)
    c_frac = np.clip(c_frac, eps, None)
    g_frac = g_frac / g_frac.sum()
    c_frac = c_frac / c_frac.sum()
    value = float(np.sum((c_frac - g_frac) * np.log(c_frac / g_frac)))

    if value >= PSI_MAJOR:
        label = "major"
    elif value >= PSI_MODERATE:
        label = "moderate"
    else:
        label = "stable"
    return PsiResult(value=value, label=label, bins=tuple(float(e) for e in edges))
