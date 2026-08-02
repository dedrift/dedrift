"""MMD-RBF two-sample test with a seeded permutation null (SPEC.md §6).

Implementation notes:

- RBF kernel with the POOLED median heuristic for bandwidth. The pooled
  median is a permutation-invariant function of the combined sample, which
  is what makes the permutation p-value exact-level under exchangeability:
  a reference-only bandwidth would make the kernel depend on the original
  labelling and the permutation distribution would no longer be the exact
  null. (The median heuristic is not a tuned parameter, so pooling is not
  "peeking" at the alternative — this deviates deliberately from an earlier
  reading of the spec, with the owner's sign-off.)
- Permutation test with >= 500 permutations, seeded; the p-value uses the
  add-one convention (b+1)/(B+1).
- Reports the biased (V-statistic) MMD^2 both as statistic and effect size;
  the permutation null makes the bias irrelevant for inference, and the
  biased estimate is non-negative, which makes floors interpretable.
- The materiality floor is calibrated with the SAME bandwidth as the
  observed statistic (pass ``sigma``), so calibrated floor and observed
  MMD^2 are commensurable numbers under one kernel.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from dedrift.detectors.scalar import TestOutcome


def _sq_dists(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d2 = aa + bb - 2.0 * (a @ b.T)
    result: npt.NDArray[np.float64] = np.maximum(d2, 0.0)
    return result


def median_heuristic_bandwidth(sample: npt.NDArray[np.float64]) -> float:
    """Median pairwise distance of a sample (use the POOLED sample for tests).

    Pooling keeps the kernel permutation-invariant, which the exactness of
    the permutation p-value requires.

    Args:
        sample: Embeddings, shape (n, d).

    Returns:
        Bandwidth sigma (median of pairwise Euclidean distances); falls back
        to 1.0 for degenerate samples.
    """
    n = len(sample)
    if n < 2:
        return 1.0
    d2 = _sq_dists(sample, sample)
    upper = d2[np.triu_indices(n, k=1)]
    med = float(np.sqrt(np.median(upper)))
    return med if med > 0 else 1.0


def _mmd2_from_kernel(
    k: npt.NDArray[np.float64],
    idx_x: npt.NDArray[np.intp],
    idx_y: npt.NDArray[np.intp],
) -> float:
    kxx = k[np.ix_(idx_x, idx_x)].mean()
    kyy = k[np.ix_(idx_y, idx_y)].mean()
    kxy = k[np.ix_(idx_x, idx_y)].mean()
    return float(kxx + kyy - 2.0 * kxy)


def mmd_rbf_test(
    ref: npt.NDArray[np.float64],
    cur: npt.NDArray[np.float64],
    n_permutations: int = 500,
    seed: int = 0,
    sigma: float | None = None,
) -> TestOutcome:
    """MMD^2 RBF two-sample test with a seeded permutation null.

    Args:
        ref: Reference-window embeddings, shape (n, d).
        cur: Current-window embeddings, shape (m, d).
        n_permutations: Permutation count (SPEC minimum 500).
        seed: RNG seed, recorded in the report.
        sigma: Kernel bandwidth; defaults to the pooled median heuristic
            (permutation-invariant, keeping the p-value exact-level).

    Returns:
        Outcome with ``statistic = effect_size = effect_raw = MMD^2`` and the
        permutation p-value.
    """
    n, m = len(ref), len(cur)
    if n < 2 or m < 2:
        return TestOutcome("mmd", float("nan"), float("nan"), 0.0, 0.0, n, m)
    pooled = np.vstack([ref, cur])
    if sigma is None:
        sigma = median_heuristic_bandwidth(pooled)
    k = np.exp(-_sq_dists(pooled, pooled) / (2.0 * sigma * sigma))

    idx = np.arange(n + m)
    observed = _mmd2_from_kernel(k, idx[:n], idx[n:])

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n + m)
        if _mmd2_from_kernel(k, perm[:n], perm[n:]) >= observed:
            exceed += 1
    p = (exceed + 1) / (n_permutations + 1)
    return TestOutcome(
        test="mmd",
        statistic=observed,
        p_value=float(p),
        effect_size=observed,
        effect_raw=observed,
        n_ref=n,
        n_cur=m,
    )


def calibrate_mmd_floor(
    cycle_embeddings: list[npt.NDArray[np.float64]],
    sigma: float,
    quantile: float = 0.95,
) -> float:
    """Calibrate the MMD^2 materiality floor from known-same reference cycles.

    Computes MMD^2 between every ordered pair of reference cycles — which are
    known-good and should differ only by sampling noise — and returns the
    given quantile of that empirical null. All pairs use the SAME bandwidth
    ``sigma`` as the observed statistic, so the floor and the observation are
    commensurable numbers under one kernel. An observed MMD^2 below the floor
    is within the project's own cycle-to-cycle noise and is never material,
    whatever its p-value. Requires >= 3 cycles (>= 3 pairs); returns 0.0
    otherwise (floor disabled, and the report should say so).

    Args:
        cycle_embeddings: One (n_i, d) embedding array per reference cycle.
        sigma: Kernel bandwidth (use the one from the observed test).
        quantile: Quantile of the pairwise null distribution to use.

    Returns:
        The calibrated floor (0.0 when not enough cycles to calibrate).
    """
    k = len(cycle_embeddings)
    if k < 3:
        return 0.0
    values: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = cycle_embeddings[i], cycle_embeddings[j]
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = np.vstack([a, b])
            kmat = np.exp(-_sq_dists(pooled, pooled) / (2.0 * sigma * sigma))
            idx = np.arange(len(a) + len(b))
            values.append(_mmd2_from_kernel(kmat, idx[: len(a)], idx[len(a) :]))
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values), quantile))
