"""Benjamini-Hochberg false discovery rate control (SPEC.md §6, principle 3).

All primary p-valued tests in a check pass through here before any alert can
exist. BH is valid under independence and PRDS, but this package's shared,
two-sided test battery has no established PRDS result. The default pipeline's
simulation calibration is therefore the operative evidence; this function
must not be described as arbitrary-dependence FDR control. Corroboration tests
(AD/Welch) are excluded from the pool upstream.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> tuple[list[bool], list[float]]:
    """Apply the BH step-up procedure.

    NaN p-values (degenerate declared tests) are conservatively treated as
    p=1 when determining the family size, but remain unrejected and display
    adjusted p-value NaN. Dropping them would make multiplicity depend on the
    observed data and can make the procedure more liberal.

    Args:
        p_values: Raw p-values, one per test.
        q: Target false discovery rate.

    Returns:
        A tuple ``(rejected, p_adjusted)`` aligned with the input order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    rejected = [False] * n
    adjusted: list[float] = [float("nan")] * n
    valid_mask = ~np.isnan(p)
    m = n
    if m == 0:
        return rejected, adjusted

    working = np.where(valid_mask, p, 1.0)
    order = np.argsort(working)
    sorted_p = working[order]
    ranks = np.arange(1, m + 1)

    # Adjusted p-values: monotone step-up.
    adj = sorted_p * m / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    threshold_idx = np.where(sorted_p <= ranks * q / m)[0]
    k = threshold_idx.max() + 1 if len(threshold_idx) else 0

    for rank_pos, original_idx in enumerate(order):
        if valid_mask[original_idx]:
            adjusted[int(original_idx)] = float(adj[rank_pos])
        if rank_pos < k and valid_mask[original_idx]:
            rejected[int(original_idx)] = True
    return rejected, adjusted
