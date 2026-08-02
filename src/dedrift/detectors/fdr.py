"""Benjamini-Hochberg false discovery rate control (SPEC.md §6, principle 3).

All p-valued tests in a check pass through here before any alert can exist.
BH is valid under independence and positive regression dependence (PRDS);
our per-family test batteries are positively correlated (KS/AD/Welch on the
same data), which PRDS covers.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> tuple[list[bool], list[float]]:
    """Apply the BH step-up procedure.

    NaN p-values (degenerate tests) are never rejected and receive adjusted
    p-value NaN; they do not count toward the number of tests m.

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
    valid = np.where(~np.isnan(p))[0]
    m = len(valid)
    if m == 0:
        return rejected, adjusted

    order = valid[np.argsort(p[valid])]
    sorted_p = p[order]
    ranks = np.arange(1, m + 1)

    # Adjusted p-values: monotone step-up.
    adj = sorted_p * m / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    threshold_idx = np.where(sorted_p <= ranks * q / m)[0]
    k = threshold_idx.max() + 1 if len(threshold_idx) else 0

    for rank_pos, original_idx in enumerate(order):
        adjusted[int(original_idx)] = float(adj[rank_pos])
        if rank_pos < k:
            rejected[int(original_idx)] = True
    return rejected, adjusted
