"""Interval arithmetic for the benchmark.

All headline rates are binomial proportions over independent seeded runs and
carry Wilson score 95% intervals. Per-comparison rates (e.g. per PSI flag
opportunity) pool correlated comparisons within a run; their intervals treat
comparisons as the unit and are therefore anti-conservative — the run-level
intervals are the primary ones wherever both appear.
"""

from __future__ import annotations

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        k: Success count.
        n: Trial count.
        z: Standard-normal quantile (1.96 for a 95% interval).

    Returns:
        ``(point_estimate, lower, upper)``.
    """
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def rate_row(k: int, n: int) -> dict[str, float | int]:
    """One results-table row: count, denominator, rate, Wilson 95% interval."""
    p, lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_low": lo, "wilson_high": hi}
