"""Page-Hinkley sequential change detection on per-cycle means (SPEC.md §6).

Interpretation of the parameters (documented per SPEC):

- ``delta`` is the magnitude of per-observation drift the statistic tolerates
  before accumulating evidence — the "dead zone". In standardized units
  (the stream is z-scored against its early observations), ``delta=0.3``
  means per-cycle shifts below 0.3 reference standard deviations are ignored.
- ``lambda_`` is the alarm threshold on the accumulated excursion, in the same
  standardized units. Under the null with KNOWN centering and scale the
  excursion is a random walk with drift ``-delta`` and the crossing
  probability is bounded by ``exp(-2 * delta * lambda_)`` per direction
  (~0.15% two-sided at the defaults ``lambda_=12``, ``delta=0.3``). Because
  centering and scale are estimated from the stream itself, the MEASURED
  null alarm rate is higher: ~1.5% per stream over 30-cycle horizons (see
  the calibration test, which enforces < 3%). We state this honestly rather
  than pretending the idealized bound applies. Note the stream is per-cycle
  MEANS: a per-record shift of d standard deviations is
  ``d * sqrt(n_records_per_cycle)`` in stream units, so material shifts cross
  the threshold within a few cycles.

Page-Hinkley is a sequential procedure and yields no batch p-value. Its alarm
enters the gating pipeline as a FLAG (SPEC §6): it must still pass the
materiality gate to alert, and it primarily serves attribution by providing a
change-point estimate (the cycle where the accumulated statistic was last at
its extremum before the alarm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PageHinkleyResult:
    """Result of a Page-Hinkley scan over an ordered stream.

    Attributes:
        alarm: True if either direction crossed ``lambda_``.
        direction: ``"up"``, ``"down"``, or ``""`` when no alarm.
        change_index: Index in the stream where the pre-alarm extremum was
            attained (best onset estimate); None when no alarm.
        statistic: The maximum accumulated deviation reached (standardized).
        lambda_: Threshold used.
        delta: Dead-zone used.
    """

    alarm: bool
    direction: str
    change_index: int | None
    statistic: float
    lambda_: float
    delta: float


def page_hinkley(
    values: npt.NDArray[np.float64],
    lambda_: float = 12.0,
    delta: float = 0.3,
    min_baseline: int = 5,
) -> PageHinkleyResult:
    """Run a two-sided Page-Hinkley scan over an ordered scalar stream.

    The stream (typically per-cycle means of one signature) is standardized
    by the mean and standard deviation of its first ``min_baseline``
    observations, so ``lambda_`` and ``delta`` are in reference-SD units.

    Args:
        values: Ordered stream (one value per cycle).
        lambda_: Alarm threshold in standardized units.
        delta: Per-observation tolerance in standardized units.
        min_baseline: Observations used to standardize; no alarm can fire
            before this many observations.

    Returns:
        The scan result, including a change-point estimate when alarmed.
    """
    n = len(values)
    if n <= min_baseline:
        return PageHinkleyResult(False, "", None, 0.0, lambda_, delta)
    base = values[:min_baseline]
    # Scale from the median absolute successive difference over the whole
    # stream: successive differences are level-shift robust (only the single
    # difference straddling a change point is contaminated, and the median
    # ignores it), and using the full stream gives a far stabler estimate
    # than the tiny baseline window would.
    diffs = np.diff(values)
    scale = float(np.median(np.abs(diffs - np.median(diffs)))) * 1.4826 / np.sqrt(2)
    if scale == 0:
        scale = float(np.std(values, ddof=1)) or 1.0
    z = (values - float(np.mean(base))) / scale

    # Accumulators for upward and downward shifts.
    running_mean = 0.0
    m_up = 0.0
    min_up = 0.0
    m_down = 0.0
    max_down = 0.0
    argmin_up = 0
    argmax_down = 0
    best_stat = 0.0
    for t in range(n):
        running_mean += (z[t] - running_mean) / (t + 1)
        m_up += z[t] - running_mean - delta
        m_down += z[t] - running_mean + delta
        if m_up < min_up:
            min_up = m_up
            argmin_up = t
        if m_down > max_down:
            max_down = m_down
            argmax_down = t
        stat_up = m_up - min_up
        stat_down = max_down - m_down
        best_stat = max(best_stat, stat_up, stat_down)
        if t >= min_baseline:
            if stat_up > lambda_:
                return PageHinkleyResult(True, "up", argmin_up, stat_up, lambda_, delta)
            if stat_down > lambda_:
                return PageHinkleyResult(True, "down", argmax_down, stat_down, lambda_, delta)
    return PageHinkleyResult(False, "", None, best_stat, lambda_, delta)
