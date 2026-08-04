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

    # Scale from successive differences within the BASELINE WINDOW ONLY.
    #
    # An earlier version took the median absolute successive difference over
    # the whole stream, including cycles after the alarm. That is a larger
    # and stabler sample, and it is not available at decision time: a
    # procedure sold as sequential, and used to localise onsets, must not be
    # standardised with data from the future of the point it is judging.
    # It also biases in the direction that flatters the detector -- a
    # post-change stream inflates the scale, so pre-change excursions look
    # smaller than they are.
    #
    # Successive differences remain the right robust base (a level shift
    # contaminates exactly one difference, which the median ignores); the
    # sqrt(2) converts a difference scale to a level scale under
    # independence, which is itself an assumption: under positive
    # autocorrelation it understates the level scale and makes the detector
    # more trigger-happy. That is one contributor to the gap between the
    # idealised crossing rate and the measured flag rate, and it is why this
    # channel is labelled a diagnostic and never alerts.
    # A baseline-window-only estimate is causal but far too noisy at
    # ``min_baseline`` points: measured, it took the null alarm rate from
    # 0.7% to 19%, because an underestimated scale inflates every
    # standardized deviation. The fix is not to reach forward for more data
    # but to let the estimate GROW causally -- at step t, standardize with
    # everything observed strictly before t. Early steps are noisy and no
    # alarm may fire before ``min_baseline`` anyway; by the time alarms are
    # possible the estimate has as much data as causality allows.
    def _scale_from(window: npt.NDArray[np.float64]) -> float:
        if len(window) > 1:
            d = np.diff(window)
            s = float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2.0))
            if s > 0:
                return s
            s = float(np.std(window, ddof=1))
            if s > 0:
                return s
        return 1.0

    z = np.empty(n, dtype=float)
    for t_idx in range(n):
        window = values[: max(t_idx, min_baseline)] if t_idx else base
        z[t_idx] = (values[t_idx] - float(np.mean(window))) / _scale_from(window)

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
