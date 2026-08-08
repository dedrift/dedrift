"""Cycle-effect-adaptive inference for the fixed check path.

The per-record two-sample battery assumes exchangeability within each
window. A hosted model behind a stable alias violates exactly that: every
record of a cycle shares a latent offset (load, routing, cache state,
rolling deployments), so records cluster BY CYCLE. Measured on the
independent audit's ladder (v0.3.1, 30 canaries x 7 reps, 3-cycle golden):
a latent per-cycle offset of sigma=0.05 lifts the per-check false-alert
rate from 2.3% to 33.5%; sigma=0.15 gives 87.5%.

This module estimates the clustering per channel from HISTORY cycles only
(never the cycle under test) and, when the estimated intraclass correlation
exceeds the configured threshold, replaces the record-level p-value with a
cluster-aware one:

* KS becomes a disjunctive composite: KS on within-cycle-standardized
  values (shape; per-cycle offsets cancel exactly, so record-level power
  survives) OR a cycle-mean Gaussian summary test (location), combined by
  Bonferroni-min. The reported effect adds the location shift expressed in
  D units so the materiality gate keeps its meaning.
* two-proportion z: each arm's rate variance is inflated by its Kish
  design effect 1+(m-1)*rho, m = records per cycle; the single-cycle
  current arm is fully clustered by construction.
* Levene and P95-permutation: no honest effective-n form exists for a
  permutation null under clustering, so the cycle-level summary test is
  used instead: per-cycle values of the statistic (MAD / P95 / mean) over
  the baseline's reference cycles give an empirical null center and spread
  that INCLUDE the cycle effect by construction; the current cycle's value
  is judged against them with a Gaussian approximation. (Reference windows
  with fewer than three cycles fall back to all-history cycles for the
  spread, accepting dilution, documented in docs/statistics.md.)

These are documented approximations, not exact tests: their null behavior
is measured on the audit sigma-ladder and the measured band is published
(next to the exactly-calibrated sigma=0 path, whose record-level p-value
the correction never undercuts — the corrected p enters through a max()
with the record-level one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.stats import t as t_dist

#: Minimum reference cycles needed to estimate clustering.
MIN_REFERENCE_CYCLES = 3


@dataclass(frozen=True)
class CycleEffectEstimate:
    """Estimated per-channel clustering in the reference window.

    Attributes:
        rho: Intraclass correlation estimate (share of variance due to
            per-cycle offsets), method-of-moments, signed (unclamped below:
            the exchangeable null estimates a symmetric-about-zero value).
        n_cycles: History cycles used.
        engaged: True when rho exceeds the configured threshold and the
            corrected p-value should replace the record-level one.
    """

    rho: float
    n_cycles: int
    engaged: bool


def estimate_icc(
    ref_values: npt.NDArray[np.float64],
    ref_cycle_ids: npt.NDArray[Any],
    *,
    threshold: float,
) -> CycleEffectEstimate:
    """Method-of-moments ICC of one signature over history cycles.

    sigma_b^2 = Var_cyc(means) - mean_cyc(s_c^2 / n_c) subtracts the
    sampling noise of the per-cycle means. The estimate is deliberately NOT
    clamped at zero: under exact exchangeability it is symmetric about 0,
    so with the engagement threshold near 0 about half of stable channels
    report a small positive rho and pay a small design-effect penalty --
    the measured, accepted cost of engagement readiness. What matters at
    canary scale is not the SHARE of record variance (ICC is small even
    under strong wobble, because within-cycle variance is large) but the
    design effect 1+(m-1)*rho with m~35: rho = 0.05 already cuts the
    single-cycle current arm's effective n almost threefold.
    """
    df = per_cycle_triples(ref_values, ref_cycle_ids)
    k = len(df)
    if k < MIN_REFERENCE_CYCLES:
        return CycleEffectEstimate(0.0, k, False)
    means = np.array([g[0] for g in df])
    noise = np.array([g[1] for g in df])  # s_c^2 / n_c
    within = np.array([g[2] for g in df])  # s_c^2
    sigma_w2 = float(np.mean(within))
    # Robust spread of the per-cycle means: a trailing minority of DRIFTED
    # cycles must not inflate the clustering estimate (a plain variance
    # absorbs the drift and the correction disarms itself exactly when a
    # real drift is present — measured). The 1.4826-scaled MAD of the cycle
    # means is consistent for the spread under the clean majority.
    med = float(np.median(means))
    mad = float(np.median(np.abs(means - med)))
    var_means = (1.4826 * mad) ** 2
    sigma_b2 = var_means - float(np.mean(noise))  # signed: no zero clamp
    denom = max(sigma_b2, 0.0) + sigma_w2
    rho = 0.0 if denom <= 0 else max(min(sigma_b2 / denom, 0.999), -0.999)
    return CycleEffectEstimate(rho, k, rho > threshold)


def per_cycle_triples(
    values: npt.NDArray[np.float64], cycle_ids: npt.NDArray[Any]
) -> list[tuple[float, float, float]]:
    """Per-cycle (mean, s_c^2/n_c, s_c^2) triples, ddof=1 within cycle."""
    out: list[tuple[float, float, float]] = []
    for c in dict.fromkeys(cycle_ids.tolist()):
        v = values[cycle_ids == c]
        v = v[np.isfinite(v)]
        n = len(v)
        if n < 2:
            continue
        s2 = float(np.var(v, ddof=1))
        out.append((float(np.mean(v)), s2 / n, s2))
    return out


def _design_effect(n: int, rho: float, m: float) -> float:
    """Kish design effect for a clustered arm of n records, m per cycle.

    Floored at 1.0: the correction may not sharpen a test below the
    record-level effective size.
    """
    if n <= 1 or m <= 1:
        return 1.0
    return max(1.0, 1.0 + (m - 1.0) * rho)


def welch_pvalue_clustered(
    ref: npt.NDArray[np.float64],
    cur: npt.NDArray[np.float64],
    rho: float,
    m_ref: float,
    m_cur: float,
) -> float:
    """Welch t p-value with design-effect-inflated arm variances.

    The Kish design effect IS the correct correction for a mean functional
    under clustering (unlike the KS sup-statistic), so the engaged
    composite carries location with this test: per-arm variances are
    inflated by 1+(m-1)*rho and the df are Satterthwaite on the inflated
    variances with effective arm sizes.
    """
    n1, n2 = len(ref), len(cur)
    if n1 < 2 or n2 < 2:
        return float("nan")
    d1 = _design_effect(n1, rho, m_ref)
    d2 = _design_effect(n2, rho, m_cur)
    v1 = d1 * float(np.var(ref, ddof=1)) / n1
    v2 = d2 * float(np.var(cur, ddof=1)) / n2
    if v1 + v2 <= 0:
        return 1.0 if float(np.mean(ref)) == float(np.mean(cur)) else 0.0
    t_stat = (float(np.mean(cur)) - float(np.mean(ref))) / math.sqrt(v1 + v2)
    n1e = n1 / d1
    n2e = n2 / d2
    denom1 = max(n1e - 1.0, 1e-9)
    denom2 = max(n2e - 1.0, 1e-9)
    df = (v1 + v2) ** 2 / (v1 * v1 / denom1 + v2 * v2 / denom2)
    df = max(df, 1.0)
    return float(min(1.0, 2.0 * t_dist.sf(abs(t_stat), df=df)))


def rate_z_pvalue_clustered(
    successes_ref: int,
    n_ref: int,
    successes_cur: int,
    n_cur: int,
    rho: float,
    m_ref: float,
    m_cur: float,
) -> float:
    """Two-proportion z p-value with design-effect-inflated variances."""
    if n_ref == 0 or n_cur == 0:
        return float("nan")
    p1 = successes_ref / n_ref
    p2 = successes_cur / n_cur
    pooled = (successes_ref + successes_cur) / (n_ref + n_cur)
    d1 = _design_effect(n_ref, rho, m_ref)
    d2 = _design_effect(n_cur, rho, m_cur)
    var = pooled * (1 - pooled) * (d1 / n_ref + d2 / n_cur)
    if var <= 0:
        return 1.0
    se = math.sqrt(var)
    diff = p2 - p1
    correction = min(0.5 * (d1 / n_ref + d2 / n_cur), abs(diff))
    z = (abs(diff) - correction) / se
    return float(min(1.0, 2.0 * norm.sf(z)))


def cycle_level_pvalue(
    ref_stat_values: npt.NDArray[np.float64],
    cur_stat: float,
) -> float:
    """Prediction-interval p-value for one cycle-level statistic (two-sided).

    The reference cycles' per-cycle values supply the null center AND
    spread; the current cycle's value is judged against the prediction
    interval center +/- t_{K-1} * sd * sqrt(1+1/K) — Student t with K-1 df,
    the exact form under Gaussian cycle statistics (a normal approximation
    here would be anticonservative at K=3-5, where the spread estimate is
    the dominant uncertainty). Cycle effects are inside the spread because
    each reference value is one cycle's own statistic.
    """
    ref = ref_stat_values[np.isfinite(ref_stat_values)]
    if len(ref) < MIN_REFERENCE_CYCLES or not math.isfinite(cur_stat):
        return float("nan")
    center = float(np.mean(ref))
    sd = float(np.std(ref, ddof=1))
    if sd <= 0:
        # A constant reference gives no spread estimate: "uncomputable",
        # never "infinitely significant" (a 0/1 P95 flipping cycles by
        # sampling noise measured as p=0 false alerts). The caller keeps
        # the record-level p-value on NaN.
        return float("nan")
    z = (cur_stat - center) / (sd * math.sqrt(1.0 + 1.0 / len(ref)))
    return float(min(1.0, 2.0 * t_dist.sf(abs(z), df=len(ref) - 1)))


def per_cycle_statistic(
    values: npt.NDArray[np.float64],
    cycle_ids: npt.NDArray[Any],
    kind: str,
) -> npt.NDArray[np.float64]:
    """Per-cycle values of a summary statistic ("p95" or "mad" or "mean")."""
    stats: list[float] = []
    for c in dict.fromkeys(cycle_ids.tolist()):
        v = values[cycle_ids == c]
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        if kind == "p95":
            stats.append(float(np.percentile(v, 95)))
        elif kind == "mad":
            med = float(np.median(v))
            stats.append(float(np.mean(np.abs(v - med))))
        elif kind == "mean":
            stats.append(float(np.mean(v)))
        else:  # pragma: no cover - internal guard
            raise ValueError(f"unknown per-cycle statistic {kind!r}")
    return np.asarray(stats, dtype=float)


def standardize_within_cycle(
    values: npt.NDArray[np.float64],
    cycle_ids: npt.NDArray[Any],
) -> npt.NDArray[np.float64]:
    """Z-score each record within its own cycle (mean 0, sd 1 per cycle).

    A latent per-cycle offset shifts every record of a cycle together, so
    it cancels exactly in within-cycle-standardized values: under the null
    they are exchangeable across cycles even with provider wobble, and the
    KS test on them keeps full record-level power for SHAPE changes. What
    it cannot see is location (every cycle is centered) — that hypothesis
    moves to the cycle-mean summary test, which is where clustered data
    says it must live.
    """
    out = np.full(len(values), np.nan)
    for c in dict.fromkeys(cycle_ids.tolist()):
        mask = cycle_ids == c
        v = values[mask]
        finite = np.isfinite(v)
        vv = v[finite]
        if len(vv) < 2:
            continue
        sd = float(np.std(vv, ddof=1))
        z = (vv - float(np.mean(vv))) / sd if sd > 0 else np.zeros(len(vv))
        sub = out[mask]
        sub[finite] = z
        out[mask] = sub
    return out
