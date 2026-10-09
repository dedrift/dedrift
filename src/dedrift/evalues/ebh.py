"""e-BH utility for fixed-collection FDR control from e-values.

Wang & Ramdas (*False discovery rate control with e-values*, JRSS-B 84(3),
2022; arXiv:2009.02824). Sort e-values decreasing, find the largest ``k``
with ``E_(k) >= m / (alpha * k)``, reject the top ``k``. FDR <= alpha for
**any** dependence structure among a fixed collection of e-values.

The subtlety we do not paper over
---------------------------------
e-BH's guarantee is stated for a *fixed* collection of e-values. Reapplying
it after every cycle to dependent running e-processes needs additional
stopped-e-BH conditions that this package has not established. The production
monitor therefore uses a predeclared, fixed pool with Ville--Bonferroni family-
wise control. This function remains available for fixed-collection analysis;
it is not the production sequential decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EBHResult:
    """Outcome of one e-BH application.

    Attributes:
        rejected: Boolean mask over the input order.
        n_rejected: ``k``, the number rejected.
        threshold: The e-value threshold ``m / (alpha * k)`` that was met;
            ``inf`` when nothing is rejected.
        e_adjusted: Per-hypothesis quantity playing the role BH-adjusted
            p-values play in reports: ``min(1, m / (alpha_needed))`` is not
            well defined for e-values, so we report the smallest ``alpha``
            at which each hypothesis would be rejected. Smaller is stronger,
            which keeps report semantics aligned with the p-value path.
    """

    rejected: list[bool]
    n_rejected: int
    threshold: float
    e_adjusted: list[float]


def ebh(evalues: list[float], alpha: float = 0.05) -> EBHResult:
    """Apply the e-BH procedure.

    Args:
        evalues: E-values (non-negative). NaN is treated as 0 — no
            evidence — rather than propagating, so one degenerate channel
            cannot silently void the battery. ``+inf`` is the opposite
            case: an e-value that overflowed float64 is the STRONGEST
            possible evidence, so it is clamped to the largest finite
            double (an earlier version mapped every non-finite value to 0,
            turning infinite evidence into no evidence).
        alpha: FDR level.

    Returns:
        The rejection set and per-hypothesis adjusted levels.
    """
    if not evalues:
        return EBHResult([], 0, float("inf"), [])
    e = np.asarray(evalues, dtype=float)
    e = np.where(np.isnan(e), 0.0, e)
    e = np.where(np.isposinf(e), np.finfo(np.float64).max, e)
    e = np.where(np.isneginf(e), 0.0, e)
    e = np.maximum(e, 0.0)
    m = len(e)

    order = np.argsort(-e)  # decreasing
    sorted_e = e[order]
    ks = np.arange(1, m + 1)
    meets = sorted_e >= m / (alpha * ks)
    k = int(ks[meets].max()) if bool(meets.any()) else 0

    rejected = np.zeros(m, dtype=bool)
    threshold = float("inf")
    if k > 0:
        rejected[order[:k]] = True
        threshold = float(m / (alpha * k))

    # Smallest alpha at which each hypothesis would be rejected. Computed by
    # the same rule, so it is consistent with the decision rather than an
    # independent approximation of it.
    adjusted = np.ones(m)
    for i in range(m):
        lo, hi = 1e-12, 1.0
        if not _rejects(e, i, hi):
            adjusted[i] = 1.0
            continue
        for _ in range(60):
            mid = (lo + hi) / 2
            if _rejects(e, i, mid):
                hi = mid
            else:
                lo = mid
        adjusted[i] = hi
    return EBHResult(rejected.tolist(), k, threshold, adjusted.tolist())


def ebh_from_logs(log_evalues: list[float], alpha: float = 0.05) -> EBHResult:
    """e-BH on ``log E`` — the form to use on real e-processes.

    A wealth process that has been accumulating for a few hundred cycles
    overflows float64 long before it stops being interesting
    (``exp(710) = inf``), at which point :func:`ebh` compares infinities and
    the rejection set becomes arbitrary. The decision rule is
    scale-monotone, so we can apply it entirely in logs: sorting by
    ``log E`` is sorting by ``E``, and ``E_(k) >= m/(alpha k)`` is
    ``log E_(k) >= log m - log alpha - log k``.

    Args:
        log_evalues: ``log E`` per hypothesis; ``-inf`` allowed.
        alpha: FDR level.

    Returns:
        The rejection set. ``e_adjusted`` is left as ones here — adjusted
        levels are for reports, and reports render finite e-values; use
        :func:`ebh` when they are needed.
    """
    if not log_evalues:
        return EBHResult([], 0, float("inf"), [])
    le = np.asarray(log_evalues, dtype=float)
    le = np.where(np.isnan(le), -np.inf, le)
    m = len(le)

    order = np.argsort(-le)
    sorted_le = le[order]
    ks = np.arange(1, m + 1)
    need = np.log(m) - np.log(alpha) - np.log(ks)
    meets = sorted_le >= need
    k = int(ks[meets].max()) if bool(meets.any()) else 0

    rejected = np.zeros(m, dtype=bool)
    threshold = float("inf")
    if k > 0:
        rejected[order[:k]] = True
        threshold = float(np.exp(min(need[k - 1], 700.0)))
    return EBHResult(rejected.tolist(), k, threshold, [1.0] * m)


def _rejects(e: np.ndarray, index: int, alpha: float) -> bool:
    """Would e-BH at ``alpha`` reject hypothesis ``index``?"""
    m = len(e)
    order = np.argsort(-e)
    sorted_e = e[order]
    ks = np.arange(1, m + 1)
    meets = sorted_e >= m / (alpha * ks)
    if not bool(meets.any()):
        return False
    k = int(ks[meets].max())
    return bool(index in set(order[:k].tolist()))
