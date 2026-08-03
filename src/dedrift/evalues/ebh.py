"""e-BH: FDR control from e-values, under arbitrary dependence.

Wang & Ramdas (*False discovery rate control with e-values*, JRSS-B 84(3),
2022; arXiv:2009.02824). Sort e-values decreasing, find the largest ``k``
with ``E_(k) >= m / (alpha * k)``, reject the top ``k``. FDR <= alpha for
**any** dependence structure among the e-values — which is why this
replaces Benjamini-Hochberg here rather than sitting beside it: our battery
shares the current cycle across baselines and the golden sample across
channels, and the PRDS condition BH needs was only ever asserted.

The subtlety we do not paper over
---------------------------------
e-BH's guarantee is stated for a *fixed* collection of e-values. Applying
it at every cycle to running e-processes is a different object. Per
*Anytime-valid FDR control with the stopped e-BH procedure*
(arXiv:2502.08539, Stat. & Prob. Letters 2025): adaptively stopped
e-processes are e-values, so e-BH may be applied at every step, and the
stopped procedure controls FDR at all stopping times **if the streams are
independent** — which ours are not. In general each stopped e-process is an
e-value only for stopping times in its own local filtration, while the
procedure stops with respect to the global filtration, which can leak
information across time. The paper gives a causal condition (no unobserved
confounding from the past) under which local e-processes are global ones
and the guarantee is restored.

Our streams plausibly satisfy it — every stream is a function of the same
observable cycle history, and cross-sectional dependence within a cycle is
exactly what e-BH already tolerates. But it is an assumption, not a
theorem we have proved for this battery, and the honest reading of the
candidate violation is unobserved provider-side state (rolling
deployments, load) correlating streams through time. Whether that is
excluded is close to being the definition of the null. Accordingly:
anything the docs say about trajectory-level FDR must be labelled as
assumption-plus-measurement, never as established.
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
            cannot silently void the battery.
        alpha: FDR level.

    Returns:
        The rejection set and per-hypothesis adjusted levels.
    """
    if not evalues:
        return EBHResult([], 0, float("inf"), [])
    e = np.asarray(evalues, dtype=float)
    e = np.where(np.isfinite(e), e, 0.0)
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
