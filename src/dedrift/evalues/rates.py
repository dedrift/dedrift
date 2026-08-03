"""Rate-channel e-values: two-sample Bernoulli, composite null.

The null is ``H0: the current cycle's success probability equals the
reference probability``, with that probability *unknown*. A likelihood
ratio against a point alternative is only an e-value once the nuisance
parameter is handled, and how we handle it depends on whether the
reference data is frozen or still accruing.

The exponential tilt, and why it is the right primitive
------------------------------------------------------
For a single cycle with ``S ~ Binomial(n, p)`` and an alternative obtained
by tilting the odds by ``psi`` (``p1/(1-p1) = psi * p/(1-p)``), the
likelihood ratio collapses to

    E(S; n, p, psi) = psi**S / (1 - p + psi * p)**n

whose denominator is exactly the binomial m.g.f., so ``E_p[E] = 1``
identically — no asymptotics, no approximation. That identity is the whole
construction; everything below is about not knowing ``p``.

Handling the unknown p: worst case over an interval we pay for
-------------------------------------------------------------
Take a ``(1 - gamma)`` interval for ``p`` computed from data available
before the cycle, and bet on its *worst* member:

    E_t = inf_{p in I} E(S_t; n, p, psi-mixture)

On the coverage event ``{p_true in I}`` we have ``E_t <= E(S_t; n, p_true,
...)``, whose conditional expectation is 1, so the product is a
supermartingale *there*; Ville bounds the trajectory at ``alpha'`` and a
union bound pays ``gamma`` for coverage. Total budget
``alpha = alpha' + gamma``.

``gamma`` is load-bearing, not a formality. Measured at our operating
scale with a Clopper-Pearson interval at ``gamma = 0.01``: the supremum of
``E[E_t]`` over ``p`` *inside* the interval is 0.73-0.91 (valid, and the
shortfall from 1 is the price of worst-casing), while over *all* ``p`` it
reaches 10**6. The guarantee holds on the coverage event and nowhere else,
which is why the interval is reported alongside the wealth.

Two regimes, deliberately different
-----------------------------------
* **Golden (frozen reference).** The interval is computed once, from the
  frozen sample, so it is a *fixed* interval: the coverage event is a
  single event settled at time zero. No time-uniformity is required and
  ``gamma`` is paid once, not per cycle. This is strictly the easier case.
* **Rolling (accruing reference).** The reference slides, so the interval
  must cover uniformly over the horizon — a confidence *sequence*, and
  ``gamma`` becomes a lifetime budget for the channel. Note the interval is
  for a *constant* ``p``: under ``H0`` that is exactly what the null
  asserts, and under the alternative the sequence chases a moving target,
  which costs power but never validity.

What is deliberately *not* here
-------------------------------
The conditional (Fisher) construction is valid only when both margins are
unobserved at bet time. It appears below solely as a test fixture with a
known-exact expectation, together with its frozen-reference counterpart
which is *invalid by design*, so the harness can prove it detects failure
before we trust it on anything real.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta, hypergeom

from dedrift.evalues.base import LOG_FLOOR, EValueOutcome, PriorState

#: Default symmetric tilt grid, spanning "materiality-sized" to "large".
#: Symmetric by construction: rates drift in both directions, and a
#: one-sided bet is blind to half the failures we care about.
DEFAULT_TILT_GRID: tuple[float, ...] = (1.5, 2.0, 3.0)


def symmetric_grid(tilts: tuple[float, ...] = DEFAULT_TILT_GRID) -> tuple[float, ...]:
    """Return ``{psi, 1/psi}`` for each tilt, sorted, duplicates dropped."""
    out = set()
    for psi in tilts:
        if psi <= 0:
            msg = f"tilt must be positive, got {psi}"
            raise ValueError(msg)
        out.add(float(psi))
        out.add(float(1.0 / psi))
    return tuple(sorted(out))


def tilt_from_materiality(p_ref: float, delta_pp: float) -> float:
    """Odds-ratio tilt corresponding to a ``delta_pp`` shift at ``p_ref``.

    Deriving the bet's support from the materiality band aims the e-value
    at effects the operator has declared they care about, which is a step
    toward testing the hypothesis that actually matters rather than testing
    nullity and filtering afterwards. It also makes predictability
    structural: the grid comes from configuration, never from data.

    Args:
        p_ref: Reference rate (from the golden baseline; predictable).
        delta_pp: Materiality band in percentage points.

    Returns:
        The tilt ``psi > 1``; clipped away from degenerate odds.
    """
    p0 = float(np.clip(p_ref, 1e-6, 1 - 1e-6))
    p1 = float(np.clip(p0 + delta_pp / 100.0, 1e-6, 1 - 1e-6))
    odds0 = p0 / (1 - p0)
    odds1 = p1 / (1 - p1)
    return max(odds1 / odds0, 1.0 + 1e-9)


def log_tilt_evalue(successes: int, trials: int, p: float, psi: float) -> float:
    """``log E`` for a single tilt at a *known* ``p``. Exact: ``E_p[E] = 1``."""
    if trials <= 0:
        return 0.0
    return float(successes * np.log(psi) - trials * np.log1p(-p + psi * p))


def _log_mixture_evalue(successes: int, trials: int, p: float, grid: tuple[float, ...]) -> float:
    """``log`` of the equal-weight mixture over ``grid`` at known ``p``.

    The arithmetic mean of e-values for the same null is an e-value by
    linearity — no independence needed — and a mixture degrades far more
    gracefully than a point bet when the true effect size is unknown,
    which it always is.
    """
    logs = np.array([log_tilt_evalue(successes, trials, p, psi) for psi in grid])
    m = float(np.max(logs))
    return float(m + np.log(np.mean(np.exp(logs - m))))


def clopper_pearson(successes: int, trials: int, gamma: float) -> tuple[float, float]:
    """Exact ``(1 - gamma)`` interval for a binomial rate.

    Exact rather than Wilson here because this interval carries a coverage
    guarantee the whole construction leans on; conservatism is the correct
    direction to err.
    """
    if trials <= 0:
        return 0.0, 1.0
    lo = 0.0 if successes == 0 else float(beta.ppf(gamma / 2, successes, trials - successes + 1))
    hi = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - gamma / 2, successes + 1, trials - successes))
    )
    return lo, hi


def worst_case_log_evalue(
    successes: int,
    trials: int,
    interval: tuple[float, float],
    grid: tuple[float, ...],
    n_grid: int = 129,
) -> float:
    """``inf`` over ``p`` in ``interval`` of the mixture's ``log E``.

    The infimum is taken of the *mixture* rather than mixing the
    per-tilt infima: both are valid, but the former is less conservative.
    """
    lo, hi = interval
    lo = float(np.clip(lo, 1e-9, 1 - 1e-9))
    hi = float(np.clip(hi, 1e-9, 1 - 1e-9))
    if hi < lo:
        lo, hi = hi, lo
    ps = np.linspace(lo, hi, n_grid)
    return float(min(_log_mixture_evalue(successes, trials, float(p), grid) for p in ps))


def rate_evalue(
    successes: int,
    trials: int,
    prior: PriorState,
    *,
    gamma: float = 0.01,
    grid: tuple[float, ...] | None = None,
    frozen_reference: bool = True,
) -> EValueOutcome:
    """Worst-case-over-interval e-value for the rate channel.

    Args:
        successes: Current-cycle successes.
        trials: Current-cycle trials. Zero means the family was suppressed
            or empty, which yields ``E_t = 1`` (no bet) — the correct
            contribution, preserving the supermartingale exactly.
        prior: Completed-cycle summaries. The *only* input to the bet.
        gamma: Coverage budget for the nuisance interval. Paid once for a
            frozen reference; a lifetime budget for an accruing one.
        grid: Symmetric tilt grid; defaults to
            :data:`DEFAULT_TILT_GRID` symmetrised.
        frozen_reference: True for the golden channel (fixed interval),
            False for rolling (the interval must be read as one member of a
            confidence sequence; the caller supplies a ``gamma`` already
            adjusted for time-uniformity).

    Returns:
        The outcome; ``log_e = 0`` whenever no admissible bet exists.
    """
    tilts = grid if grid is not None else symmetric_grid()
    if trials <= 0:
        return EValueOutcome(0.0, False, "no current-cycle trials", tilts)
    if prior.reference_trials <= 0:
        return EValueOutcome(0.0, False, "no reference data", tilts)

    interval = clopper_pearson(prior.reference_successes, prior.reference_trials, gamma)
    log_e = worst_case_log_evalue(successes, trials, interval, tilts)
    log_e = max(log_e, LOG_FLOOR)
    kind = "fixed CI" if frozen_reference else "confidence sequence member"
    return EValueOutcome(
        log_e=log_e,
        placed=True,
        detail=(
            f"worst case over p in [{interval[0]:.4f}, {interval[1]:.4f}] ({kind}, gamma={gamma})"
        ),
        bet=tilts,
    )


# --- test fixtures: exactness, and a deliberate failure ----------------------


def hypergeometric_tilt(
    successes_cur: int, trials_cur: int, successes_ref: int, trials_ref: int, psi: float
) -> float:
    """Conditional (Fisher) tilt ``E`` — exact only under random margins.

    Valid **only** if both margins are unobserved when the bet is placed.

    Averaged over both margins random this has expectation exactly 1, which
    makes it the ideal fixture for validating the martingale harness: a test
    that cannot confirm ``1.000000`` here has no business judging anything
    else. It is *not* used in the pipeline, because our reference data is
    always already observed by bet time.
    """
    tau = successes_ref + successes_cur
    n, m = trials_cur, trials_ref
    lo, hi = max(0, tau - m), min(n, tau)
    js = np.arange(lo, hi + 1)
    if len(js) <= 1:
        return 1.0
    pj = hypergeom.pmf(js, m + n, tau, n)
    norm = float(np.sum(pj * psi**js))
    if norm <= 0:
        return 1.0
    return float(psi**successes_cur / norm)


def frozen_reference_hypergeometric_INVALID(  # noqa: N802 - the name is the warning
    successes_cur: int, trials_cur: int, successes_ref: int, trials_ref: int, psi: float
) -> float:
    """The same formula applied with the reference **frozen** — NOT an e-value.

    Retained on purpose. Conditioning on the total is a bijection with the
    current count once the reference is known, so the hypergeometric is no
    longer the conditional law and the nuisance parameter returns:
    ``sup_p E[E] = 2132`` at ``m=105, r=5, n=21, psi=2.5``. The test suite
    feeds this to the martingale harness and asserts the harness *rejects*
    it, which is the only way to know a pass on a real construction means
    anything.
    """
    return hypergeometric_tilt(successes_cur, trials_cur, successes_ref, trials_ref, psi)
