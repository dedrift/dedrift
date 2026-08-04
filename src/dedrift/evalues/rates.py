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

``gamma`` is load-bearing, not a formality. Measured at the SHIPPED
per-process budget (``gamma_i = gamma_total / K = 0.02/24 = 8.3e-4``) over
the shipped three-tilt grid, by exact binomial enumeration: ``E_p[E_t]``
for ``p`` *inside* the interval lies in 0.46-0.94 across representative
streams (valid, and the shortfall from 1 is the price of worst-casing),
while over *all* ``p`` it reaches 5.4e6 as p -> 1. The guarantee holds on
the coverage event and nowhere else, which is why the interval is reported
alongside the wealth.

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
from scipy.special import logsumexp
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


def per_process_gamma(gamma_total: float, n_processes: int) -> float:
    """Split the coverage budget across processes — do not skip this.

    The temptation is to read ``alpha = alpha' + gamma`` as the whole story.
    It is the *per-process* story. e-BH's FDR guarantee requires every input
    to be a valid e-value, and a process whose nuisance interval misses the
    truth is not one, so coverage failures union-bound across the battery:

        P(ever a false alert) <= alpha' + sum_i gamma_i

    With 24 processes at ``gamma_i = 0.01`` that is ``0.04 + 0.24 = 0.28``,
    not 0.05 — the guarantee would be off by more than five times while
    every number in the report looked fine. Splitting ``gamma_total`` by the
    live process count keeps the arithmetic correct when the operator
    changes the suite, which is precisely when a hand-computed constant
    would go stale.

    The cost is real and belongs in the docs: narrower budgets mean wider
    intervals, more conservative bets, and less power.

    Args:
        gamma_total: Total coverage budget (``alpha - alpha'``).
        n_processes: Number of e-processes entering the e-BH pool.

    Returns:
        Per-process coverage level.
    """
    if n_processes <= 0:
        return gamma_total
    return gamma_total / n_processes


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


def worst_case_log_evalue_table(
    trials: int,
    interval: tuple[float, float],
    grid: tuple[float, ...],
    n_grid: int = 129,
) -> np.ndarray:
    """``log E`` for every possible success count, computed at once.

    Predictability has a useful computational consequence: the nuisance
    interval and the tilt grid are fixed before the cycle is observed, so
    the e-value is a *function of the success count alone*. Tabulating it
    over ``0..trials`` is therefore exact — not an approximation — and it
    turns long-horizon simulation studies from intractable into instant,
    which is what makes the anytime-valid null rate measurable at honest
    scale. Also the natural form for the pipeline: one table per
    (process, epoch) instead of a minimisation per cycle.

    Args:
        trials: Current-cycle trials ``n``.
        interval: Nuisance interval for ``p``.
        grid: Symmetric tilt grid.
        n_grid: Points in the ``p`` minimisation.

    Returns:
        Array of shape ``(trials + 1,)`` with ``log E`` per success count.
    """
    lo, hi = interval
    lo = float(np.clip(lo, 1e-9, 1 - 1e-9))
    hi = float(np.clip(hi, 1e-9, 1 - 1e-9))
    if hi < lo:
        lo, hi = hi, lo
    ps = np.linspace(lo, hi, n_grid)[:, None]  # (P, 1)
    psis = np.asarray(grid, dtype=float)[None, :]  # (1, T)
    s = np.arange(trials + 1)[:, None, None]  # (S, 1, 1)

    # log E(s, p, psi) = s*log(psi) - n*log(1 - p + psi*p)
    normaliser = trials * np.log1p(-ps + psis * ps)  # (P, T)
    logs = s * np.log(psis)[None, :, :] - normaliser[None, :, :]  # (S, P, T)
    mixture = logsumexp(logs, axis=2) - np.log(psis.size)  # (S, P)
    return np.asarray(np.min(mixture, axis=1))


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
        gamma: Coverage budget for the nuisance interval, paid once for the
            frozen reference.
        grid: Symmetric tilt grid; defaults to
            :data:`DEFAULT_TILT_GRID` symmetrised.
        frozen_reference: Must be True. Retained as an explicit argument so
            that the unsupported case fails loudly at the call site rather
            than silently producing a number with no guarantee behind it.

    Returns:
        The outcome; ``log_e = 0`` whenever no admissible bet exists.

    Raises:
        NotImplementedError: If ``frozen_reference`` is False. A reference
            recomputed each cycle needs a *time-uniform* interval -- a
            confidence sequence -- or its coverage event is a fresh event
            every cycle and the union bound over a horizon of ``T`` cycles
            is ``T * gamma`` rather than ``gamma``. An earlier version of
            this function accepted the flag and changed only a display
            string, which stated a guarantee it did not deliver. Refusing
            is the honest behaviour until the construction exists.
    """
    if not frozen_reference:
        msg = (
            "rate_evalue requires a frozen reference. A rolling reference needs a "
            "time-uniform interval (confidence sequence), which is not implemented; "
            "without one the lifetime coverage budget is T*gamma, not gamma. Use the "
            "fixed-sample path (`dedrift check`) for rolling comparisons."
        )
        raise NotImplementedError(msg)
    tilts = grid if grid is not None else symmetric_grid()
    if trials <= 0:
        return EValueOutcome(0.0, False, "no current-cycle trials", tilts)
    if prior.reference_trials <= 0:
        return EValueOutcome(0.0, False, "no reference data", tilts)

    interval = clopper_pearson(prior.reference_successes, prior.reference_trials, gamma)
    log_e = worst_case_log_evalue(successes, trials, interval, tilts)
    log_e = max(log_e, LOG_FLOOR)
    return EValueOutcome(
        log_e=log_e,
        placed=True,
        detail=(
            f"worst case over p in [{interval[0]:.4f}, {interval[1]:.4f}] "
            f"(frozen reference, fixed CI, gamma={gamma})"
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
    ``E_p[E] = 2132`` at ``p=0.5`` for ``m=105, r=5, n=21, psi=2.5``, rising
    to ~1.1e6 as ``p -> 1``. The test suite
    feeds this to the martingale harness and asserts the harness *rejects*
    it, which is the only way to know a pass on a real construction means
    anything.
    """
    return hypergeometric_tilt(successes_cur, trials_cur, successes_ref, trials_ref, psi)
