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
before the cycle and use a certified lower bound on the worst member of the
predeclared tilt mixture:

    E_t <= inf_{p in I} mean_psi E(S_t; n, p, psi)

On the coverage event ``{p_true in I}``, this is no larger than the
known-``p_true`` mixture, whose conditional expectation is 1. The product is
therefore a supermartingale *there*; Ville bounds the trajectory at
``alpha'`` and a union bound pays ``gamma`` for coverage. Total budget
``alpha = alpha' + gamma``.

``gamma`` is load-bearing, not a formality. The guarantee holds on the
coverage event and nowhere else, which is why the interval is reported
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

from dedrift.evalues.base import EValueOutcome, PriorState

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
    It is the *per-process* story. A process whose nuisance interval misses
    the truth is not valid, so coverage failures union-bound across the
    battery:

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
        n_processes: Number of e-processes in the frozen pool.

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
    """Certified lower bound on the mixture infimum over ``interval``.

    The positive mixture is convex in ``p``. Endpoint derivatives either
    identify an endpoint minimum, or bracket the unique interior minimum.
    In the latter case, the intersection of tangents at the bracket endpoints
    is a global lower bound on the convex function. This retains the power of
    minimizing the mixture without treating a sampled-grid minimum as exact.

    ``n_grid`` is retained for source compatibility with pre-0.5.1 callers;
    no numerical grid is used. The old grid minimum was an approximation
    from above and could violate e-validity between grid points.
    """
    del n_grid
    values = _certified_worst_case_logs(
        np.asarray([successes], dtype=float), trials, interval, grid
    )
    return float(values[0])


def worst_case_log_evalue_table(
    trials: int,
    interval: tuple[float, float],
    grid: tuple[float, ...],
    n_grid: int = 129,
) -> np.ndarray:
    """Certified ``log E`` for every possible success count at once.

    Predictability has a useful computational consequence: the nuisance
    interval and the tilt grid are fixed before the cycle is observed, so
    the e-value is a *function of the success count alone*. Tabulating it
    over ``0..trials`` is therefore exact and turns long-horizon simulation
    studies from intractable into instant. It is also the natural form for
    the pipeline: one table per ``(process, epoch)``.

    Args:
        trials: Current-cycle trials ``n``.
        interval: Nuisance interval for ``p``.
        grid: Symmetric tilt grid.
        n_grid: Retained for source compatibility; ignored.

    Returns:
        Array of shape ``(trials + 1,)`` with ``log E`` per success count.
    """
    del n_grid
    return _certified_worst_case_logs(np.arange(trials + 1, dtype=float), trials, interval, grid)


def _mixture_logs_and_slopes(
    successes: np.ndarray,
    trials: int,
    p: np.ndarray,
    psis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log-mixture and ``d/dp log(mixture)`` row-wise."""
    denominators = 1.0 + p[:, None] * (psis[None, :] - 1.0)
    component_logs = successes[:, None] * np.log(psis)[None, :] - trials * np.log(denominators)
    log_sums = logsumexp(component_logs, axis=1)
    weights = np.exp(component_logs - log_sums[:, None])
    component_slopes = -trials * (psis[None, :] - 1.0) / denominators
    slopes = np.sum(weights * component_slopes, axis=1)
    return np.asarray(log_sums - np.log(psis.size)), np.asarray(slopes)


def _componentwise_fallback_logs(
    successes: np.ndarray,
    trials: int,
    lo: float,
    hi: float,
    psis: np.ndarray,
) -> np.ndarray:
    """Simple certified fallback: mix each tilt's analytic infimum."""
    worst_p = np.where(psis > 1.0, hi, lo)
    normaliser = trials * np.log1p(worst_p * (psis - 1.0))
    logs = successes[:, None] * np.log(psis)[None, :] - normaliser[None, :]
    return np.asarray(logsumexp(logs, axis=1) - np.log(psis.size))


def _certified_worst_case_logs(
    successes: np.ndarray,
    trials: int,
    interval: tuple[float, float],
    grid: tuple[float, ...],
) -> np.ndarray:
    """Lower-bound the convex mixture minimum, with conservative rounding."""
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if np.any(successes < 0) or np.any(successes > trials):
        raise ValueError("successes must lie between zero and trials")
    psis = np.asarray(grid, dtype=float)
    if psis.size == 0 or not np.isfinite(psis).all() or np.any(psis <= 0):
        raise ValueError("grid must contain at least one finite positive tilt")
    lo, hi = (float(np.clip(value, 0.0, 1.0)) for value in interval)
    if hi < lo:
        lo, hi = hi, lo
    n_rows = successes.size
    lower_p = np.full(n_rows, lo)
    upper_p = np.full(n_rows, hi)
    log_lo, slope_lo = _mixture_logs_and_slopes(successes, trials, lower_p, psis)
    log_hi, slope_hi = _mixture_logs_and_slopes(successes, trials, upper_p, psis)
    result = np.empty(n_rows, dtype=float)

    at_lo = slope_lo >= 0.0
    at_hi = ~at_lo & (slope_hi <= 0.0)
    interior = ~(at_lo | at_hi)
    result[at_lo] = log_lo[at_lo]
    result[at_hi] = log_hi[at_hi]

    if np.any(interior):
        interior_successes = successes[interior]
        a = lower_p[interior]
        b = upper_p[interior]
        # Convexity makes the derivative monotone, so bisection preserves a
        # bracket around the mixture minimum. Eighty steps reach float64's
        # resolution even for the full [0, 1] interval.
        for _ in range(80):
            mid = a + (b - a) / 2.0
            _log_mid, slope_mid = _mixture_logs_and_slopes(interior_successes, trials, mid, psis)
            left = slope_mid <= 0.0
            a = np.where(left, mid, a)
            b = np.where(left, b, mid)

        log_a, slope_a = _mixture_logs_and_slopes(interior_successes, trials, a, psis)
        log_b, slope_b = _mixture_logs_and_slopes(interior_successes, trials, b, psis)
        scale = np.maximum(log_a, log_b)
        f_a = np.exp(log_a - scale)
        f_b = np.exp(log_b - scale)
        d_a = f_a * slope_a
        d_b = f_b * slope_b
        denominator = d_a - d_b
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = (f_b - f_a + d_a * a - d_b * b) / denominator
        crossing = np.clip(np.where(np.isfinite(crossing), crossing, (a + b) / 2.0), a, b)
        tangent_a = f_a + d_a * (crossing - a)
        tangent_b = f_b + d_b * (crossing - b)
        lower_scaled = np.maximum(tangent_a, tangent_b)
        certified = scale + np.log(lower_scaled)

        # This branch should be unreachable for finite positive mixtures, but
        # the component-wise infimum is a simpler certified fallback if
        # floating arithmetic ever loses the positive tangent intersection.
        fallback = _componentwise_fallback_logs(interior_successes, trials, lo, hi, psis)
        result[interior] = np.where(
            np.isfinite(certified) & (lower_scaled > 0.0), certified, fallback
        )

    # Every operation is in log space, so no upward floor is needed. The
    # tangent construction is a real-arithmetic lower bound; this guard also
    # keeps accumulated float64 roundoff on the conservative side.
    safety = 4096.0 * np.finfo(np.float64).eps * (trials + 1) * (psis.size + 1)
    return np.nextafter(result - safety, -np.inf)


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
        trials: Current-cycle trials. Zero means no rate observation was
            available and yields ``E_t = 1`` (no bet). Validity requires
            availability to be determined independently of its value.
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
    return EValueOutcome(
        log_e=log_e,
        placed=True,
        detail=(
            f"certified mixture worst case over p in [{interval[0]:.4f}, {interval[1]:.4f}] "
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
