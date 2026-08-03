"""Anytime-valid calibration: the claim that replaces "1.4% per check".

The headline is a statement about *trajectories*, so it has to be measured
on trajectories: simulate stable agents for many cycles and count the runs
in which the battery ever alerts. The contrast with the fixed-sample path on
identical histories is measured in the same test, because that contrast is
the reason the new path exists.

Two properties of this file are load-bearing, and both were wrong once.

**The budget is the shipped budget.** Every level here is read from
:class:`~dedrift.config.AnytimeConfig` and split with
:func:`~dedrift.evalues.rates.per_process_gamma`, so the test measures the
configuration users actually get. An earlier version hard-coded
``gamma = 0.01`` and applied it *per process* at ``K = 24``, delivering
``0.04 + 24 * 0.01 = 0.28`` while asserting a bound below ``0.05``. The
gate certified a budget the package had already fixed, and would have kept
passing if the fix were deleted. :class:`TestBudgetArithmetic` now pins the
arithmetic itself.

**The streams are dependent.** Under the null the whole point at issue is
whether e-BH survives the dependence our battery actually has: every
signature of a family is computed from the *same* records of the same
cycle. Drawing each stream from an independent binomial measures the case
in which stopped e-BH is already proven, which is no measurement at all.
Success counts here are generated through a Gaussian copula on shared
record-level latents.

Scale is stated, as always. CI runs 120 runs x 400 cycles over 24 rate
processes; the full study at 500 x 2000 lives in
``launch/anytime_null_study.py`` and its numbers are what the docs quote.
Wilson bounds are reported so the CI-scale result is honest about its own
precision rather than implying the full-scale one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from dedrift.config import AnytimeConfig
from dedrift.evalues import clopper_pearson, symmetric_grid
from dedrift.evalues.rates import per_process_gamma, worst_case_log_evalue_table

#: Read from the shipped config so the gate cannot certify a budget users
#: do not get.
_CFG = AnytimeConfig()
ALPHA = _CFG.alpha
GAMMA_TOTAL = _CFG.gamma_total
ALPHA_PRIME = _CFG.alpha_prime

#: Default suite scale: 18 canaries x 7 reps over 6 families => 21 records
#: per family per cycle; golden = 5 frozen cycles => 105 reference trials.
N_CUR, N_REF = 21, 105

#: 24 rate e-processes: 6 families x 4 rate signatures, golden baseline
#: only (rolling has no time-uniform interval, so it is not in the anytime
#: pool). ``exact_match`` is excluded exactly as the live pool filter
#: excludes it when a suite declares no expected answers. Rates span rare
#: to near-certain because the regimes behave differently.
_FAMILY_RATES = {
    "refusal": [0.05, 0.05, 0.10, 0.02, 0.15, 0.08],
    "format_valid": [0.98, 0.99, 0.97, 0.99, 0.95, 0.98],
    "args_schema_ok_all": [0.97, 0.96, 0.98, 0.94, 0.97, 0.99],
    "had_error": [0.01, 0.02, 0.01, 0.03, 0.01, 0.02],
}
N_FAMILIES = 6
BASE_RATES = np.array([r for rates in _FAMILY_RATES.values() for r in rates])
#: Stream k belongs to this family. Streams sharing a family are computed
#: from the same records, which is the dependence under test.
FAMILY_OF = np.array(list(range(N_FAMILIES)) * len(_FAMILY_RATES))
K = len(BASE_RATES)
GAMMA_I = per_process_gamma(GAMMA_TOTAL, K)

#: Within-family correlation between signatures computed on shared records.
#: A refused response is also more likely to be format-invalid; 0.5 is a
#: deliberately substantial value, since the purpose is to stress e-BH.
RHO_WITHIN = 0.5


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Upper bound of the Wilson score interval."""
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    mrg = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c + mrg) / d)


def _draw_successes(
    rng: np.random.Generator,
    cycles: int,
    rates: np.ndarray,
    *,
    sigma_cycle: float = 0.0,
    rho_within: float = RHO_WITHIN,
) -> np.ndarray:
    """Per-cycle success counts with the battery's real dependence.

    Records are shared within a family, so signatures of one family are
    correlated; a Gaussian copula on a shared record-level latent produces
    that without changing any marginal.

    Args:
        rng: Seeded generator.
        cycles: Number of cycles to draw.
        rates: Marginal success probabilities. Either ``(K,)`` for a
            constant regime or ``(cycles, K)`` to script a change part-way
            through a history.
        sigma_cycle: Standard deviation of a cycle-level logit offset shared
            by *every* stream. Zero keeps the distributional null exactly
            true (the case for calibration); positive values model
            provider-side state that moves all channels together while the
            configured stack is unchanged.
        rho_within: Correlation of the record-level latent across signatures
            of the same family.

    Returns:
        Integer array of shape ``(cycles, K)``.
    """
    rate_mat = np.broadcast_to(np.asarray(rates, dtype=float), (cycles, K))
    thresholds = norm.ppf(rate_mat)  # (cycles, K)
    out = np.empty((cycles, K), dtype=int)
    u = rng.normal(0.0, sigma_cycle, size=(cycles, 1)) if sigma_cycle > 0 else np.zeros((cycles, 1))
    shared = {f: rng.normal(size=(cycles, N_CUR)) for f in range(N_FAMILIES)}
    w_mix = np.sqrt(max(1.0 - rho_within**2, 0.0))
    for k in range(K):
        eps = shared[int(FAMILY_OF[k])]
        idio = rng.normal(size=(cycles, N_CUR))
        w = rho_within * eps + w_mix * idio
        thr = thresholds[:, k : k + 1] + u
        out[:, k] = np.sum(w < thr, axis=1)
    return out


def _ebh_any_rejection(log_wealth: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH applied at every cycle, in log space.

    In logs because wealth overflows float64 on long horizons; the decision
    rule is scale-monotone so the translation is exact.
    """
    _, k_proc = log_wealth.shape
    ks = np.arange(1, k_proc + 1)
    need = np.log(k_proc) - np.log(alpha) - np.log(ks)
    srt = -np.sort(-log_wealth, axis=1)
    return np.any(srt >= need[None, :], axis=1)


def _tables(refs: np.ndarray, grid: tuple[float, ...], gamma_i: float) -> np.ndarray:
    """One e-value lookup table per process, at the *per-process* gamma."""
    return np.stack(
        [
            worst_case_log_evalue_table(N_CUR, clopper_pearson(int(r), N_REF, gamma_i), grid)
            for r in refs
        ]
    )


def _run_null(
    n_runs: int, cycles: int, seed: int, *, sigma_cycle: float = 0.0
) -> dict[str, object]:
    """Stable agents: measure ever-alert for both inference paths."""
    grid = symmetric_grid(_CFG.tilts)
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, K + 1)
    horizons = (100, cycles)
    anytime = {h: 0 for h in horizons}
    fixed = {h: 0 for h in horizons}

    for _ in range(n_runs):
        refs = rng.binomial(N_REF, BASE_RATES)
        tbl = _tables(refs, grid, GAMMA_I)
        s = _draw_successes(rng, cycles, BASE_RATES, sigma_cycle=sigma_cycle)

        wealth = np.cumsum(np.stack([tbl[k][s[:, k]] for k in range(K)], axis=1), axis=0)
        rej_e = _ebh_any_rejection(wealth, ALPHA_PRIME)

        p_ref = refs / N_REF
        p_cur = s / N_CUR
        pooled = (refs[None, :] + s) / (N_REF + N_CUR)
        se = np.sqrt(np.clip(pooled * (1 - pooled), 1e-12, None) * (1 / N_REF + 1 / N_CUR))
        pv = 2 * norm.sf(np.abs(p_cur - p_ref[None, :]) / se)
        rej_p = np.any(np.sort(pv, axis=1) <= ALPHA * ranks[None, :] / K, axis=1)

        for h in horizons:
            if rej_e[:h].any():
                anytime[h] += 1
            if rej_p[:h].any():
                fixed[h] += 1
    return {"anytime": anytime, "fixed": fixed, "n_runs": n_runs, "cycles": cycles}


class TestBudgetArithmetic:
    """The split itself, pinned — this is what silently regressed once."""

    def test_gamma_is_split_across_the_live_pool(self) -> None:
        """``gamma_i = gamma_total / K``, not ``gamma_total``."""
        assert pytest.approx(GAMMA_TOTAL / K) == GAMMA_I
        assert pytest.approx(ALPHA) == ALPHA_PRIME + K * GAMMA_I

    def test_unsplit_gamma_would_blow_the_budget(self) -> None:
        """The error this file used to contain, stated as arithmetic.

        Spending ``gamma_total`` per process delivers a battery-wide budget
        of ``alpha' + K * gamma_total``. At the shipped defaults that is
        five times the level claimed, which is precisely why the split is
        computed from the live pool size rather than typed in.
        """
        unsplit = ALPHA_PRIME + K * GAMMA_TOTAL
        assert unsplit > 5 * ALPHA, f"expected a gross overspend, got {unsplit}"

    def test_levels_come_from_the_shipped_config(self) -> None:
        """A hard-coded level here would let the gate certify a fiction."""
        cfg = AnytimeConfig()
        assert (cfg.alpha, cfg.gamma_total, cfg.alpha_prime) == (ALPHA, GAMMA_TOTAL, ALPHA_PRIME)


@pytest.mark.calibration
class TestAnytimeValidNullRate:
    """The headline claim, measured on trajectories with dependent streams."""

    def test_ever_alert_rate_bounded_over_long_stable_histories(self) -> None:
        """P(ever falsely alerting) <= alpha over the whole horizon.

        The assertion is on the Wilson upper bound, not the point estimate,
        so a lucky zero cannot pass a claim the sample size does not
        support. Streams are dependent within family, which is the case the
        stopped-e-BH condition is assumed for and the reason this is
        reported as measured rather than proven.
        """
        res = _run_null(n_runs=120, cycles=400, seed=20260803)
        n = int(res["n_runs"])  # type: ignore[arg-type]
        hits = res["anytime"][400]  # type: ignore[index]
        bound = wilson_upper(int(hits), n)
        assert bound < ALPHA, (
            f"anytime-valid null rate {hits}/{n} over 400 cycles "
            f"(Wilson upper {bound:.4f} >= alpha={ALPHA})"
        )

    def test_rate_does_not_grow_with_horizon(self) -> None:
        """Horizon insensitivity — the qualitative signature of the fix.

        A fixed-sample guarantee decays with use; an anytime-valid one does
        not. Measured as: the rate at the full horizon is not materially
        above the rate at a quarter of it.
        """
        res = _run_null(n_runs=120, cycles=400, seed=7)
        short = res["anytime"][100] / 120  # type: ignore[index]
        long = res["anytime"][400] / 120  # type: ignore[index]
        assert long <= short + 0.02, f"rate grew with horizon: {short:.3f} -> {long:.3f}"

    def test_fixed_sample_path_degrades_on_the_same_histories(self) -> None:
        """The defect being fixed, measured rather than asserted.

        Same simulated agents, same channels, same alpha: the per-check path
        accumulates false alerts until it is certain to have raised one.
        This is the motivating evidence for the whole construction, so it is
        a test rather than a claim in prose.

        Note what this comparison is and is not: both arms adjudicate the
        same 24 rate streams with no materiality gate, isolating the
        inference layer. It is not the shipped ``dedrift check``, which
        additionally applies a Yates correction and materiality gating and
        whose measured per-check rate is correspondingly lower.
        """
        res = _run_null(n_runs=60, cycles=400, seed=11)
        fixed_long = res["fixed"][400] / 60  # type: ignore[index]
        fixed_short = res["fixed"][100] / 60  # type: ignore[index]
        anytime_long = res["anytime"][400] / 60  # type: ignore[index]
        assert fixed_short > 0.5, f"expected the p-path to false-alarm early, got {fixed_short}"
        assert fixed_long >= fixed_short, "cumulative rate should be non-decreasing"
        assert fixed_long > anytime_long + 0.5, (
            f"expected a large gap: fixed={fixed_long:.2f} vs anytime={anytime_long:.2f}"
        )


@pytest.mark.calibration
class TestCycleLevelRandomEffect:
    """What happens when the stack is unchanged but the *law* is not.

    The design's exchangeability argument needs the per-record distribution
    to be constant across cycles, which is strictly stronger than "nothing
    in the configured stack changed". Hosted models drift within a version:
    load, routing, cache state. This measures the consequence rather than
    assuming it away, and it is reported as a robustness result, not as
    part of the headline guarantee.
    """

    def test_small_cycle_effect_is_tolerated(self) -> None:
        """A modest shared logit wobble should not break the trajectory bound."""
        res = _run_null(n_runs=60, cycles=200, seed=4242, sigma_cycle=0.05)
        hits = int(res["anytime"][200])  # type: ignore[index]
        bound = wilson_upper(hits, 60)
        assert bound < 0.20, f"anytime path degraded at sigma=0.05: {hits}/60 (Wilson {bound:.3f})"

    def test_large_cycle_effect_is_recorded_not_hidden(self) -> None:
        """A large cycle effect *should* alert — and the tool cannot tell why.

        This is a genuine limitation, not a bug: a shared per-cycle shift is
        a real behavioral change, indistinguishable from drift by any
        two-sample test on these signatures. The test pins the direction so
        the documentation cannot quietly stop being true.
        """
        res = _run_null(n_runs=40, cycles=200, seed=99, sigma_cycle=0.40)
        small = _run_null(n_runs=40, cycles=200, seed=99, sigma_cycle=0.0)
        assert res["anytime"][200] >= small["anytime"][200], (  # type: ignore[index]
            "a large cycle-level effect should not alert less than none"
        )


@pytest.mark.power
class TestAnytimeDetection:
    """Anytime-validity is not free; quantify the cost rather than hide it."""

    def test_material_shift_is_detected_and_delay_is_reported(self) -> None:
        """A +10pp shift must be detected, and the delay is the honest cost.

        Detection is not immediate: the process must accumulate wealth past
        log(1/alpha'). The floor is set from the measurement, not from
        ambition: the full study (500 runs x 400 cycles) puts all-channel
        +10pp detection at 0.74, and at this test's 20 runs the 95% Wilson
        lower bound for a true 0.74 is about 0.52. The floor is therefore
        0.50 — it catches a collapse in power, not a sampling wobble.

        The previous floor was 0.80, which passed only because the test then
        spent ``gamma_total`` per process instead of ``gamma_total / K``.
        Correcting the budget widens every nuisance interval and costs real
        power; that cost is published rather than tuned away.
        """
        grid = symmetric_grid(_CFG.tilts)
        rng = np.random.default_rng(3)
        detected, delays = 0, []
        n_runs, cycles = 20, 200
        shifted = np.clip(BASE_RATES + 0.10, 1e-4, 1 - 1e-4)
        for _ in range(n_runs):
            refs = rng.binomial(N_REF, BASE_RATES)
            tbl = _tables(refs, grid, GAMMA_I)
            s = _draw_successes(rng, cycles, shifted)
            wealth = np.cumsum(np.stack([tbl[k][s[:, k]] for k in range(K)], axis=1), axis=0)
            rej = _ebh_any_rejection(wealth, ALPHA_PRIME)
            if rej.any():
                detected += 1
                delays.append(int(np.argmax(rej)) + 1)
        assert detected / n_runs >= 0.50, f"detected only {detected}/{n_runs} at +10pp"
        assert np.median(delays) < 120, f"median delay {np.median(delays)} cycles is a regression"

    def test_larger_coverage_budget_buys_power_here(self) -> None:
        """The alpha'/gamma split is a real parameter, and in this regime
        the trade is one-sided.

        Worst-casing over a wide interval makes every e-value conservative,
        so a small gamma costs power while the null rate has ample headroom.
        Measured: detection improves monotonically as ``gamma_total`` grows
        across 0.005 -> 0.02, with no null-rate cost visible at this scale.
        Both arms split the budget across the pool, so the comparison is
        between two valid configurations rather than between a valid one and
        an overspent one.
        """
        grid = symmetric_grid(_CFG.tilts)
        results = {}
        shifted = np.clip(BASE_RATES + 0.10, 1e-4, 1 - 1e-4)
        for gamma_total in (0.005, 0.02):
            rng = np.random.default_rng(101)
            gamma_i = per_process_gamma(gamma_total, K)
            det = 0
            n_runs, cycles = 15, 200
            for _ in range(n_runs):
                refs = rng.binomial(N_REF, BASE_RATES)
                tbl = _tables(refs, grid, gamma_i)
                s = _draw_successes(rng, cycles, shifted)
                wealth = np.cumsum(np.stack([tbl[k][s[:, k]] for k in range(K)], axis=1), axis=0)
                if _ebh_any_rejection(wealth, ALPHA - gamma_total).any():
                    det += 1
            results[gamma_total] = det / n_runs
        assert results[0.02] >= results[0.005], f"expected monotone power in gamma, got {results}"
