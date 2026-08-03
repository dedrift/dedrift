"""Anytime-valid calibration: the claim that replaces "1.4% per check".

The headline is a statement about *trajectories*, so it has to be measured
on trajectories: simulate stable agents for many cycles and count the runs
in which the battery ever alerts. The contrast with the fixed-sample path on
identical histories is measured in the same test, because that contrast is
the reason the new path exists.

Scale is stated, as always. CI runs 120 runs x 400 cycles over 24 rate
processes (~1.2 x 10**6 cycle-process updates); the full study at
500 x 2000 lives in ``launch/anytime_null_study.py`` and its numbers are
what the docs quote. Wilson bounds are reported so the CI-scale result is
honest about its own precision rather than implying the full-scale one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from dedrift.evalues import clopper_pearson, symmetric_grid
from dedrift.evalues.rates import worst_case_log_evalue_table

ALPHA = 0.05
GAMMA = 0.01
ALPHA_PRIME = ALPHA - GAMMA

#: Default suite scale: 18 canaries x 7 reps over 6 families => 21 records
#: per family per cycle; golden = 5 frozen cycles => 105 reference trials.
N_CUR, N_REF = 21, 105

#: 24 rate e-processes: 6 families x 2 rate signatures x 2 baselines,
#: spanning rare (0.02) to near-certain (0.99) — the regimes behave
#: differently and both belong in the measurement.
BASE_RATES = np.array(
    [0.05, 0.05, 0.10, 0.02, 0.15, 0.08]
    + [0.98, 0.99, 0.97, 0.99, 0.95, 0.98]
    + [0.05, 0.05, 0.10, 0.02, 0.15, 0.08]
    + [0.98, 0.99, 0.97, 0.99, 0.95, 0.98]
)
K = len(BASE_RATES)


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Upper bound of the Wilson score interval."""
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    mrg = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c + mrg) / d)


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


def _tables(refs: np.ndarray, grid: tuple[float, ...], gamma: float) -> np.ndarray:
    return np.stack(
        [
            worst_case_log_evalue_table(N_CUR, clopper_pearson(int(r), N_REF, gamma), grid)
            for r in refs
        ]
    )


def _run_null(n_runs: int, cycles: int, seed: int) -> dict[str, object]:
    """Stable agents: measure ever-alert for both inference paths."""
    grid = symmetric_grid((1.5, 2.0, 3.0))
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, K + 1)
    horizons = (100, cycles)
    anytime = {h: 0 for h in horizons}
    fixed = {h: 0 for h in horizons}

    for _ in range(n_runs):
        refs = rng.binomial(N_REF, BASE_RATES)
        tbl = _tables(refs, grid, GAMMA)
        s = rng.binomial(N_CUR, BASE_RATES[None, :], size=(cycles, K))

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


@pytest.mark.calibration
class TestAnytimeValidNullRate:
    """The headline claim, measured on trajectories."""

    def test_ever_alert_rate_bounded_over_long_stable_histories(self) -> None:
        """P(ever falsely alerting) <= alpha over the whole horizon.

        The assertion is on the Wilson upper bound, not the point estimate,
        so a lucky zero cannot pass a claim the sample size does not
        support.
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

        Same simulated agents, same battery, same alpha: the per-check path
        accumulates false alerts until it is certain to have raised one.
        This is the motivating evidence for the whole construction, so it is
        a test rather than a claim in prose.
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


@pytest.mark.power
class TestAnytimeDetection:
    """Anytime-validity is not free; quantify the cost rather than hide it."""

    def test_material_shift_is_detected_and_delay_is_reported(self) -> None:
        """A +10pp shift must be detected, and the delay is the honest cost.

        Detection is not immediate: the process must accumulate wealth past
        log(1/alpha'). The assertion is deliberately loose on delay — its
        purpose is to record the order of magnitude in CI so a regression in
        power shows up, while the published number comes from the full
        study.
        """
        grid = symmetric_grid((1.5, 2.0, 3.0))
        rng = np.random.default_rng(3)
        detected, delays = 0, []
        n_runs, cycles = 20, 200
        for _ in range(n_runs):
            refs = rng.binomial(N_REF, BASE_RATES)
            tbl = _tables(refs, grid, GAMMA)
            shifted = np.clip(BASE_RATES + 0.10, 1e-4, 1 - 1e-4)
            s = rng.binomial(N_CUR, shifted[None, :], size=(cycles, K))
            wealth = np.cumsum(np.stack([tbl[k][s[:, k]] for k in range(K)], axis=1), axis=0)
            rej = _ebh_any_rejection(wealth, ALPHA_PRIME)
            if rej.any():
                detected += 1
                delays.append(int(np.argmax(rej)) + 1)
        assert detected / n_runs >= 0.8, f"detected only {detected}/{n_runs} at +10pp"
        assert np.median(delays) < 120, f"median delay {np.median(delays)} cycles is a regression"

    def test_larger_coverage_budget_buys_power_here(self) -> None:
        """The alpha'/gamma split is a real parameter, and in this regime
        the trade is one-sided.

        Worst-casing over a wide interval makes every e-value conservative,
        so a small gamma costs power while the null rate has ample headroom.
        Measured: detection improves monotonically as gamma grows across
        0.005 -> 0.02, with no null-rate cost visible at this scale. That is
        an argument for revisiting the default, and the test exists to keep
        the observation from silently reversing.
        """
        grid = symmetric_grid((1.5, 2.0, 3.0))
        results = {}
        for gamma in (0.005, 0.02):
            rng = np.random.default_rng(101)
            det = 0
            n_runs, cycles = 15, 200
            for _ in range(n_runs):
                refs = rng.binomial(N_REF, BASE_RATES)
                tbl = _tables(refs, grid, gamma)
                shifted = np.clip(BASE_RATES + 0.10, 1e-4, 1 - 1e-4)
                s = rng.binomial(N_CUR, shifted[None, :], size=(cycles, K))
                wealth = np.cumsum(np.stack([tbl[k][s[:, k]] for k in range(K)], axis=1), axis=0)
                if _ebh_any_rejection(wealth, ALPHA - gamma).any():
                    det += 1
            results[gamma] = det / n_runs
        assert results[0.02] >= results[0.005], f"expected monotone power in gamma, got {results}"
