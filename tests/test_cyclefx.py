"""Cycle-effect correction (detectors/cyclefx.py) unit tests.

Statistical behavior here is asserted as measured bands on seeded draws,
per the suite's ethos: no unmeasured claims, no lucky-seed point estimates.
"""

from __future__ import annotations

import numpy as np

from dedrift.detectors.cyclefx import (
    cycle_level_pvalue,
    estimate_icc,
    per_cycle_statistic,
    rate_z_pvalue_clustered,
    standardize_within_cycle,
)


def _clustered_draw(rng, cycle_means, n_per_cycle, within_sd):
    values = []
    cycles = []
    for c, mu in enumerate(cycle_means):
        values.append(rng.normal(mu, within_sd, n_per_cycle))
        cycles += [f"c{c}"] * n_per_cycle
    return np.concatenate(values), np.array(cycles)


class TestEstimateIcc:
    def test_exchangeable_estimates_near_zero_signed(self) -> None:
        rng = np.random.default_rng(11)
        rhos = []
        for _ in range(200):
            values, cyc = _clustered_draw(rng, [50.0] * 5, 35, 10.0)
            rhos.append(estimate_icc(values, cyc, threshold=0.02).rho)
        rhos = np.array(rhos)
        # Signed estimator: symmetric about ~0 under exchangeability.
        assert abs(float(np.mean(rhos))) < 0.03
        assert (rhos < 0).any() and (rhos > 0).any()

    def test_cycle_offset_estimates_positive(self) -> None:
        rng = np.random.default_rng(13)
        engaged = 0
        rhos = []
        for _ in range(20):
            offsets = rng.normal(0, 8.0, 5)  # strong per-cycle offsets
            values, cyc = _clustered_draw(rng, 50.0 + offsets, 35, 10.0)
            ice = estimate_icc(values, cyc, threshold=0.02)
            engaged += ice.engaged
            rhos.append(ice.rho)
        # true rho ~ 0.39; allow single unlucky offset draws to slip
        assert engaged >= 17
        assert float(np.median(rhos)) > 0.2

    def test_too_few_cycles_never_engages(self) -> None:
        rng = np.random.default_rng(17)
        values, cyc = _clustered_draw(rng, [50.0, 70.0], 35, 10.0)
        ice = estimate_icc(values, cyc, threshold=0.02)
        assert not ice.engaged
        assert ice.n_cycles == 2


class TestStandardizeWithinCycle:
    def test_offsets_cancel_exactly(self) -> None:
        rng = np.random.default_rng(19)
        a, ca = _clustered_draw(rng, [10.0, 500.0, 50.0], 20, 5.0)
        z = standardize_within_cycle(a, ca)
        for c in ("c0", "c1", "c2"):
            w = z[ca == c]
            assert abs(float(np.mean(w))) < 1e-9
            assert abs(float(np.std(w, ddof=1)) - 1.0) < 1e-9

    def test_constant_cycle_yields_zeros_not_nan(self) -> None:
        values = np.array([7.0, 7.0, 7.0, 3.0, 3.2])
        cyc = np.array(["a", "a", "a", "b", "b"])
        z = standardize_within_cycle(values, cyc)
        assert np.isfinite(z).all()


class TestCycleLevelPvalue:
    def test_center_hit_is_nonsignificant(self) -> None:
        ref = np.array([10.0, 11.0, 9.5, 10.4, 9.8])
        assert cycle_level_pvalue(ref, 10.1) > 0.5

    def test_far_outlier_is_significant(self) -> None:
        ref = np.array([10.0, 11.0, 9.5, 10.4, 9.8])
        assert cycle_level_pvalue(ref, 30.0) < 0.01

    def test_constant_reference_is_uncomputable_not_p0(self) -> None:
        # A constant reference gives no spread estimate; NaN (keep the
        # record-level p) is the honest answer, never p=0.
        ref = np.full(5, 10.0)
        assert np.isnan(cycle_level_pvalue(ref, 10.0))
        assert np.isnan(cycle_level_pvalue(ref, 11.0))

    def test_too_few_cycles_is_nan(self) -> None:
        assert np.isnan(cycle_level_pvalue(np.array([1.0, 2.0]), 1.5))


class TestRateZClustered:
    def test_no_difference_stays_quiet(self) -> None:
        p = rate_z_pvalue_clustered(30, 100, 9, 30, 0.2, 33.0, 30.0)
        assert p > 0.5

    def test_correction_inflates_p(self) -> None:
        plain = rate_z_pvalue_clustered(30, 100, 20, 40, 0.0, 33.0, 40.0)
        corrected = rate_z_pvalue_clustered(30, 100, 20, 40, 0.3, 33.0, 40.0)
        assert corrected >= plain


class TestPerCycleStatistic:
    def test_mean_kind(self) -> None:
        values = np.array([1.0, 3.0, 10.0, 20.0])
        cyc = np.array(["a", "a", "b", "b"])
        assert per_cycle_statistic(values, cyc, "mean").tolist() == [2.0, 15.0]
