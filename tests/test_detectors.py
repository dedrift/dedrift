"""Unit tests for individual detectors: correctness on known inputs."""

from __future__ import annotations

import numpy as np
import pytest

from dedrift.detectors import (
    ad_test,
    benjamini_hochberg,
    ks_test,
    levene_test,
    p95_permutation_test,
    page_hinkley,
    psi,
    two_proportion_z_test,
    welch_t_test,
)

RNG = np.random.default_rng(7)


class TestScalarTests:
    def test_ks_detects_large_shift(self) -> None:
        ref = RNG.normal(0, 1, 200)
        cur = RNG.normal(2, 1, 200)
        out = ks_test(ref, cur)
        assert out.p_value < 1e-6
        # KS reports D as its effect (same scale the gate uses); for a 2-SD
        # mean shift of normals, population D = 2*Phi(1) - 1 ~ 0.68.
        assert out.effect_size == out.statistic
        assert out.effect_size == pytest.approx(0.68, abs=0.1)
        assert out.effect_raw == pytest.approx(2.0, abs=0.4)

    def test_ks_null_not_tiny(self) -> None:
        ref = RNG.normal(0, 1, 200)
        cur = RNG.normal(0, 1, 200)
        assert ks_test(ref, cur).p_value > 0.001

    def test_degenerate_gives_nan(self) -> None:
        ref = np.ones(10)
        cur = np.ones(10)
        assert np.isnan(ks_test(ref, cur).p_value)
        assert np.isnan(ad_test(ref, cur).p_value)
        assert np.isnan(welch_t_test(ref, cur).p_value)

    def test_levene_detects_variance_shift(self) -> None:
        """Effect is the ROBUST dispersion ratio, on the scale the test uses.

        Brown-Forsythe works on absolute deviations from the median, so the
        reported effect is the ratio of mean absolute deviations, not the
        variance ratio. For N(0,1) against N(0,3) that is the SD ratio 3,
        not its square 9. Gating on the variance ratio would have handed a
        robust test's decision back to the non-robust quantity it exists to
        avoid -- the same error as gating KS on Cohen's d.
        """
        ref = RNG.normal(0, 1, 300)
        cur = RNG.normal(0, 3, 300)
        out = levene_test(ref, cur)
        assert out.p_value < 1e-6
        assert out.effect_size == pytest.approx(3.0, rel=0.2)

    def test_p95_perm_seeded_reproducible(self) -> None:
        ref = RNG.normal(0, 1, 150)
        cur = RNG.normal(0.5, 1.5, 150)
        a = p95_permutation_test(ref, cur, seed=42)
        b = p95_permutation_test(ref, cur, seed=42)
        assert a == b

    def test_p95_perm_detects_tail_shift(self) -> None:
        ref = RNG.normal(0, 1, 300)
        cur = np.concatenate([RNG.normal(0, 1, 270), RNG.normal(6, 1, 30)])
        out = p95_permutation_test(ref, cur, seed=1)
        assert out.p_value < 0.05
        assert out.effect_raw > 1.0


class TestProportions:
    def test_detects_rate_shift(self) -> None:
        out = two_proportion_z_test(10, 500, 50, 500)
        assert out.p_value < 1e-6
        assert out.effect_raw == pytest.approx(0.08, abs=1e-9)

    def test_null_no_signal(self) -> None:
        out = two_proportion_z_test(25, 500, 25, 500)
        assert out.p_value == pytest.approx(1.0, abs=0.05)

    def test_zero_trials_nan(self) -> None:
        assert np.isnan(two_proportion_z_test(0, 0, 5, 10).p_value)


class TestBH:
    def test_controls_obvious_case(self) -> None:
        p = [0.001, 0.002, 0.9, 0.8, 0.7]
        rejected, adjusted = benjamini_hochberg(p, q=0.05)
        assert rejected == [True, True, False, False, False]
        assert adjusted[0] <= 0.05

    def test_nan_never_rejected_and_not_counted(self) -> None:
        p = [0.01, float("nan"), 0.5]
        rejected, adjusted = benjamini_hochberg(p, q=0.05)
        assert rejected[1] is False
        assert np.isnan(adjusted[1])
        # m=2, so p=0.01 adjusted = 0.02
        assert adjusted[0] == pytest.approx(0.02)

    def test_empty(self) -> None:
        assert benjamini_hochberg([], q=0.05) == ([], [])

    def test_monotone_adjusted(self) -> None:
        p = list(RNG.uniform(0, 1, 50))
        _, adjusted = benjamini_hochberg(p, q=0.1)
        order = np.argsort(p)
        adj_sorted = np.array(adjusted)[order]
        assert all(np.diff(adj_sorted) >= -1e-12)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, so null-rate tests bound rather than point-estimate."""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - m) / d), float((c + m) / d)


class TestPageHinkley:
    def test_no_alarm_on_stationary(self) -> None:
        values = RNG.normal(10, 1, 30)
        assert not page_hinkley(values).alarm

    def test_null_false_alarm_rate_matches_the_documented_band(self) -> None:
        """The causal estimator's honest null rate, bounded rather than
        point-estimated on a lucky seed.

        History, because it is the point of the test. The idealized bound is
        ~2*exp(-2*delta*lambda) ~= 0.5% per stream. An earlier version
        measured 0.7% -- with a scale estimated from the WHOLE stream,
        including cycles after the alarm, i.e. by reading the future of the
        point being judged. Standardizing causally costs what it should.

        Measured over 8000 draws: **8.5% at 30 points, 11.3% at 60**. An
        earlier version of this test asserted ``< 0.10`` on 300 draws at one
        seed, which reported 5.0%; across twelve seeds at 500 draws the same
        quantity ranges 5.4%-11.2%, so that gate was a coin flip and the
        quoted figure was 40% of the truth. A calibration test that passes
        on the seed it was written with is the failure mode this whole suite
        exists to prevent.

        This version uses 2000 draws and asserts the Wilson interval lies
        inside a documented band, so it is robust to reseeding and it fails
        if the estimator silently becomes non-causal again (which would send
        the rate back toward 1%).
        """
        rng = np.random.default_rng(99)
        n_draws = 2000
        alarms = sum(page_hinkley(rng.normal(0, 1, 30)).alarm for _ in range(n_draws))
        rate = alarms / n_draws
        lo, hi = _wilson(alarms, n_draws)
        assert lo >= 0.06 and hi <= 0.12, (
            f"PH null alarm rate {rate:.3f} (Wilson [{lo:.3f}, {hi:.3f}]) outside the "
            "documented band [0.06, 0.12] for 30-point streams. If it has dropped "
            "below 0.06, check that the scale estimate has not become non-causal."
        )

    def test_alarm_and_changepoint_on_step(self) -> None:
        values = np.concatenate([RNG.normal(10, 1, 15), RNG.normal(16, 1, 15)])
        res = page_hinkley(values)
        assert res.alarm
        assert res.direction == "up"
        assert res.change_index is not None
        assert 12 <= res.change_index <= 17

    def test_detects_downward(self) -> None:
        values = np.concatenate([np.full(12, 5.0) + RNG.normal(0, 0.5, 12), np.full(10, 1.0)])
        res = page_hinkley(values)
        assert res.alarm
        assert res.direction == "down"

    def test_short_stream_never_alarms(self) -> None:
        assert not page_hinkley(np.array([1.0, 2.0, 3.0])).alarm


class TestPSI:
    def test_stable_on_same_distribution(self) -> None:
        golden = RNG.normal(0, 1, 1000)
        current = RNG.normal(0, 1, 1000)
        assert psi(golden, current).label == "stable"

    def test_major_on_big_shift(self) -> None:
        golden = RNG.normal(0, 1, 1000)
        current = RNG.normal(3, 1, 1000)
        res = psi(golden, current)
        assert res.label == "major"
        assert res.value > 0.25

    def test_null_expectation_dominates_at_canary_scale(self) -> None:
        """The domain-of-validity guard's arithmetic: at canary scale PSI's
        null expectation alone exceeds 'major'; at production scale it is
        negligible. The check pipeline refuses to emit PSI in the former."""
        from dedrift.detectors import psi_null_expectation
        from dedrift.detectors.heuristic import PSI_MAJOR, PSI_MODERATE

        assert psi_null_expectation(30, 10) > PSI_MAJOR  # ~1.2: guaranteed flag
        assert psi_null_expectation(5000, 2000) < PSI_MODERATE / 2
        assert psi_null_expectation(0, 10) == float("inf")

    def test_bins_frozen_from_golden_are_reusable(self) -> None:
        golden = RNG.normal(0, 1, 1000)
        current = RNG.normal(1, 1, 1000)
        first = psi(golden, current)
        second = psi(golden, current, bins=first.bins)
        assert second.value == pytest.approx(first.value)
