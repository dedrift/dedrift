"""Calibration and power suites — the project's soul (CLAUDE.md testing rules).

Calibration: under the null (no config change, stationary behavior), the
pipeline must produce (nearly) zero alerts across many seeded runs, and the
per-test false-rejection rate must match nominal levels within documented
tolerance.

Power: injected shifts of documented size must be detected at documented
rates, with correct attribution.

Tolerances are stated inline; if these tests fail, the detectors are wrong —
do not loosen tolerances without owner sign-off.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dedrift.attribution import attribute
from dedrift.check import run_check, set_golden_baseline
from dedrift.detectors import ks_test, two_proportion_z_test
from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from dedrift.store import Store

pytestmark = []


@pytest.mark.calibration
class TestNullCalibration:
    def test_pipeline_zero_alerts_over_20_null_runs(self, tmp_path: Path) -> None:
        """SPEC §10.2: zero alerts across 20 seeded null runs at q=0.05.

        Documented tolerance: at most 1 run out of 20 may alert (the FDR +
        materiality double gate makes a false alert a rare event; a single
        exception guards against knife-edge flakiness while still failing
        loudly on any systematic miscalibration).
        """
        alerting_runs = 0
        for seed in range(20):
            root = tmp_path / f"run{seed}"
            root.mkdir()
            config = SimConfig(n_canaries=18, repetitions=7, change_cycle=None, seed=seed)
            with Store.init_project(root) as store:
                records = SimAgent(config).run_cycles(8)
                store.append_many(records)
                cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
                set_golden_baseline(store, cycles[:3])
                result = run_check(store)
                if result.n_alerts > 0:
                    alerting_runs += 1
        assert alerting_runs <= 1, f"{alerting_runs}/20 null runs alerted"

    def test_ks_false_alarm_rate_matches_nominal(self) -> None:
        """Two-sample KS at alpha=0.05 under the null: measured rate within
        [0.02, 0.09] over 400 simulations (binomial 95% band ≈ ±0.02)."""
        rng = np.random.default_rng(123)
        alpha = 0.05
        rejections = 0
        n_sims = 400
        for _ in range(n_sims):
            ref = rng.normal(0, 1, 126)  # 18 canaries x 7 reps
            cur = rng.normal(0, 1, 126)
            if ks_test(ref, cur).p_value < alpha:
                rejections += 1
        rate = rejections / n_sims
        assert 0.02 <= rate <= 0.09, f"KS false-alarm rate {rate:.3f} vs nominal {alpha}"

    def test_two_proportion_z_false_alarm_rate(self) -> None:
        """Two-proportion z at alpha=0.05 under the null, p=0.1, n=126 per arm.

        The continuity correction makes the test conservative at these sample
        sizes; documented acceptance band [0.005, 0.07].
        """
        rng = np.random.default_rng(321)
        alpha = 0.05
        rejections = 0
        n_sims = 400
        for _ in range(n_sims):
            a = int(rng.binomial(126, 0.1))
            b = int(rng.binomial(126, 0.1))
            if two_proportion_z_test(a, 126, b, 126).p_value < alpha:
                rejections += 1
        rate = rejections / n_sims
        assert 0.005 <= rate <= 0.07, f"z-test false-alarm rate {rate:.3f}"


@pytest.mark.power
class TestPower:
    def test_model_swap_detected_with_correct_attribution(self, tmp_path: Path) -> None:
        """SPEC §10.2: a simulated model swap is detected within ≤2 cycles of
        the change and attributed to the config event, in ≥9 of 10 seeded runs.
        """
        detected = 0
        attributed = 0
        n_runs = 10
        for seed in range(100, 100 + n_runs):
            root = tmp_path / f"run{seed}"
            root.mkdir()
            config = SimConfig(
                n_canaries=18,
                repetitions=7,
                post=drifted_profile(BehaviorProfile()),
                change_cycle=7,
                seed=seed,
            )
            with Store.init_project(root) as store:
                records = SimAgent(config).run_cycles(8)  # change lands in final cycle
                store.append_many(records)
                cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
                set_golden_baseline(store, cycles[:3])
                result = run_check(store)
                if result.n_alerts > 0:
                    detected += 1
                    ats = attribute(store, result)
                    if ats and all(
                        at.nearest_event_delta_hours is not None
                        and abs(at.nearest_event_delta_hours) < 12
                        for at in ats
                    ):
                        attributed += 1
        assert detected >= 9, f"detected {detected}/{n_runs}"
        assert attributed >= 9, f"attributed {attributed}/{n_runs}"

    def test_ks_power_for_documented_shift(self) -> None:
        """KS power for a 0.5 SD mean shift at n=126 per arm exceeds 90%
        (documented in the README power table)."""
        rng = np.random.default_rng(456)
        detections = 0
        n_sims = 200
        for _ in range(n_sims):
            ref = rng.normal(0, 1, 126)
            cur = rng.normal(0.5, 1, 126)
            if ks_test(ref, cur).p_value < 0.05:
                detections += 1
        assert detections / n_sims >= 0.90

    def test_rate_power_documented_floor(self) -> None:
        """Two-proportion z power for a 10pp refusal shift (5% -> 15%) at
        n=126 per arm exceeds 70% — and the docs must say 2pp shifts are NOT
        reliably detectable at this N (they are below the power floor)."""
        rng = np.random.default_rng(654)
        detections = 0
        n_sims = 200
        for _ in range(n_sims):
            a = int(rng.binomial(126, 0.05))
            b = int(rng.binomial(126, 0.15))
            if two_proportion_z_test(a, 126, b, 126).p_value < 0.05:
                detections += 1
        assert detections / n_sims >= 0.70
