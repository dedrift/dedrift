"""Calibration and power suites — the project's soul (CLAUDE.md testing rules).

Calibration: EVERY p-valued detector in the battery has a measured null
false-alarm rate with a documented acceptance band, and the full pipeline's
per-check alert probability on stable agents is bounded with a Wilson
interval over many seeded runs — not a pass/fail on a handful.

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
from dedrift.config import Materiality, ProjectConfig
from dedrift.detectors import (
    ad_test,
    ks_test,
    levene_test,
    p95_permutation_test,
    two_proportion_z_test,
    welch_t_test,
)
from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from dedrift.store import Store

#: Sample size per window used in per-detector calibration: 18 canaries x 7
#: reps, matching the documented default scale.
N = 126


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Upper bound of the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 1.0
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return float((centre + margin) / denom)


@pytest.mark.calibration
class TestPerDetectorNullCalibration:
    """Measured false-alarm rate at alpha=0.05 for every p-valued detector.

    Acceptance band [0.02, 0.09] over 400 null simulations: the binomial 95%
    band around 0.05 is roughly +/-0.021, widened slightly for tests that are
    deliberately conservative (continuity correction) or discrete
    (permutation p-values).
    """

    def _rate(self, reject_fn, n_sims: int = 400) -> float:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(1234)
        rejections = sum(bool(reject_fn(rng)) for _ in range(n_sims))
        return rejections / n_sims

    def test_ks(self) -> None:
        rate = self._rate(
            lambda rng: ks_test(rng.normal(0, 1, N), rng.normal(0, 1, N)).p_value < 0.05
        )
        assert 0.02 <= rate <= 0.09, f"KS null rate {rate:.3f}"

    def test_anderson_darling(self) -> None:
        # AD is corroboration-only, but its printed p-values must still not
        # be anticonservative. SciPy caps p to [0.001, 0.25]; under the null
        # the rejection rate at 0.05 should stay at or below nominal.
        rate = self._rate(
            lambda rng: ad_test(rng.normal(0, 1, N), rng.normal(0, 1, N)).p_value < 0.05
        )
        assert rate <= 0.09, f"AD null rate {rate:.3f}"

    def test_welch(self) -> None:
        rate = self._rate(
            lambda rng: welch_t_test(rng.normal(0, 1, N), rng.normal(0, 1, N)).p_value < 0.05
        )
        assert 0.02 <= rate <= 0.09, f"Welch null rate {rate:.3f}"

    def test_levene_normal(self) -> None:
        rate = self._rate(
            lambda rng: levene_test(rng.normal(0, 1, N), rng.normal(0, 1, N)).p_value < 0.05
        )
        assert 0.02 <= rate <= 0.09, f"Levene null rate {rate:.3f}"

    def test_levene_heavy_tailed(self) -> None:
        # Brown-Forsythe's robustness claim, measured under t(3) tails.
        rate = self._rate(
            lambda rng: levene_test(rng.standard_t(3, N), rng.standard_t(3, N)).p_value < 0.05
        )
        assert 0.02 <= rate <= 0.10, f"Levene t(3) null rate {rate:.3f}"

    def test_p95_permutation(self) -> None:
        rate = self._rate(
            lambda rng: (
                p95_permutation_test(
                    rng.normal(0, 1, N), rng.normal(0, 1, N), n_permutations=199, seed=3
                ).p_value
                < 0.05
            ),
            n_sims=300,
        )
        assert 0.02 <= rate <= 0.09, f"P95 permutation null rate {rate:.3f}"

    def test_two_proportion_z(self) -> None:
        def reject(rng: np.random.Generator) -> bool:
            a = int(rng.binomial(N, 0.1))
            b = int(rng.binomial(N, 0.1))
            return two_proportion_z_test(a, N, b, N).p_value < 0.05

        rate = self._rate(reject)
        # Continuity correction is conservative at these sample sizes.
        assert 0.005 <= rate <= 0.07, f"z-test null rate {rate:.3f}"

    # MMD's null calibration lives in tests/test_embeddings_mmd.py.


def _light_config() -> ProjectConfig:
    """Reduced-cost config for the many-run pipeline calibration."""
    return ProjectConfig(permutations=200, materiality=Materiality())


@pytest.mark.calibration
class TestPipelineNullCalibration:
    def test_pipeline_alert_rate_bounded_over_500_null_runs(self, tmp_path: Path) -> None:
        """The headline claim, measured properly — at a STATED scale.

        500 seeded stable-agent runs through the full pipeline (structural
        signatures; embeddings excluded for runtime). The per-check
        probability of ANY alert must have a Wilson 95% upper bound below
        0.05. With 500 runs this bounds the true rate meaningfully — unlike
        a pass/fail on 20 runs, which would pass 39% of the time even if the
        true alert rate were 10%.

        Scale caveat (stated in docs/statistics.md): these runs use
        12 canaries x 5 reps, not the 18 x 7 default, for runtime. Under a
        calibrated null the alert rate is approximately scale-free, but the
        materiality gates interact with N, so the bound is claimed AT THIS
        SCALE and the docs say so.

        This test also MEASURES the uncalibrated flag channel (PSI +
        Page-Hinkley) on the same null runs. Flags carry no FDR control —
        a stable agent will show flags on a nontrivial fraction of checks —
        and the docs publish this measured rate as a known limitation
        rather than pretending flags are alerts. The loose ceiling below
        only guards against regression to near-certain flagging.
        """
        n_runs = 500
        cfg = _light_config()
        alerting_runs = 0
        flagging_runs = 0
        for seed in range(n_runs):
            root = tmp_path / f"run{seed}"
            root.mkdir()
            sim = SimConfig(n_canaries=12, repetitions=5, change_cycle=None, seed=seed)
            with Store.init_project(root) as store:
                records = SimAgent(sim).run_cycles(6)
                store.append_many(records)
                cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
                set_golden_baseline(store, cycles[:3])
                result = run_check(store, config=cfg)
                if result.n_alerts > 0:
                    alerting_runs += 1
                if result.flags:
                    flagging_runs += 1
        bound = wilson_upper(alerting_runs, n_runs)
        flag_rate = flagging_runs / n_runs
        print(
            f"\npipeline null (12 canaries x 5 reps, 500 runs): "
            f"alerts {alerting_runs}/{n_runs} (Wilson upper {bound:.4f}), "
            f"any-flag rate {flag_rate:.3f}"
        )
        assert bound < 0.05, (
            f"pipeline null alert rate: {alerting_runs}/{n_runs} runs alerted "
            f"(Wilson 95% upper bound {bound:.4f} >= 0.05)"
        )
        assert flag_rate < 0.9, (
            f"uncalibrated flag channel fired on {flag_rate:.0%} of stable runs — "
            "regression beyond the documented limitation"
        )


@pytest.mark.power
class TestPower:
    def test_model_swap_detected_with_correct_attribution(self, tmp_path: Path) -> None:
        """A simulated model swap is detected in the changed cycle and
        attributed to the config event, in >= 9 of 10 seeded runs."""
        detected = 0
        attributed = 0
        n_runs = 10
        cfg = _light_config()
        for seed in range(100, 100 + n_runs):
            root = tmp_path / f"run{seed}"
            root.mkdir()
            sim = SimConfig(
                n_canaries=18,
                repetitions=7,
                post=drifted_profile(BehaviorProfile()),
                change_cycle=7,
                seed=seed,
            )
            with Store.init_project(root) as store:
                records = SimAgent(sim).run_cycles(8)
                store.append_many(records)
                cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
                set_golden_baseline(store, cycles[:3])
                result = run_check(store, config=cfg)
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
            ref = rng.normal(0, 1, N)
            cur = rng.normal(0.5, 1, N)
            if ks_test(ref, cur).p_value < 0.05:
                detections += 1
        assert detections / n_sims >= 0.90

    def test_ks_gate_passes_shape_change_with_equal_means(self) -> None:
        """The review's core case: a variance-only change (equal means) must
        be significant AND material under the KS-D gate — Cohen's d would
        have wrongly gated it out. The gate value is read from config, not
        hardcoded, so a default change cannot silently break this test."""
        gate = Materiality().ks_distance
        rng = np.random.default_rng(789)
        material_hits = 0
        n_sims = 100
        for _ in range(n_sims):
            ref = rng.normal(0, 1, N)
            cur = rng.normal(0, 2.2, N)  # equal means, very different shape
            out = ks_test(ref, cur)
            if out.p_value < 0.01 and out.effect_size >= gate:
                material_hits += 1
            # KS now reports D as its effect — identical to the gated scale.
            assert out.effect_size == out.statistic
            # Cohen's d (the Welch corroboration diagnostic) stays small:
            # the old gate's blind spot, kept as a regression witness.
            assert abs(welch_t_test(ref, cur).effect_size) < 0.6
        assert material_hits / n_sims >= 0.85

    def test_ks_gate_binding_scale_is_as_documented(self) -> None:
        """The configured ks_distance must bind exactly where the docs say.

        The raw-alpha=0.05 two-sample critical value is
        D_crit ~ c(0.05) * sqrt((n+m)/(n*m)) with c(0.05) ~ 1.358. The docs
        state that at per-family scale (e.g. 21 current vs 105 pooled
        reference) D_crit exceeds the default gate — so significance, not
        the gate, is the operative filter there — and that the gate starts
        binding for equal arms only at n >~ 2*(c/gate)^2 (~165 at 0.15).
        This test recomputes both claims from the configured value, so a
        config change that silently makes the gate decorative-everywhere or
        binding-nowhere fails loudly.
        """
        gate = Materiality().ks_distance
        c05 = 1.358

        def d_crit(n: int, m: int) -> float:
            return c05 * np.sqrt((n + m) / (n * m))

        # Documented default family scale: 3 canaries x 7 reps = 21 per
        # cycle; rolling reference pools 5 cycles. BH only raises this bar.
        assert d_crit(105, 21) > gate, (
            f"gate {gate} would bind at family scale (D_crit {d_crit(105, 21):.3f}); "
            "docs claim significance dominates there — update docs or gate"
        )
        # Equal-arm binding threshold: gate must bind at realistic large n
        # (i.e. it is a real guard, not decoration at every feasible scale).
        n_bind = int(np.ceil(2 * (c05 / gate) ** 2))
        assert d_crit(n_bind, n_bind) <= gate
        assert n_bind <= 500, (
            f"gate {gate} only binds beyond n={n_bind} per arm — decorative "
            "at any feasible canary scale; lower it or re-document"
        )

    def test_rate_power_documented_floor(self) -> None:
        """Two-proportion z power for a 10pp refusal shift (5% -> 15%) at
        n=126 per arm exceeds 70% — and the docs must say 2pp shifts are NOT
        reliably detectable at this N (they are below the power floor)."""
        rng = np.random.default_rng(654)
        detections = 0
        n_sims = 200
        for _ in range(n_sims):
            a = int(rng.binomial(N, 0.05))
            b = int(rng.binomial(N, 0.15))
            if two_proportion_z_test(a, N, b, N).p_value < 0.05:
                detections += 1
        assert detections / n_sims >= 0.70
