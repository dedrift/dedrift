"""Embeddings (pinning, cache) and MMD detector tests, incl. calibration/power."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dedrift.check import run_check, set_golden_baseline
from dedrift.detectors import calibrate_mmd_floor, mmd_rbf_test
from dedrift.embeddings import (
    EmbedderMismatchError,
    embed_records,
    get_pinned_embedder,
    pin_embedder,
    resolve_embedder,
)
from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from dedrift.store import Store

RNG = np.random.default_rng(2027)


class TestHashEmbedder:
    def test_deterministic_and_normalized(self) -> None:
        embed = resolve_embedder("hash")
        a = embed(["hello world", "hello world", "different text"])
        assert np.allclose(a[0], a[1])
        assert not np.allclose(a[0], a[2])
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)

    def test_unknown_embedder_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown embedder"):
            resolve_embedder("nonsense")


class TestPinning:
    def test_pin_and_get(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            assert get_pinned_embedder(store) is None
            pin_embedder(store, "hash")
            assert get_pinned_embedder(store) == "hash"
            pin_embedder(store, "hash")  # idempotent re-pin is fine

    def test_refuses_repin_to_different(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            pin_embedder(store, "hash")
            with pytest.raises(EmbedderMismatchError, match="invalidates all history"):
                pin_embedder(store, "st:all-MiniLM-L6-v2")

    def test_embed_refuses_unpinned(self, tmp_path: Path) -> None:
        with (
            Store.init_project(tmp_path) as store,
            pytest.raises(ValueError, match="no embedder pinned"),
        ):
            embed_records(store, [])


class TestCache:
    def test_cache_round_trip_and_determinism(self, tmp_path: Path) -> None:
        config = SimConfig(n_canaries=4, repetitions=3, seed=3)
        records = SimAgent(config).run_cycles(2)
        with Store.init_project(tmp_path) as store:
            store.append_many(records)
            pin_embedder(store, "hash")
            first = embed_records(store, records)
            second = embed_records(store, records)  # from cache
            for rid in first:
                assert np.allclose(first[rid], second[rid])


class TestMMD:
    def test_detects_distribution_swap(self) -> None:
        ref = RNG.normal(0, 1, size=(80, 8))
        cur = RNG.normal(1.0, 1, size=(80, 8))
        out = mmd_rbf_test(ref, cur, seed=1)
        assert out.p_value < 0.01
        assert out.effect_size > 0

    def test_seeded_reproducible(self) -> None:
        ref = RNG.normal(0, 1, size=(40, 4))
        cur = RNG.normal(0, 1, size=(40, 4))
        assert repr(mmd_rbf_test(ref, cur, seed=9)) == repr(mmd_rbf_test(ref, cur, seed=9))

    def test_degenerate_nan(self) -> None:
        assert np.isnan(mmd_rbf_test(np.ones((1, 2)), np.ones((5, 2))).p_value)

    def test_floor_calibration_orders_correctly(self) -> None:
        # Floor from same-distribution cycles should be exceeded by a real
        # shift, with ONE shared bandwidth for floor and observation
        # (commensurability, per owner review finding #4).
        from dedrift.detectors.mmd import median_heuristic_bandwidth

        cycles = [RNG.normal(0, 1, size=(40, 6)) for _ in range(5)]
        shifted = RNG.normal(1.5, 1, size=(40, 6))
        sigma = median_heuristic_bandwidth(np.vstack([cycles[0], shifted]))
        floor = calibrate_mmd_floor(cycles, sigma=sigma)
        observed = mmd_rbf_test(cycles[0], shifted, seed=2, sigma=sigma).effect_size
        assert floor > 0
        assert observed > floor

    def test_floor_uncalibratable_below_five_cycles(self) -> None:
        """Four cycles give six pairs; the 0.95 quantile of six draws is the
        maximum in all but name. The minimum was raised from three cycles to
        five (ten pairs) rather than keep calling that a calibrated
        threshold."""
        for k in (2, 3, 4):
            cycles = [RNG.normal(0, 1, size=(40, 6)) for _ in range(k)]
            assert calibrate_mmd_floor(cycles, sigma=1.0) == 0.0, f"{k} cycles"
        cycles = [RNG.normal(0, 1, size=(40, 6)) for _ in range(5)]
        assert calibrate_mmd_floor(cycles, sigma=1.0) > 0.0


@pytest.mark.calibration
class TestMMDCalibration:
    def test_null_false_alarm_rate(self) -> None:
        """MMD permutation test at alpha=0.05 under the null: measured rate
        within [0.02, 0.09] over 200 simulations (binomial band ~ +/-0.03)."""
        rng = np.random.default_rng(77)
        rejections = 0
        n_sims = 200
        for _ in range(n_sims):
            ref = rng.normal(0, 1, size=(60, 6))
            cur = rng.normal(0, 1, size=(60, 6))
            if mmd_rbf_test(ref, cur, n_permutations=199, seed=5).p_value < 0.05:
                rejections += 1
        rate = rejections / n_sims
        assert 0.02 <= rate <= 0.09, f"MMD null rate {rate:.3f}"


@pytest.mark.power
class TestMMDPower:
    def test_power_for_mean_shift(self) -> None:
        """MMD power for a 0.75 SD mean shift in 6-d at n=m=60 exceeds 90%."""
        rng = np.random.default_rng(88)
        detections = 0
        n_sims = 100
        for _ in range(n_sims):
            ref = rng.normal(0, 1, size=(60, 6))
            cur = rng.normal(0.75, 1, size=(60, 6))
            if mmd_rbf_test(ref, cur, n_permutations=199, seed=6).p_value < 0.05:
                detections += 1
        assert detections / n_sims >= 0.90


class TestEndToEndWithEmbeddings:
    def _project(self, tmp_path: Path, change_cycle: int | None) -> Store:
        config = SimConfig(
            n_canaries=12,
            repetitions=5,
            post=drifted_profile(BehaviorProfile()),
            change_cycle=change_cycle,
            seed=21,
        )
        store = Store.init_project(tmp_path)
        records = SimAgent(config).run_cycles(6)
        store.append_many(records)
        cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        set_golden_baseline(store, cycles[:3])
        pin_embedder(store, "hash")
        return store

    def test_mmd_and_displacement_join_the_battery(self, tmp_path: Path) -> None:
        store = self._project(tmp_path, change_cycle=None)
        result = run_check(store)
        test_names = {t.outcome.test for t in result.tests}
        signatures = {t.signature for t in result.tests}
        assert "mmd" in test_names
        assert "semantic_displacement" in signatures
        store.close()

    def test_null_run_with_embeddings_stays_quiet(self, tmp_path: Path) -> None:
        store = self._project(tmp_path, change_cycle=None)
        result = run_check(store)
        assert result.n_alerts == 0
        store.close()


class TestDisplacementIsLeaveOneOut:
    """The reference window must not be scored in-sample.

    A reference record helps define its own canary centroid. Scored against
    the full mean it sits closer than a current-cycle record does purely by
    construction, so every test on this column would see a location and
    dispersion shift on unchanged data. The centroid is therefore recomputed
    without the record being scored; these tests pin that, because the bug
    is invisible in any single check and shows up only as a channel that
    alerts more than it should.
    """

    def _embeddings(self, n_ref: int, dim: int, seed: int) -> tuple[dict, list, list]:
        from dedrift.schema import AgentConfig, InteractionRecord

        cfg = AgentConfig(model="m", prompt_hash="p", tool_schema_hash="t", agent_version="v")
        rng = np.random.default_rng(seed)
        # Realistic geometry: outputs for one canary cluster around a canary
        # mean with modest noise. Zero-mean isotropic vectors would make the
        # centroid pure noise and cosine distance meaningless.
        mu = rng.normal(size=dim)
        mu = mu / np.linalg.norm(mu)
        noise = 0.3
        records, emb = [], {}
        for i in range(n_ref + n_ref):
            rid = f"r{i}"
            cycle = "cycle-0000" if i < n_ref else "cycle-0001"
            records.append(
                InteractionRecord(
                    id=rid,
                    source="canary",
                    input={"text": "x"},
                    output={"text": "y"},
                    canary_id="c1",
                    cycle_id=cycle,
                    config=cfg,
                )
            )
            emb[rid] = mu + noise * rng.normal(size=dim)
        return emb, records, ["cycle-0000"]

    def test_reference_and_current_have_equal_null_means(self) -> None:
        """Under H0 both windows must have the same expected displacement.

        Measured over many draws: with the full-mean centroid the reference
        window sits systematically closer; leave-one-out removes the gap.
        The assertion is on the ratio of *grand* means, not the mean of
        per-draw ratios — with 12 records a window, the denominator is noisy
        enough that averaging ratios is visibly upward-biased by Jensen and
        would fail a correct implementation.

        Theory for squared distance: in-sample gives ``(1+1/n)/(1-1/n)``
        = 1.18 at n=12, leave-one-out gives ``(1+1/n)/(1+1/(n-1))`` = 0.993.
        The residual 0.7% is the one-record difference in centroid precision
        and is far below anything the battery can resolve.
        """
        import pandas as pd

        from dedrift.check import _add_semantic_displacement

        ref_vals: list[float] = []
        cur_vals: list[float] = []
        for seed in range(400):
            emb, records, golden = self._embeddings(n_ref=12, dim=8, seed=seed)
            frame = pd.DataFrame(
                {
                    "record_id": [r.id for r in records],
                    "cycle_id": [r.cycle_id for r in records],
                }
            )
            _add_semantic_displacement(frame, records, emb, golden)
            ref_vals.extend(frame[frame["cycle_id"] == "cycle-0000"]["semantic_displacement"])
            cur_vals.extend(frame[frame["cycle_id"] == "cycle-0001"]["semantic_displacement"])
        mean_ratio = float(np.mean(cur_vals) / np.mean(ref_vals))
        assert 0.96 <= mean_ratio <= 1.04, (
            f"current/reference displacement ratio {mean_ratio:.3f} — the reference "
            "window is being scored in-sample"
        )

    def test_in_sample_centroid_would_be_visibly_biased(self) -> None:
        """The contrast that makes the test above meaningful.

        Without leave-one-out the same data gives a ratio near 1.18, a
        location shift the battery would happily call drift. Asserting the
        broken behaviour is large is what turns the previous test from
        "passes" into "passes for the right reason".
        """
        rng = np.random.default_rng(11)
        n, dim = 12, 8
        ref_in, cur_in = [], []
        for _ in range(400):
            mu = rng.normal(size=dim)
            mu = mu / np.linalg.norm(mu)
            ref = mu + 0.3 * rng.normal(size=(n, dim))
            cur = mu + 0.3 * rng.normal(size=(n, dim))
            centroid = ref.mean(axis=0)

            def cos(x: np.ndarray, c: np.ndarray = centroid) -> float:
                return float(1 - x @ c / (np.linalg.norm(x) * np.linalg.norm(c)))

            ref_in.extend(cos(x) for x in ref)
            cur_in.extend(cos(x) for x in cur)
        biased_ratio = float(np.mean(cur_in) / np.mean(ref_in))
        assert biased_ratio > 1.15, (
            f"expected a large in-sample bias to guard against, got {biased_ratio:.3f}"
        )

    def test_single_reference_record_yields_nan_not_zero(self) -> None:
        """One reference record cannot be scored leave-one-out.

        Returning 0.0 (distance to itself) would look like a perfectly
        stable canary; NaN is dropped by the battery, which is correct.
        """
        import pandas as pd

        from dedrift.check import _add_semantic_displacement

        emb, records, golden = self._embeddings(n_ref=1, dim=8, seed=3)
        frame = pd.DataFrame(
            {
                "record_id": [r.id for r in records],
                "cycle_id": [r.cycle_id for r in records],
            }
        )
        _add_semantic_displacement(frame, records, emb, golden)
        ref_val = frame[frame["cycle_id"] == "cycle-0000"]["semantic_displacement"].iloc[0]
        assert np.isnan(ref_val)
