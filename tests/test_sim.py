"""Simulator reproducibility and scripted-config-change tests."""

from __future__ import annotations

from pathlib import Path

from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from dedrift.store import Store


def small_config(**overrides: object) -> SimConfig:
    base: dict[str, object] = {"n_canaries": 5, "repetitions": 3, "seed": 42}
    base.update(overrides)
    return SimConfig(**base)  # type: ignore[arg-type]


class TestReproducibility:
    def test_same_seed_identical_stream(self) -> None:
        a = SimAgent(small_config()).run_cycles(3)
        b = SimAgent(small_config()).run_cycles(3)
        assert [r.to_jsonl() for r in a] == [r.to_jsonl() for r in b]

    def test_different_seed_differs(self) -> None:
        a = SimAgent(small_config()).run_cycles(2)
        b = SimAgent(small_config(seed=43)).run_cycles(2)
        assert [r.to_jsonl() for r in a] != [r.to_jsonl() for r in b]


class TestStructure:
    def test_cycle_size(self) -> None:
        records = SimAgent(small_config()).run_cycle(0)
        assert len(records) == 5 * 3
        assert {r.canary_id for r in records} == {f"canary-{i:03d}" for i in range(5)}
        assert all(r.repetition in {1, 2, 3} for r in records)

    def test_timestamps_monotonic(self) -> None:
        records = SimAgent(small_config()).run_cycles(2)
        timestamps = [r.ts for r in records]
        assert timestamps == sorted(timestamps)


class TestConfigChange:
    def test_null_scenario_single_fingerprint(self) -> None:
        records = SimAgent(small_config()).run_cycles(4)
        assert len({r.config_fingerprint for r in records}) == 1

    def test_change_cycle_switches_fingerprint(self) -> None:
        config = small_config(change_cycle=2, post=drifted_profile(BehaviorProfile()))
        agent = SimAgent(config)
        pre = agent.run_cycle(0) + agent.run_cycle(1)
        post = agent.run_cycle(2) + agent.run_cycle(3)
        pre_fps = {r.config_fingerprint for r in pre}
        post_fps = {r.config_fingerprint for r in post}
        assert len(pre_fps) == 1
        assert len(post_fps) == 1
        assert pre_fps != post_fps

    def test_behavior_shifts_after_change(self) -> None:
        base = BehaviorProfile()
        config = SimConfig(
            n_canaries=20,
            repetitions=7,
            post=drifted_profile(base),
            change_cycle=1,
            seed=7,
        )
        agent = SimAgent(config)
        pre = agent.run_cycle(0)
        post = agent.run_cycle(1)
        pre_len = sum(len(r.output.text.split()) for r in pre) / len(pre)
        post_len = sum(len(r.output.text.split()) for r in post) / len(post)
        assert post_len > pre_len * 1.2  # 1.5x shift, generous test margin


class TestEndToEnd:
    def test_phase0_round_trip(self, tmp_path: Path) -> None:
        """Phase 0 done-when: init + simulated logging round-trips."""
        config = small_config(change_cycle=1, post=drifted_profile(BehaviorProfile()))
        records = SimAgent(config).run_cycles(3)
        with Store.init_project(tmp_path) as store:
            store.append_many(records)
            assert store.count_records() == len(records)
            assert store.read_records() == records
            events = store.config_events()
            assert len(events) == 2  # initial config + scripted change
