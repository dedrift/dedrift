"""The anytime path end to end: state, epochs, budget, and CLI parity.

These tests are about the *plumbing* being trustworthy — state that survives
between invocations, epochs that reset when they must, and a budget that
stays correct when the operator changes the suite. The statistical validity
of the construction is established in ``test_evalues.py`` and its
trajectory-level behaviour in ``test_evalues_anytime.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

import dedrift.anytime as anytime_module
from dedrift.anytime import (
    load_pool,
    load_processed_cycles,
    load_states,
    run_anytime_check,
    wealth_table,
)
from dedrift.check import run_check, set_golden_baseline
from dedrift.cli import app
from dedrift.config import AnytimeConfig, ProjectConfig
from dedrift.embeddings import pin_embedder
from dedrift.evalues import EValueOutcome, PriorState
from dedrift.evalues.rates import per_process_gamma
from dedrift.report import render_anytime_report
from dedrift.schema import InteractionInput, InteractionRecord, Source
from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from dedrift.store import FinalizedSnapshotChangedError, Store

runner = CliRunner()


def project(tmp_path: Path, cycles: int = 6, change_cycle: int | None = None) -> Store:
    cfg = SimConfig(
        n_canaries=18,
        repetitions=7,
        post=drifted_profile(BehaviorProfile()),
        change_cycle=change_cycle,
        seed=5,
    )
    store = Store.init_project(tmp_path)
    records = SimAgent(cfg).run_cycles(cycles)
    store.append_many(records)
    ids = sorted({r.cycle_id for r in records if r.cycle_id is not None})
    set_golden_baseline(store, ids[:3])
    return store


def make_records(*, cycles: int = 6, change_cycle: int | None = None) -> list[InteractionRecord]:
    """Deterministic records used by mutation-focused pipeline tests."""
    cfg = SimConfig(
        n_canaries=18,
        repetitions=7,
        post=drifted_profile(BehaviorProfile()),
        change_cycle=change_cycle,
        seed=5,
    )
    return SimAgent(cfg).run_cycles(cycles)


def store_records(tmp_path: Path, records: list[InteractionRecord]) -> Store:
    """Create a store whose anytime epoch predates all non-golden records."""
    store = Store.init_project(tmp_path)
    ids = list(dict.fromkeys(r.cycle_id for r in records if r.cycle_id is not None))
    golden = set(ids[:3])
    store.append_many([record for record in records if record.cycle_id in golden])
    set_golden_baseline(store, ids[:3])
    declaration = run_anytime_check(store)
    assert declaration.processed_cycles == ()
    store.append_many([record for record in records if record.cycle_id not in golden])
    return store


def stamp_suite(record: InteractionRecord, fingerprint: str) -> InteractionRecord:
    """Stamp the full-suite identity that production CanaryRunner records carry."""
    metadata = {**record.input.metadata, "_dedrift": {"suite_fingerprint": fingerprint}}
    return record.model_copy(
        update={"input": InteractionInput(text=record.input.text, metadata=metadata)}
    )


class TestBudgetArithmetic:
    """The error that would misstate the guarantee five-fold."""

    def test_gamma_is_split_across_the_pool(self) -> None:
        """Coverage failures union-bound across the frozen process pool.

        ``alpha = alpha' + gamma`` is the per-process statement. Battery-wide
        it is ``alpha' + sum_i gamma_i``, so the total must be divided by the
        pool size — at 24 processes, not doing so turns a claimed 0.05 into
        an actual 0.28.
        """
        assert per_process_gamma(0.01, 24) == pytest.approx(0.01 / 24)
        assert per_process_gamma(0.01, 1) == pytest.approx(0.01)
        assert per_process_gamma(0.01, 0) == pytest.approx(0.01)  # degenerate: no split

    def test_config_budget_decomposes(self) -> None:
        ac = AnytimeConfig()
        assert ac.alpha == pytest.approx(0.05)
        assert ac.alpha_prime == pytest.approx(ac.alpha - ac.gamma_total)

    def test_reported_gamma_reflects_the_live_pool(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        res = run_anytime_check(store)
        assert res.n_processes > 1
        assert res.gamma_per_process == pytest.approx(res.gamma_total / res.n_processes)
        assert res.alpha == pytest.approx(res.alpha_prime + res.gamma_total)
        store.close()


class TestStatePersistence:
    def test_only_cycles_after_declaration_are_folded_once_and_recheck_is_read_only(
        self, tmp_path: Path
    ) -> None:
        store = project(tmp_path)
        declaration = run_anytime_check(store)
        assert declaration.processed_cycles == ()
        assert declaration.verdict == "NO CURRENT OBSERVATION"
        assert all(state.cycles == 0 for state in load_states(store).values())

        future = make_records(cycles=9)
        store.append_many(
            [
                record
                for record in future
                if record.cycle_id in {"cycle-0006", "cycle-0007", "cycle-0008"}
            ]
        )
        first = run_anytime_check(store)
        saved = load_states(store)
        assert len(saved) == first.n_processes
        assert first.processed_cycles == ("cycle-0006", "cycle-0007", "cycle-0008")
        assert load_processed_cycles(store, first.fingerprint) == set(first.processed_cycles)
        assert all(s.cycles == 3 for s in saved.values())
        compact = (
            store.connect()
            .execute(
                "SELECT params_json, snapshot_record_ids_json FROM checks "
                "WHERE baseline_kind = 'anytime' ORDER BY id DESC LIMIT 1"
            )
            .fetchone()
        )
        assert compact[1] is None
        assert '"snapshot_record_ids"' not in str(compact[0])
        checks_before = (
            store.connect()
            .execute("SELECT COUNT(*) FROM checks WHERE baseline_kind = 'anytime'")
            .fetchone()[0]
        )
        db_files = (store.index_path, Path(f"{store.index_path}-wal"))
        db_before = {path: path.read_bytes() for path in db_files if path.exists()}

        second = run_anytime_check(store)
        again = load_states(store)
        checks_after = (
            store.connect()
            .execute("SELECT COUNT(*) FROM checks WHERE baseline_kind = 'anytime'")
            .fetchone()[0]
        )
        assert second.processed_cycles == ()
        assert again == saved
        assert checks_after == checks_before
        assert {path: path.read_bytes() for path in db_files if path.exists()} == db_before
        assert second.n_processes == first.n_processes
        store.close()

    def test_cycle_open_before_declaration_is_never_replayed_after_finalization(
        self, tmp_path: Path
    ) -> None:
        records = make_records(cycles=5)
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))

        preexisting_open = [record for record in records if record.cycle_id == "cycle-0003"]
        store.append_many(preexisting_open, finalize_cycles=False)
        declaration = run_anytime_check(store)
        assert declaration.processed_cycles == ()

        store.finalize_cycle("cycle-0003")
        after_finalization = run_anytime_check(store)
        assert after_finalization.processed_cycles == ()
        assert all(state.cycles == 0 for state in load_states(store).values())

        store.append_many([record for record in records if record.cycle_id == "cycle-0004"])
        future = run_anytime_check(store)
        assert future.processed_cycles == ("cycle-0004",)
        assert all(state.cycles == 1 for state in load_states(store).values())
        store.close()

    def test_post_declaration_cycles_wait_for_contiguous_open_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = make_records(cycles=5)
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))
        assert run_anytime_check(store).processed_cycles == ()
        monkeypatch.setattr(
            anytime_module,
            "rate_evalue",
            lambda *_args, **_kwargs: EValueOutcome(0.0, True, "neutral scripted evidence"),
        )

        store.append_many(
            [record for record in records if record.cycle_id == "cycle-0003"],
            finalize_cycles=False,
        )
        store.append_many([record for record in records if record.cycle_id == "cycle-0004"])
        deferred = run_anytime_check(store)

        assert deferred.current_cycle == "cycle-0004"
        assert deferred.processed_cycles == ()
        assert deferred.coverage_status == "NO OBSERVATION"
        assert deferred.verdict == "NO CURRENT OBSERVATION"
        assert all(state.cycles == 0 for state in load_states(store).values())

        store.finalize_cycle("cycle-0003")
        folded = run_anytime_check(store)

        assert folded.current_cycle == "cycle-0003"
        assert folded.processed_cycles == ("cycle-0003", "cycle-0004")
        assert load_processed_cycles(store, folded.fingerprint) == {
            "cycle-0003",
            "cycle-0004",
        }
        assert all(state.cycles == 2 for state in load_states(store).values())
        store.close()

    def test_crash_tail_observation_is_inside_the_declaration_watermark(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = make_records(cycles=4)
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))
        crash_tail = [record for record in records if record.cycle_id == "cycle-0003"]
        original_index_record = Store._index_record

        def fail_after_fsync(target: Store, conn: object, record: InteractionRecord) -> None:
            del target, conn, record
            raise RuntimeError("simulated index failure after JSONL fsync")

        monkeypatch.setattr(Store, "_index_record", fail_after_fsync)
        with pytest.raises(RuntimeError, match="after JSONL fsync"):
            store.append_many(crash_tail, finalize_cycles=False)
        monkeypatch.setattr(Store, "_index_record", original_index_record)

        declaration = run_anytime_check(store)
        assert declaration.processed_cycles == ()
        store.close()

        with Store(tmp_path) as reopened:
            reopened.finalize_cycle("cycle-0003")
            result = run_anytime_check(reopened)
            assert result.processed_cycles == ()
            assert all(state.cycles == 0 for state in load_states(reopened).values())

    def test_legacy_epoch_boundary_migrates_to_observation_sequence(self, tmp_path: Path) -> None:
        records = make_records(cycles=5)
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id != "cycle-0004"])
        set_golden_baseline(store, ["cycle-0000", "cycle-0001", "cycle-0002"])
        declaration = run_anytime_check(store)
        assert declaration.current_cycle == "cycle-0003"
        index_path = store.index_path
        store.close()

        with sqlite3.connect(index_path) as legacy:
            legacy.execute("ALTER TABLE anytime_epochs DROP COLUMN start_after_open_seq")

        with Store(tmp_path) as migrated:
            boundary = (
                migrated.connect()
                .execute(
                    "SELECT e.start_after_open_seq, c.open_seq "
                    "FROM anytime_epochs AS e JOIN canary_cycles AS c "
                    "ON c.cycle_id = e.start_after_cycle"
                )
                .fetchone()
            )
            assert boundary[0] == boundary[1]
            migrated.append_many([record for record in records if record.cycle_id == "cycle-0004"])
            result = run_anytime_check(migrated)
            assert result.processed_cycles == ("cycle-0004",)

    def test_concurrent_golden_change_uses_the_snapshot_retry_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = project(tmp_path)
        original = anytime_module.get_golden_baseline
        calls = 0

        def racing_read(target: Store) -> list[str]:
            nonlocal calls
            calls += 1
            if calls == 2:
                (target.project_dir / "baseline.json").write_text(
                    '{"golden_cycles":["cycle-0000","cycle-0001"]}',
                    encoding="utf-8",
                )
            return original(target)

        monkeypatch.setattr(anytime_module, "get_golden_baseline", racing_read)
        result = run_anytime_check(store)
        store.close()

        assert calls >= 4
        assert result.processed_cycles == ()
        assert result.resets

    def test_stale_snapshot_cannot_mix_with_newer_process_state(self, tmp_path: Path) -> None:
        store = project(tmp_path, cycles=5)
        records, offset, revision = store.read_finalized_canary_snapshot_with_revision()
        all_records = make_records(cycles=6)
        store.append_many([r for r in all_records if r.cycle_id == "cycle-0005"])
        before = store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0]

        with pytest.raises(FinalizedSnapshotChangedError, match="snapshot changed"):
            anytime_module._run_anytime_check_snapshot(
                store,
                ProjectConfig.load(store.project_dir),
                records,
                offset,
                revision,
            )

        after = store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0]
        assert after == before
        assert load_states(store) == {}
        store.close()

    def test_epoch_resets_when_the_golden_baseline_changes(self, tmp_path: Path) -> None:
        """A different baseline is a different null; evidence cannot carry over."""
        store = project(tmp_path)
        run_anytime_check(store)
        run_anytime_check(store)
        before = load_states(store)
        assert all(s.epoch == 0 for s in before.values())

        cycles = sorted({r.cycle_id for r in store.read_records() if r.cycle_id})
        set_golden_baseline(store, cycles[:2])  # redefine the golden era
        res = run_anytime_check(store)

        after = load_states(store)
        assert any(s.epoch == 1 for s in after.values()), "no process reset on a new baseline"
        assert res.resets, "reset happened but was not reported"
        assert "historical cycles" in res.resets[0]
        assert res.processed_cycles == ()
        assert res.verdict == "NO CURRENT OBSERVATION"
        golden_states = [s for k, s in after.items() if k[0] == "golden"]
        assert all(s.cycles == 0 for s in golden_states), "historical cycles were replayed"
        store.close()

    def test_wealth_table_orders_by_evidence(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        res = run_anytime_check(store)
        table = wealth_table(res)
        assert list(table["log_wealth"]) == sorted(table["log_wealth"], reverse=True)
        store.close()


class TestProductionStateGuards:
    def test_degraded_cycle_is_evaluated_without_outcome_dependent_skipping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = make_records(cycles=4)
        current = records[-1].cycle_id
        degraded = [
            record.model_copy(update={"errors": ["forced collection failure"]})
            if record.cycle_id == current
            else record
            for record in records
        ]
        store = store_records(tmp_path, degraded)

        monkeypatch.setattr(
            anytime_module,
            "rate_evalue",
            lambda *_args, **_kwargs: EValueOutcome(0.25, True, "scripted evidence"),
        )

        result = run_anytime_check(store)
        states = load_states(store)
        store.close()

        assert result.verdict == "DEGRADED DATA"
        assert result.processed_cycles == (current,)
        assert all(state.cycles == 1 for state in states.values())
        assert all(state.log_wealth == pytest.approx(0.25) for state in states.values())
        assert all(state.bets_placed == 1 for state in states.values())
        assert all(state.prior.n_cycles == 1 for state in states.values())
        assert all(state.prior.trials > 0 for state in states.values())

    def test_degraded_cycle_does_not_hide_an_earlier_persistent_crossing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A later degraded loss cannot erase a prior valid alert."""
        records = make_records(cycles=5)
        current = records[-1].cycle_id
        degraded = [
            record.model_copy(update={"errors": ["forced collection failure"]})
            if record.cycle_id == current
            else record
            for record in records
        ]

        def crossing_evalue(
            successes: int,
            trials: int,
            prior: PriorState,
            **_kwargs: object,
        ) -> EValueOutcome:
            del successes, trials
            return EValueOutcome(
                8.0 if prior.n_cycles == 0 else -8.0,
                True,
                "scripted crossing then loss",
            )

        monkeypatch.setattr(anytime_module, "rate_evalue", crossing_evalue)
        store = store_records(tmp_path, degraded)
        result = run_anytime_check(store)
        store.close()

        assert result.degraded
        assert result.n_alerts > 0
        assert result.verdict == "DRIFT DETECTED"
        assert all(process.crossed_at == 1 for process in result.alerts())

    def test_degraded_backlog_cycle_is_not_hidden_by_a_healthy_latest_cycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = make_records(cycles=5)
        degraded_cycle = "cycle-0003"
        with_errors = [
            record.model_copy(update={"errors": ["forced collection failure"]})
            if record.cycle_id == degraded_cycle
            else record
            for record in records
        ]
        store = store_records(tmp_path, with_errors)
        monkeypatch.setattr(
            anytime_module,
            "rate_evalue",
            lambda *_args, **_kwargs: EValueOutcome(0.0, True, "neutral scripted evidence"),
        )

        result = run_anytime_check(store)
        repeated = run_anytime_check(store)
        repeated_report = render_anytime_report(repeated)
        store.close()

        assert result.current_cycle == "cycle-0004"
        assert result.processed_cycles == ("cycle-0003", "cycle-0004")
        assert result.degraded_cycles == (degraded_cycle,)
        assert result.degraded
        assert result.verdict == "DEGRADED DATA"
        assert repeated.processed_cycles == ()
        assert repeated.degraded_cycles == (degraded_cycle,)
        assert repeated.verdict == "DEGRADED DATA"
        assert degraded_cycle in repeated_report

    def test_degraded_epoch_history_survives_a_later_healthy_cycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = make_records(cycles=5)
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        degraded_cycle = "cycle-0003"
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))
        assert run_anytime_check(store).processed_cycles == ()
        monkeypatch.setattr(
            anytime_module,
            "rate_evalue",
            lambda *_args, **_kwargs: EValueOutcome(0.0, True, "neutral scripted evidence"),
        )

        degraded_records = [
            record.model_copy(update={"errors": ["forced collection failure"]})
            for record in records
            if record.cycle_id == degraded_cycle
        ]
        store.append_many(degraded_records)
        first = run_anytime_check(store)
        store.append_many([record for record in records if record.cycle_id == "cycle-0004"])
        second = run_anytime_check(store)
        repeated = run_anytime_check(store)
        report = render_anytime_report(repeated)
        store.close()

        assert first.processed_cycles == (degraded_cycle,)
        assert first.degraded_cycles == (degraded_cycle,)
        assert first.verdict == "DEGRADED DATA"
        assert second.processed_cycles == ("cycle-0004",)
        assert second.degraded_cycles == (degraded_cycle,)
        assert second.degraded
        assert second.verdict == "DEGRADED DATA"
        assert repeated.processed_cycles == ()
        assert repeated.degraded_cycles == (degraded_cycle,)
        assert repeated.verdict == "DEGRADED DATA"
        assert degraded_cycle in report

    def test_late_finalized_excluded_cycle_cannot_hide_eligible_suite_change(
        self, tmp_path: Path
    ) -> None:
        records = [
            stamp_suite(
                record,
                "sha256:suite-b" if record.cycle_id == "cycle-0004" else "sha256:suite-a",
            )
            for record in make_records(cycles=5)
        ]
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))
        store.append_many(
            [record for record in records if record.cycle_id == "cycle-0003"],
            finalize_cycles=False,
        )
        assert run_anytime_check(store).processed_cycles == ()
        store.append_many([record for record in records if record.cycle_id == "cycle-0004"])
        store.finalize_cycle("cycle-0003")
        before_states = load_states(store)
        before_ledger = (
            store.connect().execute("SELECT COUNT(*) FROM anytime_processed_cycles").fetchone()[0]
        )
        before_checks = store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0]

        with pytest.raises(ValueError, match="backlog spans a canary-suite boundary") as exc:
            run_anytime_check(store)

        assert "cycle-0004" in str(exc.value)
        assert load_states(store) == before_states
        assert (
            store.connect().execute("SELECT COUNT(*) FROM anytime_processed_cycles").fetchone()[0]
            == before_ledger
        )
        assert store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0] == before_checks
        store.close()

    def test_excluded_suite_change_cannot_block_eligible_golden_suite_cycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = [
            stamp_suite(
                record,
                "sha256:suite-b" if record.cycle_id == "cycle-0003" else "sha256:suite-a",
            )
            for record in make_records(cycles=5)
        ]
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))
        store.append_many(
            [record for record in records if record.cycle_id == "cycle-0003"],
            finalize_cycles=False,
        )
        declaration = run_anytime_check(store)
        store.append_many([record for record in records if record.cycle_id == "cycle-0004"])
        store.finalize_cycle("cycle-0003")
        monkeypatch.setattr(
            anytime_module,
            "rate_evalue",
            lambda *_args, **_kwargs: EValueOutcome(0.0, True, "neutral scripted evidence"),
        )

        result = run_anytime_check(store)
        store.close()

        assert result.fingerprint == declaration.fingerprint
        assert result.current_cycle == "cycle-0003"
        assert result.processed_cycles == ("cycle-0004",)
        assert result.coverage_status == "NO OBSERVATION"
        assert result.verdict == "NO CURRENT OBSERVATION"

    def test_excluded_suite_families_do_not_contaminate_epoch_composition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = make_records(cycles=5)
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        selected = [
            stamp_suite(
                record,
                "sha256:suite-b" if record.cycle_id == "cycle-0003" else "sha256:suite-a",
            )
            for record in records
            if (
                record.cycle_id == "cycle-0003"
                and record.input.metadata.get("family") == "edge_case"
            )
            or (
                record.cycle_id != "cycle-0003"
                and record.input.metadata.get("family") == "happy_path"
            )
        ]
        store = Store.init_project(tmp_path)
        store.append_many([record for record in selected if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))
        store.append_many(
            [record for record in selected if record.cycle_id == "cycle-0003"],
            finalize_cycles=False,
        )
        assert run_anytime_check(store).processed_cycles == ()
        store.append_many([record for record in selected if record.cycle_id == "cycle-0004"])
        store.finalize_cycle("cycle-0003")
        monkeypatch.setattr(
            anytime_module,
            "rate_evalue",
            lambda *_args, **_kwargs: EValueOutcome(0.0, True, "neutral scripted evidence"),
        )

        result = run_anytime_check(store)
        store.close()

        assert result.current_cycle == "cycle-0003"
        assert result.processed_cycles == ("cycle-0004",)
        assert result.coverage_status == "NO OBSERVATION"
        assert result.verdict == "NO CURRENT OBSERVATION"
        assert all(process.key[1] == "happy_path" for process in result.processes)

    @pytest.mark.parametrize(
        ("log_e", "expected_verdict", "expect_alert"),
        [(0.0, "NO CURRENT OBSERVATION", False), (8.0, "DRIFT DETECTED", True)],
    )
    def test_late_finalized_excluded_current_cycle_is_not_green(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        log_e: float,
        expected_verdict: str,
        expect_alert: bool,
    ) -> None:
        records = [stamp_suite(record, "sha256:suite-a") for record in make_records(cycles=5)]
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in golden_ids])
        set_golden_baseline(store, sorted(golden_ids))
        store.append_many(
            [record for record in records if record.cycle_id == "cycle-0003"],
            finalize_cycles=False,
        )
        assert run_anytime_check(store).processed_cycles == ()
        store.append_many([record for record in records if record.cycle_id == "cycle-0004"])
        store.finalize_cycle("cycle-0003")
        monkeypatch.setattr(
            anytime_module,
            "rate_evalue",
            lambda *_args, **_kwargs: EValueOutcome(log_e, True, "scripted evidence"),
        )

        result = run_anytime_check(store)
        states = load_states(store)
        store.close()

        assert result.current_cycle == "cycle-0003"
        assert result.processed_cycles == ("cycle-0004",)
        assert result.coverage_status == "NO OBSERVATION"
        assert result.verdict == expected_verdict
        assert bool(result.n_alerts) is expect_alert
        assert all(state.cycles == 1 for state in states.values())

    def test_composition_mismatch_fails_closed_without_state_mutation(self, tmp_path: Path) -> None:
        records = [stamp_suite(record, "sha256:suite-a") for record in make_records(cycles=4)]
        current = records[-1].cycle_id
        bad_family = "happy_path"
        incomplete = [
            record
            for record in records
            if not (
                record.cycle_id == current and record.input.metadata.get("family") == bad_family
            )
        ]
        store = store_records(tmp_path, incomplete)
        before_states = load_states(store)
        before_checks = store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0]

        with pytest.raises(ValueError, match="Refusing to choose E_t=1"):
            run_anytime_check(store)

        assert load_states(store) == before_states
        assert (
            store.connect().execute("SELECT COUNT(*) FROM anytime_processed_cycles").fetchone()[0]
            == 0
        )
        assert store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0] == before_checks
        store.close()

    def test_mixed_suite_backlog_is_rejected_without_state_mutation(self, tmp_path: Path) -> None:
        records = make_records(cycles=6)
        changed = [
            stamp_suite(
                record,
                "sha256:suite-b" if record.cycle_id == "cycle-0004" else "sha256:suite-a",
            )
            for record in records
        ]
        store = store_records(tmp_path, changed)
        before_states = load_states(store)
        before_checks = store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0]

        with pytest.raises(ValueError, match="backlog spans a canary-suite boundary"):
            run_anytime_check(store)

        assert load_states(store) == before_states
        assert (
            store.connect().execute("SELECT COUNT(*) FROM anytime_processed_cycles").fetchone()[0]
            == 0
        )
        assert store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0] == before_checks
        store.close()

    def test_partially_stamped_cycle_is_rejected_without_state_mutation(
        self, tmp_path: Path
    ) -> None:
        records = make_records(cycles=4)
        golden_ids = {"cycle-0000", "cycle-0001", "cycle-0002"}
        golden = [
            stamp_suite(record, "sha256:suite-a")
            for record in records
            if record.cycle_id in golden_ids
        ]
        current = [record for record in records if record.cycle_id == "cycle-0003"]
        partially_stamped = [
            record if index == 0 else stamp_suite(record, "sha256:suite-a")
            for index, record in enumerate(current)
        ]
        store = Store.init_project(tmp_path)
        store.append_many(golden)
        set_golden_baseline(store, sorted(golden_ids))
        assert run_anytime_check(store).processed_cycles == ()
        store.append_many(partially_stamped)
        before_states = load_states(store)
        before_checks = store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0]

        with pytest.raises(ValueError, match="mixes records with and without"):
            run_anytime_check(store)

        assert load_states(store) == before_states
        assert (
            store.connect().execute("SELECT COUNT(*) FROM anytime_processed_cycles").fetchone()[0]
            == 0
        )
        assert store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0] == before_checks
        store.close()

    def test_actual_pinned_embedder_not_config_text_defines_epoch(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        first = run_anytime_check(store, ProjectConfig(embedder="ignored-a"))
        second = run_anytime_check(store, ProjectConfig(embedder="ignored-b"))
        assert second.fingerprint == first.fingerprint
        assert second.processed_cycles == ()

        pin_embedder(store, "hash")
        third = run_anytime_check(store, ProjectConfig(embedder="still-ignored"))
        assert third.fingerprint != first.fingerprint
        assert third.processed_cycles == ()
        assert third.verdict == "NO CURRENT OBSERVATION"
        assert all(state.epoch == 1 for state in load_states(store).values())
        store.close()

    def test_production_records_with_cycle_ids_do_not_enter_the_experiment(
        self, tmp_path: Path
    ) -> None:
        store = project(tmp_path)
        template = store.read_records()[0]
        production = template.model_copy(
            update={
                "id": str(uuid4()),
                "source": Source.PRODUCTION,
                "cycle_id": "production-cycle",
            }
        )
        store.append(production)

        result = run_anytime_check(store)
        assert result.current_cycle == "cycle-0005"
        assert "production-cycle" not in result.processed_cycles
        store.close()

    def test_state_ledger_pool_and_check_roll_back_together(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = project(tmp_path)

        def fail_persist(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated check-row failure")

        monkeypatch.setattr(anytime_module, "_persist", fail_persist)
        with pytest.raises(RuntimeError, match="simulated check-row failure"):
            run_anytime_check(store)

        conn = store.connect()
        assert load_states(store) == {}
        assert conn.execute("SELECT COUNT(*) FROM anytime_processed_cycles").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM epoch_pool").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM checks").fetchone()[0] == 0
        store.close()


class TestBothPathsOnIdenticalLogs:
    """The comparison must be reproducible from the tool, not a study script."""

    def test_stable_agent_is_quiet_on_both_paths(self, tmp_path: Path) -> None:
        store = project(tmp_path, change_cycle=None)
        fixed = run_check(store, config=ProjectConfig(permutations=200))
        declaration = run_anytime_check(store)
        assert declaration.verdict == "NO CURRENT OBSERVATION"
        future = make_records(cycles=7, change_cycle=None)
        store.append_many([record for record in future if record.cycle_id == "cycle-0006"])
        anytime = run_anytime_check(store)
        assert fixed.n_alerts == 0
        assert anytime.n_alerts == 0
        assert anytime.verdict == "OK"
        store.close()

    def test_anytime_needs_evidence_to_accumulate(self, tmp_path: Path) -> None:
        """One cycle after a swap is not enough, and that is the design.

        The fixed-sample path can fire on a single cycle; an e-process has to
        accumulate past log(1/alpha'). Slower on purpose — the trade bought
        is a guarantee that survives repeated looking.
        """
        records = make_records(cycles=8, change_cycle=7)
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id != "cycle-0007"])
        set_golden_baseline(store, ["cycle-0000", "cycle-0001", "cycle-0002"])
        declaration = run_anytime_check(store)
        assert declaration.processed_cycles == ()
        store.append_many([record for record in records if record.cycle_id == "cycle-0007"])
        anytime = run_anytime_check(store)
        assert anytime.verdict in {"OK", "DRIFT DETECTED"}
        assert all(p.cycles == 1 for p in anytime.processes)
        store.close()


class TestCli:
    def test_inference_flag_selects_the_path(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        store.close()
        out = runner.invoke(app, ["check", "--project", str(tmp_path), "--inference", "anytime"])
        assert out.exit_code == 3, out.output
        assert "Anytime-valid: alpha=" in out.output
        assert "e-processes" in out.output
        assert "PER EPOCH" in out.output
        assert "historical cycles were not replayed" in out.output

    def test_default_is_the_fixed_path(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        store.close()
        out = runner.invoke(app, ["check", "--project", str(tmp_path)])
        assert out.exit_code in (0, 2), out.output
        assert "observed-effect gated" in out.output  # the fixed path's wording

    def test_unknown_mode_is_rejected(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        store.close()
        out = runner.invoke(app, ["check", "--project", str(tmp_path), "--inference", "bayes"])
        assert out.exit_code == 1
        assert "must be 'fixed' or 'anytime'" in out.output


class TestAnytimeReport:
    """The report has to state the guarantee exactly, caveats included."""

    def test_report_states_the_budget_and_the_per_epoch_caveat(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        for _ in range(3):
            res = run_anytime_check(store)
        md = render_anytime_report(res)
        store.close()

        assert "per epoch" in md
        assert "unbounded" in md
        # the budget decomposition, and *why* gamma is divided
        assert "γ per process" in md
        assert "union-bound" in md
        assert f"÷ {res.n_processes}" in md
        # no float noise in the headline numbers
        assert "0.030000" not in md
        # honesty sections that must never be dropped
        assert "What is proven, and what is measured" in md
        assert "Multiplicity spent, and on what" in md
        assert "not** affirmative evidence" in md

    def test_report_is_deterministic_given_state(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        res = run_anytime_check(store)
        first = render_anytime_report(res)
        second = render_anytime_report(res)
        store.close()
        assert first == second

    def test_reset_is_announced_in_the_report(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        run_anytime_check(store)
        cycles = sorted({r.cycle_id for r in store.read_records() if r.cycle_id})
        set_golden_baseline(store, cycles[:2])
        res = run_anytime_check(store)
        md = render_anytime_report(res)
        store.close()
        assert "Epoch resets at this check" in md
        assert "returned to zero" in md

    def test_cli_report_honours_the_inference_flag(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        store.close()
        out = runner.invoke(app, ["report", "--project", str(tmp_path), "--inference", "anytime"])
        assert out.exit_code == 3, out.output
        assert "anytime-valid" in out.output
        assert "per epoch" in out.output


class TestEpochPool:
    """Membership is decided once per epoch — the property the guarantee needs."""

    def test_signatures_without_reference_data_are_excluded(self, tmp_path: Path) -> None:
        """A process that cannot produce evidence must not consume budget.

        The simulator declares no expected answers, so ``exact_match`` is
        entirely absent. Including it would shrink every other process's
        coverage and crossing budgets for nothing.
        """
        store = project(tmp_path)
        res = run_anytime_check(store)
        pool = load_pool(store, res.fingerprint)
        store.close()

        signatures = {k[2] for k in pool}
        assert "exact_match" not in signatures, "a dead signature is charging the budget"
        assert signatures, "pool is empty"
        assert len(pool) == res.n_processes

    def test_pool_is_declared_once_and_then_frozen(self, tmp_path: Path) -> None:
        """Only the epoch's first check declares; later ones reuse.

        This is what keeps the frozen-baseline coverage event a *single* event:
        a pool recomputed per cycle would vary the per-process budget, hence
        the interval, hence what the coverage guarantee refers to.
        """
        store = project(tmp_path)
        first = run_anytime_check(store)
        second = run_anytime_check(store)
        third = run_anytime_check(store)
        store.close()

        assert first.pool_declared_now is True
        assert second.pool_declared_now is False
        assert third.pool_declared_now is False
        assert first.n_processes == second.n_processes == third.n_processes
        assert first.gamma_per_process == pytest.approx(second.gamma_per_process)

    def test_pool_size_is_stable_even_as_data_accumulates(self, tmp_path: Path) -> None:
        """New cycles must not move the budget mid-epoch."""
        store = project(tmp_path)
        first = run_anytime_check(store)
        # add another cycle's worth of records under the same epoch
        cfg = SimConfig(n_canaries=18, repetitions=7, change_cycle=None, seed=99)
        added = [
            record.model_copy(update={"id": str(uuid4()), "cycle_id": "cycle-0006"})
            for record in SimAgent(cfg).run_cycles(1)
        ]
        store.append_many(added)
        later = run_anytime_check(store)
        store.close()
        assert later.n_processes == first.n_processes
        assert later.gamma_per_process == pytest.approx(first.gamma_per_process)

    def test_new_epoch_declares_a_fresh_pool(self, tmp_path: Path) -> None:
        """A changed hypothesis gets a new pool as well as new wealth."""
        store = project(tmp_path)
        first = run_anytime_check(store)
        cycles = sorted({r.cycle_id for r in store.read_records() if r.cycle_id})
        set_golden_baseline(store, cycles[:2])
        second = run_anytime_check(store)
        store.close()

        assert second.fingerprint != first.fingerprint
        assert second.pool_declared_now is True, "new epoch reused the old pool"
        assert load_pool(store, second.fingerprint)

    def test_report_explains_when_membership_was_fixed(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        first = render_anytime_report(run_anytime_check(store))
        later = render_anytime_report(run_anytime_check(store))
        store.close()
        assert "declared at this check" in first
        assert "frozen when this epoch began" in later
        assert "Multiplicity spent, and on what" in later


class TestProductionRateModel:
    """The production rate process is valid first and conservative by design."""

    def test_frozen_cp_detects_strong_rate_shift(self, tmp_path: Path) -> None:
        cfg = SimConfig(
            n_canaries=18,
            repetitions=7,
            pre=BehaviorProfile(refusal_prob=0.05),
            post=BehaviorProfile(refusal_prob=0.55),
            change_cycle=8,
            seed=31,
        )
        store = Store.init_project(tmp_path)
        records = SimAgent(cfg).run_cycles(16)
        ids = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        store.append_many([record for record in records if record.cycle_id in set(ids[:8])])
        set_golden_baseline(store, ids[:3])
        declaration = run_anytime_check(store)
        assert declaration.processed_cycles == ()
        store.append_many([record for record in records if record.cycle_id in set(ids[8:])])
        result = run_anytime_check(store)
        assert result.n_alerts > 0
        refusal = [p for p in result.processes if p.key[2] == "refusal"]
        assert any(p.rejected for p in refusal)
        store.close()

    def test_frozen_cp_stays_quiet_on_stable(self, tmp_path: Path) -> None:
        records = make_records(cycles=10, change_cycle=None)
        initial_ids = {f"cycle-{index:04d}" for index in range(6)}
        store = Store.init_project(tmp_path)
        store.append_many([record for record in records if record.cycle_id in initial_ids])
        set_golden_baseline(store, ["cycle-0000", "cycle-0001", "cycle-0002"])
        declaration = run_anytime_check(store)
        assert declaration.processed_cycles == ()
        store.append_many([record for record in records if record.cycle_id not in initial_ids])
        result = run_anytime_check(store)
        assert result.n_alerts == 0
        assert result.processed_cycles == (
            "cycle-0006",
            "cycle-0007",
            "cycle-0008",
            "cycle-0009",
        )
        assert all(state.cycles == 4 for state in load_states(store).values())
        store.close()

    def test_frozen_cp_is_the_default_and_spends_coverage_budget(self, tmp_path: Path) -> None:
        store = project(tmp_path, cycles=6, change_cycle=None)
        cfg = ProjectConfig.load(store.project_dir)
        explicit = ProjectConfig(
            name=cfg.name,
            anytime=AnytimeConfig(rate_model="frozen_cp"),
        )
        assert cfg.anytime.rate_model == "frozen_cp"
        declaration = run_anytime_check(store, config=explicit)
        assert declaration.processed_cycles == ()
        future = make_records(cycles=7, change_cycle=None)
        store.append_many([record for record in future if record.cycle_id == "cycle-0006"])
        result = run_anytime_check(store, config=explicit)
        assert result.n_alerts == 0
        assert result.gamma_per_process > 0
        assert result.alpha_prime < result.alpha
        assert all("worst case over p" in p.detail for p in result.processes)
        store.close()

    def test_withdrawn_twosample_configuration_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="withdrawn"):
            AnytimeConfig(rate_model="twosample")

    def test_batched_backlog_preserves_a_transient_crossing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A later losing cycle cannot erase an earlier anytime alert."""
        records = make_records(cycles=5)
        cycle_ids = list(dict.fromkeys(r.cycle_id for r in records if r.cycle_id is not None))

        def scripted_evalue(
            successes: int,
            trials: int,
            prior: PriorState,
            **_kwargs: object,
        ) -> EValueOutcome:
            del successes, trials
            return EValueOutcome(8.0 if prior.n_cycles == 0 else -8.0, True, "scripted")

        monkeypatch.setattr(anytime_module, "rate_evalue", scripted_evalue)

        initial = [record for record in records if record.cycle_id in set(cycle_ids[:3])]
        future = [record for record in records if record.cycle_id in set(cycle_ids[3:])]

        batched = Store.init_project(tmp_path / "batched")
        batched.append_many(initial)
        set_golden_baseline(batched, cycle_ids[:3])
        run_anytime_check(batched)
        batched.append_many(future)
        batched_result = run_anytime_check(batched)

        incremental = Store.init_project(tmp_path / "incremental")
        incremental.append_many(initial)
        set_golden_baseline(incremental, cycle_ids[:3])
        run_anytime_check(incremental)
        incremental.append_many([r for r in records if r.cycle_id == cycle_ids[3]])
        first = run_anytime_check(incremental)
        incremental.append_many([r for r in records if r.cycle_id == cycle_ids[4]])
        second = run_anytime_check(incremental)

        assert first.n_alerts > 0
        assert batched_result.n_alerts == second.n_alerts == first.n_alerts
        assert all(p.crossed_at == 1 for p in batched_result.alerts())
        assert all(p.log_wealth == pytest.approx(0.0) for p in batched_result.alerts())
        batched.close()
        incremental.close()
