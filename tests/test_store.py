"""Store round-trip, config-event detection, and index-rebuild tests."""

from __future__ import annotations

import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from dedrift.schema import InteractionRecord
from dedrift.store import Store
from tests.test_schema import make_config, make_record


class TestInit:
    def test_init_creates_structure(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            assert store.exists()
            assert store.config_path.exists()
            assert store.records_path.exists()
            assert store.index_path.exists()

    def test_init_idempotent(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as first:
            first.config_path.write_text("# customized", encoding="utf-8")
        with Store.init_project(tmp_path) as second:
            assert second.config_path.read_text(encoding="utf-8") == "# customized"

    def test_managed_paths_are_owner_only(self, tmp_path: Path) -> None:
        if os.name != "posix":
            return
        with Store.init_project(tmp_path) as store:
            assert stat.S_IMODE(store.project_dir.stat().st_mode) == 0o700
            assert stat.S_IMODE(store.records_path.parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(store.config_path.stat().st_mode) == 0o600
            assert stat.S_IMODE(store.records_path.stat().st_mode) == 0o600
            assert stat.S_IMODE(store.index_path.stat().st_mode) == 0o600

    def test_connection_uses_durable_concurrency_pragmas(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            conn = store.connect()
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000

    def test_anytime_processed_cycle_ledger_is_migrated(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            row = (
                store.connect()
                .execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'anytime_processed_cycles'"
                )
                .fetchone()
            )
            assert row == ("anytime_processed_cycles",)


class TestRoundTrip:
    def test_append_and_read(self, tmp_path: Path) -> None:
        records = [make_record(canary_id=f"canary-{i:03d}") for i in range(5)]
        with Store.init_project(tmp_path) as store:
            for r in records:
                store.append(r)
            assert store.read_records() == records
            assert store.count_records() == 5

    def test_append_many(self, tmp_path: Path) -> None:
        records = [make_record() for _ in range(10)]
        with Store.init_project(tmp_path) as store:
            store.append_many(records)
            assert store.count_records() == 10
            assert store.read_records() == records

    def test_duplicate_ids_are_idempotent_within_batch_and_on_retry(self, tmp_path: Path) -> None:
        record = make_record()
        with Store.init_project(tmp_path) as store:
            store.append_many([record, record])
            store.append_many([record])
            assert store.count_records() == 1
            assert store.read_records() == [record]

    def test_conflicting_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        first = make_record()
        conflicting = first.model_copy(update={"canary_id": "different"})
        with Store.init_project(tmp_path) as store:
            with pytest.raises(ValueError, match="conflicting records"):
                store.append_many([first, conflicting])
            assert store.count_records() == 0
            assert store.read_records() == []

    def test_conflicting_existing_id_is_rejected(self, tmp_path: Path) -> None:
        first = make_record()
        conflicting = first.model_copy(update={"canary_id": "different"})
        with Store.init_project(tmp_path) as store:
            store.append(first)
            with pytest.raises(ValueError, match="already exists with a different payload"):
                store.append(conflicting)
            assert store.count_records() == 1
            assert store.read_records() == [first]

    def test_streamed_cycle_is_allowed_until_anytime_processing(self, tmp_path: Path) -> None:
        first = make_record(cycle_id="cycle-0001")
        second = make_record(cycle_id="cycle-0001")
        third = make_record(cycle_id="cycle-0001")
        with Store.init_project(tmp_path) as store:
            store.append(first)
            store.append(first)  # exact retry remains idempotent
            store.append(second)
            store.connect().execute(
                "INSERT INTO anytime_processed_cycles "
                "(fingerprint, cycle_id, processed_ts) VALUES ('epoch', 'cycle-0001', 'now')"
            )
            store.connect().commit()
            with pytest.raises(ValueError, match="already processed by anytime inference"):
                store.append(third)
            assert store.read_records() == [first, second]

    def test_processed_cycle_exact_retry_remains_idempotent(self, tmp_path: Path) -> None:
        first = make_record(cycle_id="cycle-0001")
        with Store.init_project(tmp_path) as store:
            store.append(first)
            store.connect().execute(
                "INSERT INTO anytime_processed_cycles "
                "(fingerprint, cycle_id, processed_ts) VALUES ('epoch', 'cycle-0001', 'now')"
            )
            store.connect().commit()
            store.append(first)
            assert store.read_records() == [first]

    def test_new_id_in_processed_cycle_is_rejected(self, tmp_path: Path) -> None:
        first = make_record(cycle_id="cycle-0001")
        second = make_record(cycle_id="cycle-0001")
        with Store.init_project(tmp_path) as store:
            store.append(first)
            store.connect().execute(
                "INSERT INTO anytime_processed_cycles "
                "(fingerprint, cycle_id, processed_ts) VALUES ('epoch', 'cycle-0001', 'now')"
            )
            store.connect().commit()
            with pytest.raises(ValueError, match="already processed by anytime inference"):
                store.append(second)
            assert store.read_records() == [first]

    def test_first_canary_cycle_batch_accepts_multiple_ids(self, tmp_path: Path) -> None:
        records = [make_record(cycle_id="cycle-0001") for _ in range(3)]
        with Store.init_project(tmp_path) as store:
            store.append_many(records)
            assert store.read_records() == records

    def test_retry_reconciles_fsynced_jsonl_after_index_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = [make_record(), make_record()]

        def fail_index(conn: object, record: object) -> None:
            del conn, record
            raise RuntimeError("index failed")

        with Store.init_project(tmp_path) as store:
            with monkeypatch.context() as patch:
                patch.setattr(store, "_index_record", fail_index)
                with pytest.raises(RuntimeError, match="index failed"):
                    store.append_many(records)

            assert store.count_records() == 0
            assert len(store.read_records()) == 2
            store.append_many(records)
            assert store.count_records() == 2
            assert store.read_records() == records

    def test_append_many_flushes_and_fsyncs_before_index_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        original_fsync = os.fsync

        def tracking_fsync(descriptor: int) -> None:
            calls.append(descriptor)
            original_fsync(descriptor)

        with Store.init_project(tmp_path) as store:
            calls.clear()  # ignore durable creation of the initial config
            monkeypatch.setattr(os, "fsync", tracking_fsync)
            store.append_many([make_record(), make_record()])
            assert calls
            assert store.count_records() == 2

    def test_concurrent_store_instances_serialize_batches(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path):
            pass

        batches = [
            [
                make_record(
                    canary_id=f"writer-{writer}-{index}",
                    cycle_id=f"cycle-writer-{writer}",
                )
                for index in range(20)
            ]
            for writer in range(2)
        ]

        def write_batch(records: list[InteractionRecord]) -> None:
            with Store(tmp_path) as store:
                store.append_many(records)

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(write_batch, batches))

        with Store(tmp_path) as store:
            assert store.count_records() == 40
            assert len(store.read_records()) == 40

    def test_reader_waits_for_complete_writer_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with Store.init_project(tmp_path):
            pass
        records = [make_record() for _ in range(20)]
        fsync_started = Event()
        release_fsync = Event()
        reader_started = Event()
        original_fsync = os.fsync

        def blocking_fsync(descriptor: int) -> None:
            fsync_started.set()
            if not release_fsync.wait(timeout=5):
                raise TimeoutError("test did not release fsync")
            original_fsync(descriptor)

        def write() -> None:
            with Store(tmp_path) as store:
                store.append_many(records)

        def read() -> list[InteractionRecord]:
            reader_started.set()
            with Store(tmp_path) as store:
                return store.read_records()

        monkeypatch.setattr(os, "fsync", blocking_fsync)
        with ThreadPoolExecutor(max_workers=2) as executor:
            writer = executor.submit(write)
            assert fsync_started.wait(timeout=5)
            reader = executor.submit(read)
            assert reader_started.wait(timeout=5)
            assert not reader.done()
            release_fsync.set()
            writer.result(timeout=5)
            assert reader.result(timeout=5) == records

    def test_read_rejects_incomplete_trailing_record(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            with store.records_path.open("ab") as stream:
                stream.write(b'{"incomplete":')
                stream.flush()
                os.fsync(stream.fileno())
            with pytest.raises(RuntimeError, match="incomplete trailing record"):
                store.read_records()


class TestConfigEvents:
    def test_first_record_creates_event(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            store.append(make_record())
            events = store.config_events()
            assert len(events) == 1
            assert events[0][1] is None  # no prior fingerprint

    def test_fingerprint_change_creates_event(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            store.append(make_record())
            store.append(make_record())  # same config: no new event
            store.append(make_record(config=make_config(model="provider/model@v2")))
            events = store.config_events()
            assert len(events) == 2
            assert events[1][1] == events[0][2]  # old = previous new

    def test_no_event_without_change(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            store.append_many([make_record() for _ in range(20)])
            assert len(store.config_events()) == 1


class TestRebuild:
    def test_rebuild_matches_original(self, tmp_path: Path) -> None:
        records = [make_record() for _ in range(5)] + [
            make_record(config=make_config(model="provider/model@v2")) for _ in range(5)
        ]
        with Store.init_project(tmp_path) as store:
            store.append_many(records)
            events_before = store.config_events()
            n = store.rebuild_index()
            assert n == 10
            assert store.config_events() == events_before
            assert store.count_records() == 10
