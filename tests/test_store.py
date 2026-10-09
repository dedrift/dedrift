"""Store round-trip, config-event detection, and index-rebuild tests."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock
from typing import BinaryIO

import pytest

import dedrift.store as store_module
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
            assert conn.execute("PRAGMA application_id").fetchone()[0] == 0x44445246
            assert conn.execute(
                "SELECT value FROM store_metadata WHERE key = 'store_identity'"
            ).fetchone() == ("dedrift-index-v1",)

    def test_zero_byte_authoritative_index_is_never_reconstructed(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            store.append_many([make_record(cycle_id="finalized")])
            index_path = store.index_path
        index_path.write_bytes(b"")

        with pytest.raises(FileNotFoundError, match="authoritative index"):
            Store(tmp_path)
        assert index_path.read_bytes() == b""

    def test_unrelated_sqlite_file_is_never_converted_to_an_index(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            store.append_many([make_record(cycle_id="finalized")])
            index_path = store.index_path
        index_path.unlink()
        with sqlite3.connect(index_path) as unrelated:
            unrelated.execute("CREATE TABLE unrelated (value TEXT)")
            unrelated.execute("INSERT INTO unrelated VALUES ('keep me')")

        with pytest.raises(RuntimeError, match="not a recognized dedrift index"), Store(tmp_path):
            pass
        with sqlite3.connect(index_path) as untouched:
            assert untouched.execute("SELECT value FROM unrelated").fetchone() == ("keep me",)

    def test_connected_store_is_explicitly_one_instance_per_thread(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store, ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(store.count_records)
            with pytest.raises(RuntimeError, match="one Store per worker thread"):
                future.result(timeout=5)

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

    @pytest.mark.parametrize("missing", ["config", "records", "index"])
    def test_require_project_requires_all_authoritative_files(
        self, tmp_path: Path, missing: str
    ) -> None:
        root = tmp_path / missing
        with Store.init_project(root) as initialized:
            missing_path = {
                "config": initialized.config_path,
                "records": initialized.records_path,
                "index": initialized.index_path,
            }[missing]
        missing_path.unlink()

        with pytest.raises(FileNotFoundError, match="authoritative index"):
            Store(root)

    def test_missing_authoritative_index_is_never_silently_recreated(self, tmp_path: Path) -> None:
        record = make_record(cycle_id="finalized")
        with Store.init_project(tmp_path) as store:
            store.append_many([record])
            index_path = store.index_path
        index_path.unlink()

        damaged = Store(tmp_path, require_project=False)
        assert not damaged.exists()
        with pytest.raises(FileNotFoundError, match="authoritative index"):
            Store(tmp_path)
        with pytest.raises(FileNotFoundError, match="refusing to recreate"):
            Store.init_project(tmp_path)
        assert not index_path.exists()

    def test_concurrent_initializers_publish_only_complete_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_writer = store_module._atomic_private_writer
        first_writer_entered = Event()
        release_first_writer = Event()
        calls = 0
        calls_lock = Lock()

        @contextmanager
        def delayed_writer(path: Path) -> Iterator[BinaryIO]:
            nonlocal calls
            with original_writer(path) as stream:
                with calls_lock:
                    calls += 1
                    call = calls
                if call == 1:
                    first_writer_entered.set()
                    if not release_first_writer.wait(timeout=5):
                        raise TimeoutError("test did not release the first initializer")
                yield stream

        monkeypatch.setattr(store_module, "_atomic_private_writer", delayed_writer)

        def initialize() -> str:
            with Store.init_project(tmp_path) as store:
                return store.config_path.read_text(encoding="utf-8")

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(initialize)
            assert first_writer_entered.wait(timeout=5)
            second = pool.submit(initialize)
            with pytest.raises(FutureTimeoutError):
                second.result(timeout=0.1)
            release_first_writer.set()
            assert first.result(timeout=5) == store_module.DEFAULT_CONFIG_TOML
            assert second.result(timeout=5) == store_module.DEFAULT_CONFIG_TOML

    def test_concurrent_legacy_migrations_are_writer_serialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with Store.init_project(tmp_path) as initialized:
            index_path = initialized.index_path
        with sqlite3.connect(index_path) as conn:
            conn.execute("DROP INDEX idx_canary_cycles_finalize_seq")
            conn.execute("ALTER TABLE canary_cycles DROP COLUMN finalize_seq")
            conn.execute("DELETE FROM store_metadata WHERE key = 'finalized_revision'")

        original = Store._migrate_schema
        first_entered = Event()
        release_first = Event()
        second_entered = Event()
        counter_lock = Lock()
        calls = 0

        def delayed_migration(store: Store, conn: sqlite3.Connection) -> None:
            nonlocal calls
            with counter_lock:
                calls += 1
                call = calls
            if call == 1:
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise TimeoutError("test did not release first migration")
            else:
                second_entered.set()
            original(store, conn)

        monkeypatch.setattr(Store, "_migrate_schema", delayed_migration)

        def open_store() -> None:
            with Store(tmp_path):
                pass

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(open_store)
            assert first_entered.wait(timeout=5)
            second = executor.submit(open_store)
            assert not second_entered.wait(timeout=0.2)
            release_first.set()
            first.result(timeout=5)
            second.result(timeout=5)

        with sqlite3.connect(index_path) as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(canary_cycles)")}
            assert "finalize_seq" in columns

    def test_failed_connect_closes_and_resets_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with Store.init_project(tmp_path):
            pass
        store = Store(tmp_path)
        opened: list[sqlite3.Connection] = []
        real_connect = sqlite3.connect

        def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        def fail_migration(_conn: sqlite3.Connection) -> None:
            raise RuntimeError("migration failed")

        with monkeypatch.context() as patch:
            patch.setattr("dedrift.store.sqlite3.connect", tracked_connect)
            patch.setattr(store, "_migrate_schema", fail_migration)
            with pytest.raises(RuntimeError, match="migration failed"):
                store.connect()

        assert store._conn is None
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            opened[0].execute("SELECT 1")
        with store:
            assert store.connect().execute("SELECT 1").fetchone() == (1,)


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

    def test_finalization_validation_fails_before_jsonl_is_touched(self, tmp_path: Path) -> None:
        """A logically rejected batch must not survive reconciliation."""
        first = make_record(cycle_id="bad-cycle", canary_id="same", repetition=1)
        duplicate_coordinate = make_record(cycle_id="bad-cycle", canary_id="same", repetition=1)
        with Store.init_project(tmp_path) as store:
            before = store.records_path.read_bytes()
            before_offset = store.records_path.stat().st_size

            with pytest.raises(ValueError, match=r"duplicate.*coordinates"):
                store.append_many([first, duplicate_coordinate])

            assert store.records_path.read_bytes() == before
            assert store.records_path.stat().st_size == before_offset
            assert store.count_records() == 0
            assert store.cycle_status("bad-cycle") is None

        with Store(tmp_path) as reopened:
            assert reopened.read_records() == []
            assert reopened.count_records() == 0
            assert reopened.cycle_status("bad-cycle") is None

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

    def test_recovered_crash_tail_keeps_earlier_cycle_observation_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = make_record(cycle_id="cycle-a")
        later = make_record(cycle_id="cycle-b")

        def fail_index(conn: object, record: object) -> None:
            del conn, record
            raise RuntimeError("index failed")

        with Store.init_project(tmp_path) as store:
            with monkeypatch.context() as patch:
                patch.setattr(store, "_index_record", fail_index)
                with pytest.raises(RuntimeError, match="index failed"):
                    store.append_many([first], finalize_cycles=False)

            store.append_many([later])
            rows = (
                store.connect()
                .execute(
                    "SELECT cycle_id, status, open_seq FROM canary_cycles "
                    "WHERE cycle_id IN ('cycle-a', 'cycle-b') ORDER BY open_seq"
                )
                .fetchall()
            )
            assert rows == [("cycle-a", "open", 1), ("cycle-b", "finalized", 2)]

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


class TestFinalizedSnapshots:
    def test_finalized_revision_reconstructs_exact_historical_snapshot(
        self, tmp_path: Path
    ) -> None:
        first = make_record(cycle_id="cycle-a")
        second = make_record(cycle_id="cycle-b")
        with Store.init_project(tmp_path) as store:
            store.append_many([first])
            store.append_many([second])

            assert store.read_finalized_canary_snapshot_at_revision(0) == []
            assert store.read_finalized_canary_snapshot_at_revision(1) == [first]
            assert store.read_finalized_canary_snapshot_at_revision(2) == [first, second]
            with pytest.raises(ValueError, match="newer than current"):
                store.read_finalized_canary_snapshot_at_revision(3)

    @pytest.mark.parametrize("missing_column", ["record_digest", "record_json"])
    def test_reconcile_backfills_only_a_missing_cache_field(
        self, tmp_path: Path, missing_column: str
    ) -> None:
        record = make_record(cycle_id="immutable")
        with Store.init_project(tmp_path) as store:
            store.append_many([record])
            store.connect().execute(
                f"UPDATE records SET {missing_column} = NULL WHERE id = ?",
                (record.id,),
            )
            store.connect().commit()

        with Store(tmp_path) as reopened:
            row = (
                reopened.connect()
                .execute(
                    "SELECT record_digest, record_json FROM records WHERE id = ?", (record.id,)
                )
                .fetchone()
            )
            assert row[0] == reopened._record_digest(record)
            assert row[1] == record.to_jsonl()

    @pytest.mark.parametrize("missing_column", ["record_digest", "record_json"])
    def test_reconcile_rejects_conflict_with_surviving_cache_field(
        self, tmp_path: Path, missing_column: str
    ) -> None:
        record = make_record(cycle_id="immutable")
        conflicting = record.model_copy(
            update={"output": record.output.model_copy(update={"text": "externally changed"})}
        )
        with Store.init_project(tmp_path) as store:
            store.append_many([record])
            store.connect().execute(
                f"UPDATE records SET {missing_column} = NULL WHERE id = ?",
                (record.id,),
            )
            store.connect().commit()
            records_path = store.records_path

        records_path.write_text(conflicting.to_jsonl() + "\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="conflict"), Store(tmp_path):
            pass

    def test_reconcile_rejects_external_extension_of_finalized_cycle(self, tmp_path: Path) -> None:
        first = make_record(cycle_id="immutable", canary_id="first")
        injected = make_record(cycle_id="immutable", canary_id="injected")
        with Store.init_project(tmp_path) as store:
            store.append_many([first])
            index_path = store.index_path
            records_path = store.records_path

        with records_path.open("a", encoding="utf-8") as stream:
            stream.write(injected.to_jsonl() + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        with pytest.raises(RuntimeError, match="externally extends finalized"), Store(tmp_path):
            pass

        with sqlite3.connect(index_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
            assert conn.execute(
                "SELECT finalized_records, finalize_seq FROM canary_cycles "
                "WHERE cycle_id = 'immutable'"
            ).fetchone() == (1, 1)

    def test_multi_cycle_batch_preserves_first_appearance_order(self, tmp_path: Path) -> None:
        """Opaque cycle IDs must not reorder the detector's chronology."""
        observed_first = make_record(cycle_id="z-observed-first")
        observed_second = make_record(cycle_id="a-observed-second")

        with Store.init_project(tmp_path) as store:
            store.append_many([observed_first, observed_second])
            records, _offset, revision = store.read_finalized_canary_snapshot_with_revision()

            assert records == [observed_first, observed_second]
            assert store.finalized_cycle_ids() == [
                "z-observed-first",
                "a-observed-second",
            ]
            assert revision == 2

    def test_late_finalized_cycle_is_ordered_by_finalize_sequence(self, tmp_path: Path) -> None:
        early_record = make_record(cycle_id="recorded-first")
        later_record = make_record(cycle_id="finalized-first")
        with Store.init_project(tmp_path) as store:
            store.append(early_record)
            store.append_many([later_record])

            first, _offset, first_revision = store.read_finalized_canary_snapshot_with_revision()
            assert first == [later_record]
            assert first_revision == 1

            store.finalize_cycle("recorded-first")
            records, offset, revision = store.read_finalized_canary_snapshot_with_revision()
            legacy_records, legacy_offset = store.read_finalized_canary_snapshot()

            assert records == [later_record, early_record]
            assert (legacy_records, legacy_offset) == (records, offset)
            assert store.finalized_cycle_ids() == ["finalized-first", "recorded-first"]
            assert revision == 2
            assert store.connect().execute(
                "SELECT cycle_id, finalize_seq FROM canary_cycles "
                "WHERE status = 'finalized' ORDER BY finalize_seq"
            ).fetchall() == [("finalized-first", 1), ("recorded-first", 2)]

            store.finalize_cycle("recorded-first")
            assert store.read_finalized_canary_snapshot_with_revision()[2] == revision

    def test_legacy_finalized_cycles_are_backfilled_in_first_record_order(
        self, tmp_path: Path
    ) -> None:
        first = make_record(cycle_id="first")
        second = make_record(cycle_id="second")
        with Store.init_project(tmp_path) as store:
            store.append_many([first])
            store.append_many([second])
            index_path = store.index_path

        with sqlite3.connect(index_path) as conn:
            conn.execute("UPDATE canary_cycles SET finalize_seq = NULL")
            conn.execute("DELETE FROM store_metadata WHERE key = 'finalized_revision'")

        with Store(tmp_path) as migrated:
            records, _offset, revision = migrated.read_finalized_canary_snapshot_with_revision()
            assert records == [first, second]
            assert revision == 2
            assert migrated.connect().execute(
                "SELECT cycle_id, finalize_seq FROM canary_cycles ORDER BY finalize_seq"
            ).fetchall() == [("first", 1), ("second", 2)]

    def test_revision_assertion_detects_late_finalization(self, tmp_path: Path) -> None:
        first = make_record(cycle_id="first")
        second = make_record(cycle_id="second")
        with Store.init_project(tmp_path) as store:
            store.append_many([first])
            _records, _offset, old_revision = store.read_finalized_canary_snapshot_with_revision()
            store.append_many([second])

            conn = store.connect()
            conn.execute("BEGIN IMMEDIATE")
            try:
                with pytest.raises(RuntimeError, match="expected revision 1, found 2"):
                    store.assert_finalized_revision(old_revision)
            finally:
                conn.rollback()

            _records, _offset, current_revision = (
                store.read_finalized_canary_snapshot_with_revision()
            )
            conn.execute("BEGIN IMMEDIATE")
            try:
                store.assert_finalized_revision(current_revision)
            finally:
                conn.rollback()

            with pytest.raises(RuntimeError, match="existing writer transaction"):
                store.assert_finalized_revision(current_revision)

    def test_snapshot_records_and_revision_share_one_read_transaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = make_record(cycle_id="first")
        finalized_during_read = make_record(cycle_id="finalized-during-read")
        with Store.init_project(tmp_path) as setup:
            setup.append_many([first])
            setup.append(finalized_during_read)

        reader = Store(tmp_path)
        reader.connect()
        original_revision_read = Store._current_finalized_revision

        def finalize_late_cycle() -> None:
            with Store(tmp_path) as writer:
                writer.finalize_cycle("finalized-during-read")

        with ThreadPoolExecutor(max_workers=1) as executor:

            def revision_after_concurrent_finalize(conn: sqlite3.Connection) -> int:
                executor.submit(finalize_late_cycle).result(timeout=5)
                return original_revision_read(conn)

            monkeypatch.setattr(
                reader, "_current_finalized_revision", revision_after_concurrent_finalize
            )
            records, _offset, revision = reader.read_finalized_canary_snapshot_with_revision()
        reader.close()

        assert records == [first]
        assert revision == 1
        with Store(tmp_path) as current:
            current_records, _offset, current_revision = (
                current.read_finalized_canary_snapshot_with_revision()
            )
            assert current_records == [first, finalized_during_read]
            assert current_revision == 2


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
            snapshot_before = store.read_finalized_canary_snapshot_with_revision()
            n = store.rebuild_index()
            assert n == 10
            assert store.config_events() == events_before
            assert store.count_records() == 10
            assert store.read_finalized_canary_snapshot_with_revision() == snapshot_before

    @pytest.mark.parametrize("mutation", ["content", "identity", "order"])
    def test_rebuild_rejects_changes_to_finalized_records(
        self, tmp_path: Path, mutation: str
    ) -> None:
        first = make_record(cycle_id="immutable", canary_id="first")
        second = make_record(cycle_id="immutable", canary_id="second")
        with Store.init_project(tmp_path) as store:
            store.append_many([first, second])
            before = store.read_finalized_canary_snapshot_with_revision()
            if mutation == "content":
                replacement = first.model_copy(
                    update={
                        "output": first.output.model_copy(
                            update={"text": "X" * len(first.output.text)}
                        )
                    }
                )
                raw = [replacement, second]
            elif mutation == "identity":
                replacement = first.model_copy(update={"id": "x" * len(first.id)})
                raw = [replacement, second]
            else:
                raw = [second, first]
            store.records_path.write_text(
                "".join(record.to_jsonl() + "\n" for record in raw),
                encoding="utf-8",
            )

            with pytest.raises(RuntimeError, match="alter immutable record"):
                store.rebuild_index()

            assert store.read_finalized_canary_snapshot_with_revision() == before

    def test_rebuild_protects_anytime_processed_cycle_even_if_open(self, tmp_path: Path) -> None:
        record = make_record(cycle_id="processed-open")
        with Store.init_project(tmp_path) as store:
            store.append(record)
            store.connect().execute(
                "INSERT INTO anytime_processed_cycles "
                "(fingerprint, cycle_id, processed_ts) VALUES ('epoch', ?, 'now')",
                (record.cycle_id,),
            )
            store.connect().commit()
            replacement = record.model_copy(
                update={"output": record.output.model_copy(update={"text": "XXXXX"})}
            )
            store.records_path.write_text(replacement.to_jsonl() + "\n", encoding="utf-8")

            with pytest.raises(RuntimeError, match="alter immutable record"):
                store.rebuild_index()

            indexed = (
                store.connect()
                .execute("SELECT record_json FROM records WHERE id = ?", (record.id,))
                .fetchone()
            )
            assert indexed is not None
            assert InteractionRecord.from_jsonl(str(indexed[0])) == record
