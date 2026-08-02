"""Store round-trip, config-event detection, and index-rebuild tests."""

from __future__ import annotations

from pathlib import Path

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
