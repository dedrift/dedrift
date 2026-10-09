"""Read-only detector-health and ``dedrift doctor`` tests."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from typer.testing import CliRunner

from dedrift.anytime import run_anytime_check
from dedrift.check import _persistence_fingerprint, set_golden_baseline
from dedrift.cli import app
from dedrift.config import ProjectConfig
from dedrift.health import assess_health
from dedrift.schema import AgentConfig, InteractionOutput
from dedrift.sim import SimAgent, SimConfig
from dedrift.store import _FINALIZED_REVISION_KEY, Store

runner = CliRunner()


def _persist_full_check(store: Store, current_cycle: str) -> None:
    revision_row = store.connect().execute(
        "SELECT value FROM store_metadata WHERE key = ?", (_FINALIZED_REVISION_KEY,)
    ).fetchone()
    revision = int(revision_row[0]) if revision_row is not None else 0
    assessment = [
        {
            "baseline": baseline,
            "coverage_status": "FULL",
            "power_status": "NOT ASSESSED",
            "n_families_total": 6,
            "n_families_tested": 6,
            "n_primary_tests": 60,
            "n_valid_primary_tests": 60,
            "n_undefined_primary_tests": 0,
            "suppressed_families": [],
            "n_channels_total": 30,
            "n_channels_tested": 30,
            "missing_channels": [],
        }
        for baseline in ("rolling", "golden")
    ]
    golden = ["cycle-0000", "cycle-0001"]
    records, _offset = store.read_finalized_canary_snapshot()
    latest_records = [record for record in records if record.cycle_id == current_cycle]
    persistence_fingerprint = _persistence_fingerprint(
        ProjectConfig.load(store.project_dir),
        golden,
        latest_records,
        None,
    )
    params = {
        "current_cycle": current_cycle,
        "snapshot_finalized_revision": revision,
        "assessments": assessment,
        "overall_verdict": "OK",
        "declared_golden": golden,
        "pinned_embedder": None,
        "persistence_fingerprint": persistence_fingerprint,
    }
    store.connect().execute(
        "INSERT INTO checks (ts, baseline_kind, params_json, verdict, snapshot_offset, "
        "snapshot_record_ids_json) VALUES (?, 'dual', ?, ?, ?, NULL)",
        (
            "2026-01-01T00:00:00+00:00",
            json.dumps(params),
            "sudden=OK;cumulative=OK;overall=OK",
            store.records_path.stat().st_size,
        ),
    )
    store.connect().commit()


def _seed_ready_project(
    path: Path,
    *,
    custom_scalars: tuple[str, ...] = (),
    unset_fingerprints: bool = False,
) -> None:
    store = Store.init_project(path)
    config_text = store.config_path.read_text(encoding="utf-8")
    config_text = config_text.replace("canary_repetitions = 7", "canary_repetitions = 2")
    if custom_scalars:
        rendered = ", ".join(json.dumps(name) for name in custom_scalars)
        config_text = config_text.replace("custom_scalars = []", f"custom_scalars = [{rendered}]")
    store.config_path.write_text(config_text, encoding="utf-8")
    records = SimAgent(SimConfig(n_canaries=6, repetitions=2, seed=731)).run_cycles(5)
    if unset_fingerprints:
        unset = AgentConfig(
            model="test/model",
            prompt_hash="sha256:unset",
            tool_schema_hash="sha256:unset",
            agent_version="1",
        )
        records = [record.model_copy(update={"config": unset}) for record in records]
    store.append_many(records)
    set_golden_baseline(store, ["cycle-0000", "cycle-0001"])
    _persist_full_check(store, "cycle-0004")
    store.close()


def _digest(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def _project_snapshot(root: Path) -> dict[str, tuple[int, int, int, str | None]]:
    """Capture project names, metadata, and file bytes (but not access times)."""
    snapshot: dict[str, tuple[int, int, int, str | None]] = {}
    for path in sorted((root, *root.rglob("*"))):
        metadata = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        snapshot[str(path.relative_to(root.parent))] = (
            stat.S_IMODE(metadata.st_mode),
            metadata.st_size,
            metadata.st_mtime_ns,
            digest,
        )
    return snapshot


def test_ready_project_is_warning_not_false_green_and_read_only(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    store = Store(tmp_path, require_project=False)
    authoritative = [
        store.config_path,
        store.records_path,
        store.index_path,
        store.project_dir / "baseline.json",
    ]
    before = {path: _digest(path) for path in authoritative}
    before_tree = _project_snapshot(store.project_dir)
    assert not Path(f"{store.index_path}-wal").exists()
    assert not Path(f"{store.index_path}-shm").exists()

    health = assess_health(store)

    assert health.failures == ()
    assert health.status == "WARN"
    assert any(item.code == "embedder" for item in health.warnings)
    assert any(item.code == "evidence.power" for item in health.warnings)
    assert {path: _digest(path) for path in authoritative} == before
    assert _project_snapshot(store.project_dir) == before_tree

    cli = runner.invoke(app, ["doctor", "--project", str(tmp_path)])
    assert cli.exit_code == 0, cli.output
    assert "SUMMARY WARN" in cli.output
    assert "PASS [evidence.coverage.golden]" in cli.output
    assert _project_snapshot(store.project_dir) == before_tree


def test_doctor_private_snapshot_reads_live_wal_without_touching_it(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    store = Store(tmp_path)
    connection = store.connect()
    raw = connection.execute(
        "SELECT params_json FROM checks WHERE baseline_kind = 'dual' "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    params = json.loads(str(raw[0]))
    params["assessments"][0]["coverage_status"] = "NONE"
    connection.execute(
        "UPDATE checks SET params_json = ? WHERE id = "
        "(SELECT MAX(id) FROM checks WHERE baseline_kind = 'dual')",
        (json.dumps(params),),
    )
    connection.commit()
    wal = Path(f"{store.index_path}-wal")
    assert wal.is_file() and wal.stat().st_size > 0
    before_tree = _project_snapshot(store.project_dir)

    health = assess_health(Store(tmp_path, require_project=False))

    assert any(
        item.code == "evidence.coverage.rolling" and item.status == "FAIL"
        for item in health.findings
    )
    assert _project_snapshot(store.project_dir) == before_tree
    store.close()


def test_doctor_fails_on_complete_unfinalized_cycle(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    with Store(tmp_path) as store:
        pending = SimAgent(SimConfig(n_canaries=6, repetitions=2, seed=911)).run_cycle(5)
        store.append_many(
            pending,
            finalize_cycles=False,
            expected_cycle_counts={"cycle-0005": len(pending)},
        )

    result = runner.invoke(app, ["doctor", "--project", str(tmp_path)])

    assert result.exit_code == 1
    assert "FAIL [cycles.open]" in result.output
    assert "ready to finalize" in result.output
    assert "SUMMARY FAIL" in result.output


def test_doctor_fails_when_declared_custom_scalar_has_no_data(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path, custom_scalars=("quality_score",))

    health = assess_health(Store(tmp_path, require_project=False))

    failure = next(item for item in health.failures if item.code == "channel.quality_score")
    assert "latest=0/" in failure.message
    assert "reference=0/" in failure.message


def test_doctor_fails_on_unset_attribution_fingerprints(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path, unset_fingerprints=True)

    health = assess_health(Store(tmp_path, require_project=False))

    attribution = next(item for item in health.failures if item.code == "attribution")
    assert "prompt_hash" in attribution.message
    assert "tool_schema_hash" in attribution.message


def test_anytime_health_fails_on_post_epoch_imported_cycle(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    with Store(tmp_path) as store:
        config = store.config_path.read_text(encoding="utf-8")
        store.config_path.write_text(
            config.replace('inference = "fixed"', 'inference = "anytime"'),
            encoding="utf-8",
        )
        store.connect().execute(
            "INSERT INTO anytime_epochs (fingerprint, base_fingerprint, epoch_index, "
            "start_after_cycle, start_after_open_seq, declared_ts) "
            "VALUES ('epoch:e0', 'epoch', 0, 'cycle-0002', 3, ?)",
            ("2026-01-01T00:00:00+00:00",),
        )
        store.connect().commit()

    health = assess_health(Store(tmp_path, require_project=False))

    prospective = next(item for item in health.failures if item.code == "cycles.prospective")
    assert "cycle-0003" in prospective.message
    assert "cycle-0004" in prospective.message


def test_fresh_anytime_declaration_includes_latest_golden_cycle_identity(
    tmp_path: Path,
) -> None:
    store = Store.init_project(tmp_path)
    config_text = store.config_path.read_text(encoding="utf-8")
    config_text = config_text.replace("canary_repetitions = 7", "canary_repetitions = 2")
    config_text = config_text.replace('inference = "fixed"', 'inference = "anytime"')
    store.config_path.write_text(config_text, encoding="utf-8")
    records = SimAgent(SimConfig(n_canaries=6, repetitions=2, seed=194)).run_cycles(3)
    store.append_many(records)
    golden = ["cycle-0000", "cycle-0001", "cycle-0002"]
    set_golden_baseline(store, golden)
    declaration = run_anytime_check(store, ProjectConfig.load(store.project_dir))
    assert declaration.processed_cycles == ()
    store.close()

    health = assess_health(Store(tmp_path, require_project=False))

    freshness = next(
        item for item in health.findings if item.code == "evidence.freshness"
    )
    assert freshness.status == "PASS", freshness.message


def test_check_freshness_fails_after_baseline_changes_without_new_records(
    tmp_path: Path,
) -> None:
    _seed_ready_project(tmp_path)
    with Store(tmp_path) as store:
        set_golden_baseline(store, ["cycle-0000", "cycle-0002"])

    health = assess_health(Store(tmp_path, require_project=False))

    freshness = next(
        item for item in health.failures if item.code == "evidence.freshness"
    )
    assert "golden baseline changed after the check" in freshness.message
    assert "fingerprint changed after the check" in freshness.message
    cli = runner.invoke(app, ["doctor", "--project", str(tmp_path)])
    assert cli.exit_code == 1
    assert cli.output.count("FAIL [evidence.freshness]") == 1


def test_doctor_reports_invalid_config_and_corrupt_database(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        store.config_path.write_text("[project]\nunknown = true\n", encoding="utf-8")

    invalid = runner.invoke(app, ["doctor", "--project", str(tmp_path)])
    assert invalid.exit_code == 1
    assert "FAIL [config]" in invalid.output

    index = tmp_path / ".dedrift" / "index.db"
    index.write_bytes(b"not a sqlite database")
    corrupt = runner.invoke(app, ["doctor", "--project", str(tmp_path)])
    assert corrupt.exit_code == 1
    assert "FAIL [database]" in corrupt.output


def test_doctor_detects_equal_length_indexed_prefix_tampering(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        record = SimAgent(SimConfig(n_canaries=1, repetitions=2, seed=417)).run_cycle(0)[0]
        record = record.model_copy(update={"output": InteractionOutput(text="AAA")})
        store.append_many([record])
        payload = store.records_path.read_bytes()
        assert b"AAA" in payload
        store.records_path.write_bytes(payload.replace(b"AAA", b"ZZZ", 1))

    health = assess_health(Store(tmp_path, require_project=False))
    failures = [item for item in health.failures if item.code == "log.index"]
    assert len(failures) == 1
    assert "content validation failed" in failures[0].message

    cli = runner.invoke(app, ["doctor", "--project", str(tmp_path)])
    assert cli.exit_code == 1
    assert cli.output.count("FAIL [log.index]") == 1
    assert cli.output.count("SUMMARY FAIL") == 1


def test_doctor_detects_reordered_indexed_jsonl_lines(tmp_path: Path) -> None:
    _seed_ready_project(tmp_path)
    records_path = tmp_path / ".dedrift" / "logs" / "records.jsonl"
    lines = records_path.read_bytes().splitlines(keepends=True)
    assert len(lines) > 2
    lines[0], lines[1] = lines[1], lines[0]
    records_path.write_bytes(b"".join(lines))

    health = assess_health(Store(tmp_path, require_project=False))

    failures = [item for item in health.failures if item.code == "log.index"]
    assert len(failures) == 1
    assert "order conflicts at position 1" in failures[0].message
    cli = runner.invoke(app, ["doctor", "--project", str(tmp_path)])
    assert cli.exit_code == 1
    assert cli.output.count("FAIL [log.index]") == 1


def test_baseline_warning_covers_fixed_and_anytime_confirmatory_validity(
    tmp_path: Path,
) -> None:
    _seed_ready_project(tmp_path)

    result = runner.invoke(
        app,
        ["baseline", "set", "--first", "2", "--project", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "CONFIRMATORY VALIDITY" in result.output
    assert "fixed-sample p-values" in result.output
    assert "anytime evidence exploratory" in result.output
