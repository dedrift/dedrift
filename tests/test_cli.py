"""CLI configuration authority and validation errors."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dedrift.check import set_golden_baseline
from dedrift.cli import app
from dedrift.sim import SimAgent, SimConfig
from dedrift.store import Store

runner = CliRunner()


def test_canary_run_rejects_imported_non_callable(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """\
version: '1'
canaries:
  - id: one
    family: happy_path
    input:
      text: hello
""",
        encoding="utf-8",
    )
    with Store.init_project(tmp_path):
        pass

    result = runner.invoke(
        app,
        [
            "canary",
            "run",
            "--suite",
            str(suite),
            "--agent",
            "math:pi",
            "--model",
            "test/model",
            "--project",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "not callable" in result.output
    with Store(tmp_path) as store:
        assert store.count_records() == 0


def test_init_rejects_unrelated_nonempty_sqlite_file(tmp_path: Path) -> None:
    project = tmp_path / ".dedrift"
    (project / "logs").mkdir(parents=True)
    (project / "config.toml").write_text("[project]\n", encoding="utf-8")
    (project / "logs" / "records.jsonl").write_text("", encoding="utf-8")
    connection = sqlite3.connect(project / "index.db")
    connection.execute("CREATE TABLE unrelated (id INTEGER)")
    connection.commit()
    connection.close()

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 1
    assert "failed integrity validation" in result.output


def test_init_refuses_to_replace_a_missing_authoritative_index(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        store.append_many(SimAgent(SimConfig(n_canaries=1, repetitions=2)).run_cycles(1))
        index_path = store.index_path
    index_path.unlink()

    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 1
    assert "Could not initialize" in result.output
    assert "refusing to recreate" in result.output
    assert not index_path.exists()


def set_repetitions(store: Store, repetitions: int) -> None:
    content = store.config_path.read_text(encoding="utf-8")
    store.config_path.write_text(
        content.replace("canary_repetitions = 7", f"canary_repetitions = {repetitions}"),
        encoding="utf-8",
    )


def test_sim_uses_authoritative_project_repetitions(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        set_repetitions(store, 3)

    result = runner.invoke(
        app,
        ["sim", "--cycles", "1", "--canaries", "6", "--project", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    with Store(tmp_path) as store:
        assert store.count_records() == 18


def test_sim_rejects_repetition_conflict(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        set_repetitions(store, 3)

    result = runner.invoke(
        app,
        [
            "sim",
            "--cycles",
            "1",
            "--canaries",
            "6",
            "--repetitions",
            "4",
            "--project",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "conflicts with project.canary_repetitions=3" in result.output
    with Store(tmp_path) as store:
        assert store.count_records() == 0


def test_sim_large_foreign_cycle_id_is_bounded(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        foreign = [
            record.model_copy(update={"cycle_id": "cycle-999999999"})
            for record in SimAgent(SimConfig(n_canaries=1, repetitions=2, seed=99)).run_cycles(1)
        ]
        store.append_many(foreign)

    result = runner.invoke(
        app,
        ["sim", "--cycles", "1", "--canaries", "6", "--project", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "(0..0)" in result.output
    with Store(tmp_path) as store:
        assert store.finalized_cycle_ids() == ["cycle-999999999", "cycle-0000"]


def test_sim_finds_a_contiguous_gap_for_the_whole_requested_run(tmp_path: Path) -> None:
    source = SimAgent(SimConfig(n_canaries=1, repetitions=2, seed=99)).run_cycles(3)
    with Store.init_project(tmp_path) as store:
        store.append_many(
            [record for record in source if record.cycle_id in {"cycle-0000", "cycle-0002"}]
        )

    result = runner.invoke(
        app,
        ["sim", "--cycles", "2", "--canaries", "1", "--project", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "(3..4)" in result.output
    with Store(tmp_path) as store:
        assert store.finalized_cycle_ids() == [
            "cycle-0000",
            "cycle-0002",
            "cycle-0003",
            "cycle-0004",
        ]


def test_log_rejects_empty_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with Store.init_project(tmp_path):
        pass
    result = runner.invoke(app, ["log", str(empty), "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "no interaction records" in result.output


def test_embedder_show_requires_a_project(tmp_path: Path) -> None:
    result = runner.invoke(app, ["embedder", "show", "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "No dedrift project" in result.output


def test_report_output_error_is_clean(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        records = SimAgent(SimConfig(n_canaries=6, repetitions=2, seed=8)).run_cycles(4)
        store.append_many(records)
        set_golden_baseline(store, ["cycle-0000", "cycle-0001"])
    output_directory = tmp_path / "report-target"
    output_directory.mkdir()

    result = runner.invoke(
        app,
        ["report", "--out", str(output_directory), "--project", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "Could not write report" in result.output
    assert "Traceback" not in result.output


def test_canary_command_rejects_repetition_conflict_before_execution(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        set_repetitions(store, 3)

    result = runner.invoke(
        app,
        [
            "canary",
            "run",
            "--suite",
            str(tmp_path / "not-needed.yaml"),
            "--agent",
            "not_needed:agent",
            "--model",
            "provider/model@v1",
            "--repetitions",
            "4",
            "--project",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "conflicts with project.canary_repetitions=3" in result.output


def test_check_reports_invalid_config_without_traceback(tmp_path: Path) -> None:
    with Store.init_project(tmp_path) as store:
        store.config_path.write_text(
            "[detection]\nfdr_qq = 0.05\n",
            encoding="utf-8",
        )

    result = runner.invoke(app, ["check", "--project", str(tmp_path)])
    assert result.exit_code == 1
    assert "Invalid" in result.output
    assert "detection.fdr_qq" in result.output


@pytest.mark.parametrize(
    ("overall", "exit_code"),
    [
        ("OK", 0),
        ("DRIFT DETECTED", 2),
        ("DEGRADED DATA", 3),
        ("NO VALID COMPARISON", 3),
        ("PARTIAL COVERAGE", 3),
        ("NO REFERENCE", 3),
    ],
)
def test_fixed_check_exit_codes_follow_overall_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overall: str,
    exit_code: int,
) -> None:
    with Store.init_project(tmp_path):
        pass
    fake = SimpleNamespace(
        current_cycle="cycle-0001",
        overall_verdict=overall,
        rolling_cycles=[],
        golden_cycles=[],
        verdict_sudden="NO REFERENCE",
        verdict_cumulative="NO REFERENCE",
        degraded=overall == "DEGRADED DATA",
        composition_issues=[],
        n_alerts=int(overall == "DRIFT DETECTED"),
        fdr_q=0.05,
        alerts=lambda: [],
    )
    monkeypatch.setattr("dedrift.check.run_check", lambda store, config: fake)

    result = runner.invoke(app, ["check", "--project", str(tmp_path)])
    assert result.exit_code == exit_code
    assert f"Overall: {overall}" in result.output


def test_anytime_without_processes_exits_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Store.init_project(tmp_path):
        pass
    fake = SimpleNamespace(
        current_cycle="cycle-0001",
        fingerprint="sha256:test",
        alpha=0.05,
        alpha_prime=0.03,
        gamma_total=0.02,
        gamma_per_process=0.02,
        n_processes=0,
        verdict="NO GOLDEN BASELINE",
        degraded=False,
        coverage_status="NO REFERENCE",
        suppressed_families=(),
        resets=[],
        processes=[],
        n_alerts=0,
        alerts=lambda: [],
    )
    monkeypatch.setattr("dedrift.anytime.run_anytime_check", lambda store, config: fake)

    result = runner.invoke(
        app,
        ["check", "--inference", "anytime", "--project", str(tmp_path)],
    )
    assert result.exit_code == 3
    assert "not active" in result.output
