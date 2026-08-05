"""CLI configuration authority and validation errors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from dedrift.cli import app
from dedrift.store import Store

runner = CliRunner()


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
