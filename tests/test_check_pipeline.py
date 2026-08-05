"""End-to-end check pipeline, attribution, and report tests (fast paths)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dedrift.attribution import attribute
from dedrift.canary import Canary, CanaryRunner, CanarySuite
from dedrift.check import get_golden_baseline, run_check, set_golden_baseline
from dedrift.config import ProjectConfig
from dedrift.report import render_report
from dedrift.schema import Source
from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from dedrift.store import Store
from tests.test_schema import make_config


def seeded_project(
    tmp_path: Path,
    n_cycles: int = 8,
    change_cycle: int | None = None,
    seed: int = 5,
    golden_last: slice = slice(0, 3),
) -> Store:
    """Create a project with simulated cycles and a frozen golden baseline."""
    config = SimConfig(
        n_canaries=18,
        repetitions=7,
        post=drifted_profile(BehaviorProfile()),
        change_cycle=change_cycle,
        seed=seed,
    )
    store = Store.init_project(tmp_path)
    records = SimAgent(config).run_cycles(n_cycles)
    store.append_many(records)
    cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
    set_golden_baseline(store, cycles[golden_last])
    return store


class TestBaseline:
    def test_set_and_get(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            store.append_many(
                SimAgent(SimConfig(n_canaries=6, repetitions=2, seed=2)).run_cycles(2)
            )
            set_golden_baseline(store, ["cycle-0001", "cycle-0000"])
            assert get_golden_baseline(store) == ["cycle-0000", "cycle-0001"]

    def test_unset_is_empty(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            assert get_golden_baseline(store) == []


class TestCheckNull:
    def test_null_run_reports_ok(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        result = run_check(store)
        assert result.verdict_sudden == "OK"
        assert result.verdict_cumulative == "OK"
        assert result.n_alerts == 0
        store.close()

    def test_reproducible(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        a = run_check(store)
        b = run_check(store)
        # repr comparison: dataclass == is False for NaN fields even when
        # bit-identical, and NaN p-values (degenerate tests) are legitimate.
        assert [repr(t.outcome) for t in a.tests] == [repr(t.outcome) for t in b.tests]
        assert a.verdict_sudden == b.verdict_sudden
        store.close()


class TestCheckDrift:
    def test_drift_detected_both_baselines(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, n_cycles=8, change_cycle=7)
        result = run_check(store)
        assert result.verdict_sudden == "DRIFT DETECTED"
        assert result.verdict_cumulative == "DRIFT DETECTED"
        assert result.n_alerts > 0
        signatures = {t.signature for t in result.alerts()}
        assert "output_words" in signatures
        store.close()

    def test_expected_contract_detects_correct_to_wrong_answers(self, tmp_path: Path) -> None:
        suite = CanarySuite(
            version="correctness-v1",
            canaries=[
                Canary(
                    id=f"answer-{i}",
                    family="happy_path",
                    input={"text": f"question {i}"},
                    expected={"answer": "42"},
                    rubric_id="answer-quality-v1",
                )
                for i in range(2)
            ],
        )
        answer = {"value": "42"}

        def agent(_: dict[str, object]) -> dict[str, object]:
            return {
                "text": "stable presentation",
                "structured": {"answer": answer["value"], "extra": "allowed"},
            }

        store = Store.init_project(tmp_path)
        runner = CanaryRunner(suite, agent, make_config(), repetitions=30)
        for index in range(4):
            runner.run_cycle(store=store, cycle_id=f"cycle-{index:04d}")
        set_golden_baseline(store, ["cycle-0000", "cycle-0001"])
        answer["value"] = "wrong"
        runner.run_cycle(store=store, cycle_id="cycle-0004")

        result = run_check(store)
        exact_tests = [t for t in result.tests if t.signature == "exact_match"]
        assert exact_tests
        assert {t.baseline for t in exact_tests} == {"rolling", "golden"}
        assert all(t.outcome.effect_raw == -1.0 for t in exact_tests)
        assert {t.baseline for t in result.alerts() if t.signature == "exact_match"} == {
            "rolling",
            "golden",
        }
        md = render_report(store, result)
        assert "structured_expected_subset.v1" in md
        assert "does not execute rubric/LLM judges" in md
        store.close()

    def test_alerts_persisted(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, n_cycles=8, change_cycle=7)
        run_check(store)
        rows = store.connect().execute("SELECT COUNT(*) FROM alerts").fetchone()
        assert rows[0] > 0
        store.close()

    def test_attribution_points_to_config_event(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, n_cycles=8, change_cycle=7)
        result = run_check(store)
        attributions = attribute(store, result)
        assert attributions
        for at in attributions:
            assert at.nearest_event_ts is not None
            # The model swap happened at the onset cycle; the event should be
            # within a few simulated hours of onset.
            assert at.nearest_event_delta_hours is not None
            assert abs(at.nearest_event_delta_hours) < 12
        store.close()


class TestMaterialityGate:
    def test_significant_but_immaterial_never_alerts(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        # Impossible materiality thresholds: nothing can alert even if BH rejects.
        cfg = ProjectConfig.load(store.project_dir)
        strict = ProjectConfig(
            name=cfg.name,
            canary_repetitions=cfg.canary_repetitions,
            rolling_window_cycles=cfg.rolling_window_cycles,
            fdr_q=0.5,  # deliberately loose significance
            permutations=200,
            seed=cfg.seed,
            ph_lambda=cfg.ph_lambda,
            ph_delta=cfg.ph_delta,
            materiality=type(cfg.materiality)(
                refusal_rate_pp=100.0,
                format_validity_pp=100.0,
                rate_default_pp=100.0,
                scalar_cohen_d=99.0,
                ks_distance=1.0,  # maximum possible D; unattained in this stable fixture
                dispersion_ratio=99.0,
                p95_relative=99.0,
            ),
        )
        result = run_check(store, config=strict)
        assert result.n_alerts == 0
        store.close()


class TestCompositionGuard:
    """Review finding #7: missing canary records must not masquerade as drift."""

    def _project_missing_canary(self, tmp_path: Path) -> tuple[Store, str, str]:
        """Stable project where one canary's records vanish from the current
        cycle (as if it timed out and the runner never logged it)."""
        config = SimConfig(n_canaries=18, repetitions=7, change_cycle=None, seed=5)
        store = Store.init_project(tmp_path)
        records = SimAgent(config).run_cycles(6)
        cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        current = cycles[-1]
        victim = next(r.canary_id for r in records if r.canary_id is not None)
        kept = [r for r in records if not (r.cycle_id == current and r.canary_id == victim)]
        victim_family = next(
            r.input.metadata.get("family", "unknown") for r in records if r.canary_id == victim
        )
        store.append_many(kept)
        set_golden_baseline(store, cycles[:3])
        return store, victim, str(victim_family)

    def test_missing_canary_suppresses_family_not_flagged_as_drift(self, tmp_path: Path) -> None:
        store, victim, family = self._project_missing_canary(tmp_path)
        result = run_check(store)
        # The issue is reported, for both baselines, naming the canary...
        issues = [i for i in result.composition_issues if i.family == family]
        assert {i.baseline for i in issues} == {"rolling", "golden"}
        assert all(victim in i.missing_canaries for i in issues)
        # ...the family's comparison is suppressed entirely (no tests, no
        # diagnostics manufactured on a known-broken design)...
        assert not [t for t in result.tests if t.family == family]
        assert not [f for f in result.flags if f.family == family]
        # ...so nothing alerts, and other families are still tested.
        assert result.n_alerts == 0
        assert {t.family for t in result.tests}  # other families present
        store.close()

    def test_unbalanced_reps_detected(self, tmp_path: Path) -> None:
        config = SimConfig(n_canaries=18, repetitions=7, change_cycle=None, seed=6)
        store = Store.init_project(tmp_path)
        records = SimAgent(config).run_cycles(6)
        cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        current = cycles[-1]
        victim = next(r.canary_id for r in records if r.canary_id is not None)
        # Keep the canary present but drop most of its current-cycle reps.
        dropped = 0
        kept = []
        for r in records:
            if r.cycle_id == current and r.canary_id == victim and dropped < 5:
                dropped += 1
                continue
            kept.append(r)
        with pytest.raises(ValueError, match="different repetition sets"):
            store.append_many(kept)
        store.close()

    def test_balanced_project_has_no_issues(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        result = run_check(store)
        assert result.composition_issues == []
        store.close()

    def test_report_shows_suppression(self, tmp_path: Path) -> None:
        store, _, _ = self._project_missing_canary(tmp_path)
        result = run_check(store)
        md = render_report(store, result)
        assert "COMPOSITION MISMATCH" in md
        assert "suppressed" in md
        store.close()

    def test_fully_suppressed_is_machine_readable_and_not_ok(self, tmp_path: Path) -> None:
        suite = CanarySuite(
            version="1",
            canaries=[
                Canary(
                    id=f"only-{i}",
                    family="happy_path",
                    input={"text": f"question {i}"},
                    expected={"answer": "42"},
                )
                for i in range(2)
            ],
        )

        def agent(_: dict[str, object]) -> dict[str, object]:
            return {"text": "same", "structured": {"answer": "42"}}

        store = Store.init_project(tmp_path)
        runner = CanaryRunner(suite, agent, make_config(), repetitions=5)
        for index in range(3):
            runner.run_cycle(store=store, cycle_id=f"cycle-000{index}")
        set_golden_baseline(store, ["cycle-0000", "cycle-0001"])
        partial = runner.run_cycle(cycle_id="cycle-0003")
        store.append_many([r for r in partial if r.canary_id == "only-0"])

        result = run_check(store)
        assert result.tests == []
        assert result.verdict_sudden == "NO VALID COMPARISON"
        assert result.verdict_cumulative == "NO VALID COMPARISON"
        assert result.overall_verdict == "NO VALID COMPARISON"
        assert {a.coverage_status for a in result.assessments} == {"NONE"}
        assert {a.power_status for a in result.assessments} == {"NOT ASSESSED"}
        assert all(a.n_valid_primary_tests == 0 for a in result.assessments)

        params_raw, persisted_verdict = (
            store.connect()
            .execute("SELECT params_json, verdict FROM checks ORDER BY id DESC LIMIT 1")
            .fetchone()
        )
        params = json.loads(params_raw)
        assert params["overall_verdict"] == "NO VALID COMPARISON"
        assert all(a["coverage_status"] == "NONE" for a in params["assessments"])
        assert "overall=NO VALID COMPARISON" in persisted_verdict

        md = render_report(store, result)
        assert "# dedrift report — NO VALID COMPARISON" in md
        assert "not a green result" in md
        store.close()

    def test_suite_version_change_is_an_experiment_boundary(self, tmp_path: Path) -> None:
        base_suite = CanarySuite(
            version="1",
            canaries=[
                Canary(
                    id="only",
                    family="happy_path",
                    input={"text": "question"},
                    expected={"answer": "42"},
                )
            ],
        )

        def agent(_: dict[str, object]) -> dict[str, object]:
            return {"text": "same", "structured": {"answer": "42"}}

        store = Store.init_project(tmp_path)
        runner = CanaryRunner(base_suite, agent, make_config(), repetitions=5)
        for index in range(3):
            runner.run_cycle(store=store, cycle_id=f"cycle-000{index}")
        set_golden_baseline(store, ["cycle-0000", "cycle-0001"])
        changed_suite = CanarySuite(version="2", canaries=base_suite.canaries)
        CanaryRunner(changed_suite, agent, make_config(), repetitions=5).run_cycle(
            store=store, cycle_id="cycle-0003"
        )

        result = run_check(store)
        assert result.overall_verdict == "NO VALID COMPARISON"
        assert result.tests == []
        assert all(i.changed_canaries == ("only",) for i in result.composition_issues)
        store.close()


class TestSourceIsolation:
    def test_production_record_with_cycle_id_cannot_enter_canary_check(
        self, tmp_path: Path
    ) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        canary_result = run_check(store)
        last = store.read_records()[-1]
        production = last.model_copy(
            update={
                "id": "production-contamination",
                "source": Source.PRODUCTION,
                "cycle_id": "cycle-9999",
            }
        )
        store.append(production)

        isolated_result = run_check(store)
        assert isolated_result.current_cycle == canary_result.current_cycle
        assert isolated_result.verdict_sudden == canary_result.verdict_sudden
        assert isolated_result.verdict_cumulative == canary_result.verdict_cumulative
        store.close()


class TestReport:
    def test_report_renders_and_is_deterministic(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, n_cycles=8, change_cycle=7)
        result = run_check(store)
        md1 = render_report(store, result)
        md2 = render_report(store, result)
        assert md1 == md2
        assert "DRIFT DETECTED" in md1
        assert "consistent with" in md1 or "Attribution" in md1
        assert 'never "caused by"' in md1
        assert "heuristic" in md1.lower()
        store.close()

    def test_ok_report(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        result = run_check(store)
        md = render_report(store, result)
        assert "# dedrift report — OK" in md
        store.close()
