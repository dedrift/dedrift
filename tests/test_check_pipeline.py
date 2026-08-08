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
            assert at.event_relation in {"precedes_onset", "precedes_detection"}
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


class TestAuditRegressions:
    """Regression tests for defects found by the independent audit (v0.3.1).

    Each test names the defect it pins. The audit harness measured these on
    the installed 0.3.1 wheel; these tests keep them fixed.
    """

    def test_page_hinkley_survives_mid_history_gap(self, tmp_path: Path) -> None:
        """Defect: one missing (family, cycle) poisoned the family's PH
        streams with NaN — they could never alarm again (audit: 0/30).

        The default lambda=12 needs more than four post-drift cycles to
        cross because the causal re-standardization absorbs a persistent
        step within a handful of cycles; ph_lambda=4 makes the regression
        observable at this horizon. The invariant under test is not PH's
        power but that the stream still FUNCTIONS after a gap (before the
        fix the statistic was NaN-frozen at 0 and no lambda could alarm).
        """
        config = SimConfig(
            n_canaries=18,
            repetitions=7,
            post=drifted_profile(BehaviorProfile()),
            change_cycle=6,
            seed=11,
        )
        records = SimAgent(config).run_cycles(10)
        gapped = [
            r
            for r in records
            if not (r.cycle_id == "cycle-0003" and r.input.metadata.get("family") == "edge_case")
        ]
        store = Store.init_project(tmp_path)
        (tmp_path / ".dedrift" / "config.toml").write_text(
            "[detection]\nph_lambda = 4.0\n", encoding="utf-8"
        )
        store.append_many(gapped)
        cycles = sorted({r.cycle_id for r in gapped if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        result = run_check(store)  # current = cycle-0009; gap outside both windows
        assert result.current_cycle == "cycle-0009"
        assert not any(i.family == "edge_case" for i in result.composition_issues)
        ph = [
            f
            for f in result.flags
            if f.kind == "page_hinkley" and f.family == "edge_case"
        ]
        assert ph, "expected Page-Hinkley flags for the drifted family"
        assert any(f.change_cycle_id is not None for f in ph)
        store.close()

    def test_attribution_never_calls_post_onset_event_preceding(
        self, tmp_path: Path
    ) -> None:
        """Defect: absolute-time nearest-event search nominated config events
        that happened AFTER the estimated onset as 'before onset'."""
        config = SimConfig(
            n_canaries=18,
            repetitions=7,
            post=drifted_profile(BehaviorProfile()),
            change_cycle=5,
            seed=13,
        )
        records = SimAgent(config).run_cycles(9)
        pre_config = next(r.config for r in records if r.cycle_id == "cycle-0000")
        # Silent drift: behavior changes at cycle 5 but the config change is
        # deferred to cycle 7 (an unrelated, post-onset event).
        rewritten = [
            r.model_copy(update={"config": pre_config})
            if r.cycle_id in {"cycle-0005", "cycle-0006"}
            else r
            for r in records
        ]
        store = Store.init_project(tmp_path)
        store.append_many(rewritten)
        cycles = sorted({r.cycle_id for r in rewritten if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        result = run_check(store)
        assert result.n_alerts > 0
        for at in attribute(store, result):
            if at.event_relation == "precedes_onset":
                assert at.nearest_event_delta_hours is not None
                assert at.nearest_event_delta_hours >= 0
            elif at.event_relation == "precedes_detection":
                assert at.nearest_event_delta_hours is not None
                assert at.nearest_event_delta_hours < 0
            else:
                assert at.event_relation is None
        md = render_report(store, result)
        # A post-onset event must never be rendered as preceding the onset
        # (the 0.3.1 wording produced "-0.46 h after onset").
        import re

        assert not re.findall(r"-\d+(?:\.\d+)?\s*h before onset", md)
        store.close()

    def test_degraded_cycle_still_alerts_on_error_rate(self, tmp_path: Path) -> None:
        """Defect: >20% errors suppressed ALL alerts, so an error storm was a
        perfect drift evasion. had_error now escapes the suppression."""
        config = SimConfig(
            n_canaries=18,
            repetitions=7,
            post=drifted_profile(BehaviorProfile()),
            change_cycle=None,
            seed=17,
        )
        records = SimAgent(config).run_cycles(8)
        modified = [
            r.model_copy(update={"errors": ["simulated-failure"]})
            if r.cycle_id == "cycle-0007" and r.repetition <= 4
            else r
            for r in records
        ]
        store = Store.init_project(tmp_path)
        store.append_many(modified)
        cycles = sorted({r.cycle_id for r in modified if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        result = run_check(store)
        assert result.degraded
        assert result.overall_verdict == "DEGRADED DATA"
        error_alerts = [t for t in result.alerts() if t.signature == "had_error"]
        assert error_alerts, "error-rate surge must alert under degradation"
        other_alerts = [t for t in result.alerts() if t.signature != "had_error"]
        assert not other_alerts, "non-error channels stay suppressed when degraded"
        store.close()


class TestToolOrderChannel:
    def test_tool_order_reversal_alerts(self, tmp_path: Path) -> None:
        """The audit measured tool-call ORDER drift as invisible (counts and
        schema unchanged). tool_order_inversions closes the blind spot."""
        from dedrift.schema import ToolCall

        config = SimConfig(
            n_canaries=18,
            repetitions=7,
            pre=BehaviorProfile(tool_call_rate=3.5),
            change_cycle=None,
            seed=19,
        )
        records = SimAgent(config).run_cycles(8)

        def _sorted_calls(rec, *, reverse: bool) -> list:
            ordered = sorted(rec.tool_calls, key=lambda c: c.name, reverse=reverse)
            return [
                ToolCall(name=c.name, args_schema_ok=c.args_schema_ok, order=i + 1)
                for i, c in enumerate(ordered)
            ]

        modified = [
            r.model_copy(
                update={
                    "tool_calls": _sorted_calls(
                        r, reverse=r.cycle_id == "cycle-0007"
                    )
                }
            )
            if r.input.metadata.get("family") == "tool_heavy"
            else r
            for r in records
        ]
        store = Store.init_project(tmp_path)
        store.append_many(modified)
        cycles = sorted({r.cycle_id for r in modified if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        result = run_check(store)
        order_alerts = {
            t.baseline
            for t in result.alerts()
            if t.family == "tool_heavy" and t.signature == "tool_order_inversions"
        }
        assert "golden" in order_alerts
        store.close()

    def test_stable_order_does_not_alert(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        result = run_check(store)
        assert not [t for t in result.alerts() if t.signature == "tool_order_inversions"]
        store.close()


class TestReportCaveats:
    def test_refusal_alert_carries_semantics_note(self, tmp_path: Path) -> None:
        config = SimConfig(
            n_canaries=18,
            repetitions=7,
            pre=BehaviorProfile(refusal_prob=0.05),
            post=BehaviorProfile(refusal_prob=0.60),
            change_cycle=7,
            seed=23,
        )
        store = Store.init_project(tmp_path)
        records = SimAgent(config).run_cycles(8)
        store.append_many(records)
        cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        result = run_check(store)
        assert any(t.signature == "refusal" for t in result.alerts())
        md = render_report(store, result)
        assert "pattern-matched" in md and "phrasing" in md
        store.close()

    def test_semantic_channel_off_note_without_embedder(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        result = run_check(store)
        md = render_report(store, result)
        assert "Semantic channel OFF" in md
        store.close()

    def test_semantic_note_absent_with_embedder(self, tmp_path: Path) -> None:
        from dedrift.embeddings import pin_embedder

        store = Store.init_project(tmp_path)
        pin_embedder(store, "hash")
        config = SimConfig(n_canaries=12, repetitions=5, change_cycle=None, seed=29)
        records = SimAgent(config).run_cycles(7)
        store.append_many(records)
        cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        result = run_check(store)
        assert result.mmd_floors
        md = render_report(store, result)
        assert "Semantic channel OFF" not in md
        store.close()


class TestCycleEffectMode:
    """The v0.4.0 cluster-aware path: engagement, and the off switch."""

    def test_engagement_reported_under_cycle_offsets(self, tmp_path: Path) -> None:
        cfg = SimConfig(n_canaries=18, repetitions=7, change_cycle=None, seed=37,
                        cycle_effect_sigma=0.25)
        store = Store.init_project(tmp_path)
        records = SimAgent(cfg).run_cycles(8)
        store.append_many(records)
        cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        auto = ProjectConfig(cycle_effect="auto")
        result = run_check(store, config=auto)
        assert result.cycle_effects, "expected cycle-effect engagement under wobble"
        assert all(c["rho"] > 0.02 for c in result.cycle_effects)
        md = render_report(store, result)
        assert "Cycle-effect correction engaged" in md
        store.close()

    def test_off_restores_record_level_path(self, tmp_path: Path) -> None:
        cfg = SimConfig(n_canaries=18, repetitions=7, change_cycle=None, seed=41,
                        cycle_effect_sigma=0.25)
        store = Store.init_project(tmp_path)
        records = SimAgent(cfg).run_cycles(8)
        store.append_many(records)
        cycles = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        set_golden_baseline(store, cycles[0:3])
        base = ProjectConfig.load(store.project_dir)
        legacy = ProjectConfig(name=base.name, cycle_effect="off")
        result = run_check(store, config=legacy)
        assert result.cycle_effects == []
        store.close()

    def test_stable_agent_mostly_unengaged_and_quiet(self, tmp_path: Path) -> None:
        store = seeded_project(tmp_path, change_cycle=None)
        result = run_check(store)
        assert result.n_alerts == 0
        store.close()
