"""The anytime path end to end: state, epochs, budget, and CLI parity.

These tests are about the *plumbing* being trustworthy — state that survives
between invocations, epochs that reset when they must, and a budget that
stays correct when the operator changes the suite. The statistical validity
of the construction is established in ``test_evalues.py`` and its
trajectory-level behaviour in ``test_evalues_anytime.py``.
"""

from __future__ import annotations

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
from dedrift.evalues.rates import per_process_gamma
from dedrift.report import render_anytime_report
from dedrift.schema import InteractionInput, InteractionRecord, Source
from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from dedrift.store import Store

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
    store = Store.init_project(tmp_path)
    store.append_many(records)
    ids = list(dict.fromkeys(r.cycle_id for r in records if r.cycle_id is not None))
    set_golden_baseline(store, ids[:3])
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
        """e-BH needs every input valid, so coverage failures union-bound.

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
    def test_all_unseen_cycles_are_folded_once_and_recheck_is_read_only(
        self, tmp_path: Path
    ) -> None:
        store = project(tmp_path)
        first = run_anytime_check(store)
        saved = load_states(store)
        assert len(saved) == first.n_processes
        assert first.processed_cycles == ("cycle-0003", "cycle-0004", "cycle-0005")
        assert load_processed_cycles(store, first.fingerprint) == set(first.processed_cycles)
        assert all(s.cycles == 3 for s in saved.values())
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
    def test_degraded_cycle_is_neutral_and_does_not_train_the_prior(self, tmp_path: Path) -> None:
        records = make_records(cycles=4)
        current = records[-1].cycle_id
        degraded = [
            record.model_copy(update={"errors": ["forced collection failure"]})
            if record.cycle_id == current
            else record
            for record in records
        ]
        store = store_records(tmp_path, degraded)

        result = run_anytime_check(store)
        states = load_states(store)
        store.close()

        assert result.verdict == "DEGRADED DATA"
        assert result.processed_cycles == (current,)
        assert all(state.cycles == 1 for state in states.values())
        assert all(state.log_wealth == 0.0 for state in states.values())
        assert all(state.bets_placed == 0 for state in states.values())
        assert all(state.prior.n_cycles == 0 for state in states.values())
        assert all(state.prior.successes == 0 for state in states.values())
        assert all(state.prior.trials == 0 for state in states.values())

    def test_composition_suppression_is_neutral_for_only_the_bad_family(
        self, tmp_path: Path
    ) -> None:
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

        run_anytime_check(store)
        states = load_states(store)
        store.close()

        bad = [state for key, state in states.items() if key[1] == bad_family]
        good = [state for key, state in states.items() if key[1] != bad_family]
        assert bad
        assert all(state.log_wealth == 0.0 for state in bad)
        assert all(state.bets_placed == 0 for state in bad)
        assert all(state.prior.n_cycles == 0 for state in bad)
        assert all(state.prior.successes == 0 for state in bad)
        assert all(state.prior.trials == 0 for state in bad)
        assert any(state.bets_placed == 1 for state in good)

    def test_mixed_suite_backlog_is_rejected_without_state_mutation(self, tmp_path: Path) -> None:
        records = make_records(cycles=6)
        changed = [
            stamp_suite(
                record,
                "sha256:suite-b" if record.cycle_id == "cycle-0005" else "sha256:suite-a",
            )
            for record in records
        ]
        store = store_records(tmp_path, changed)

        with pytest.raises(ValueError, match="backlog spans a canary-suite boundary"):
            run_anytime_check(store)

        assert load_states(store) == {}
        assert (
            store.connect().execute("SELECT COUNT(*) FROM anytime_processed_cycles").fetchone()[0]
            == 0
        )
        assert store.connect().execute("SELECT COUNT(*) FROM checks").fetchone()[0] == 0
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
        store = project(tmp_path, cycles=8, change_cycle=7)
        anytime = run_anytime_check(store)
        assert anytime.verdict in {"OK", "DRIFT DETECTED"}
        assert all(p.cycles == 5 for p in anytime.processes)
        store.close()

    def test_degraded_data_suppresses_alerts_on_the_anytime_path_too(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        res = run_anytime_check(store, ProjectConfig())
        assert res.degraded is False  # sim data is clean; the branch exists and is exercised
        store.close()


class TestCli:
    def test_inference_flag_selects_the_path(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        store.close()
        out = runner.invoke(app, ["check", "--project", str(tmp_path), "--inference", "anytime"])
        assert out.exit_code in (0, 2), out.output
        assert "Anytime-valid: alpha=" in out.output
        assert "e-processes" in out.output
        assert "PER EPOCH" in out.output

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
        assert out.exit_code == 0, out.output
        assert "anytime-valid" in out.output
        assert "per epoch" in out.output


class TestEpochPool:
    """Membership is decided once per epoch — the property the guarantee needs."""

    def test_signatures_without_reference_data_are_excluded(self, tmp_path: Path) -> None:
        """A process that cannot produce evidence must not consume budget.

        The simulator declares no expected answers, so ``exact_match`` is
        entirely absent. Including it would shrink every other process's
        coverage budget and raise the e-BH threshold for nothing.
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


class TestTwoSampleRateModel:
    """The v0.3.2 two-sample SAFE rate process: power and the config switch."""

    def test_twosample_detects_strong_rate_shift(self, tmp_path: Path) -> None:
        """The frozen-CP construction could not do this at canary scale: the
        audit measured 0/30 detections at +20pp over 60 cycles (v0.3.1)."""
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
        store.append_many(records)
        ids = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        set_golden_baseline(store, ids[:3])
        result = run_anytime_check(store)  # folds all unseen post-golden cycles
        assert result.n_alerts > 0
        refusal = [p for p in result.processes if p.key[2] == "refusal"]
        assert any(p.rejected for p in refusal)
        store.close()

    def test_twosample_stays_quiet_on_stable(self, tmp_path: Path) -> None:
        store = project(tmp_path, cycles=10, change_cycle=None)
        result = run_anytime_check(store)
        assert result.n_alerts == 0
        store.close()

    def test_frozen_cp_still_available(self, tmp_path: Path) -> None:
        store = project(tmp_path, cycles=6, change_cycle=None)
        cfg = ProjectConfig.load(store.project_dir)
        legacy = ProjectConfig(
            name=cfg.name,
            anytime=AnytimeConfig(rate_model="frozen_cp"),
        )
        result = run_anytime_check(store, config=legacy)
        assert result.n_alerts == 0
        assert result.gamma_per_process > 0  # coverage budget is spent here
        store.close()

    def test_twosample_spends_no_coverage_budget(self, tmp_path: Path) -> None:
        store = project(tmp_path, cycles=6, change_cycle=None)
        result = run_anytime_check(store)
        assert result.gamma_per_process == 0.0
        assert result.alpha_prime == result.alpha
        store.close()

    def test_twosample_detects_midrate_shift_frozen_cp_missed(self, tmp_path: Path) -> None:
        """Audit cell: p0=0.30 -> 0.50. The frozen-CP construction's interval
        contains the alternative at this base rate (measured 0/30 in 60
        cycles on the audit harness); the two-sample process detects within
        60 cycles (its per-cycle growth decays ~1/t as the pooled posterior
        learns the alternative, so mid-rate shifts take tens of cycles —
        documented in docs/anytime.md)."""
        cfg = SimConfig(
            n_canaries=18,
            repetitions=7,
            pre=BehaviorProfile(refusal_prob=0.30),
            post=BehaviorProfile(refusal_prob=0.50),
            change_cycle=8,
            seed=43,
        )
        store = Store.init_project(tmp_path)
        records = SimAgent(cfg).run_cycles(60)
        store.append_many(records)
        ids = sorted({r.cycle_id for r in records if r.cycle_id is not None})
        set_golden_baseline(store, ids[:3])
        result = run_anytime_check(store)
        assert result.n_alerts > 0
        store.close()
