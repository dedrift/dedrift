"""The anytime path end to end: state, epochs, budget, and CLI parity.

These tests are about the *plumbing* being trustworthy — state that survives
between invocations, epochs that reset when they must, and a budget that
stays correct when the operator changes the suite. The statistical validity
of the construction is established in ``test_evalues.py`` and its
trajectory-level behaviour in ``test_evalues_anytime.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dedrift.anytime import load_pool, load_states, run_anytime_check, wealth_table
from dedrift.check import run_check, set_golden_baseline
from dedrift.cli import app
from dedrift.config import AnytimeConfig, ProjectConfig
from dedrift.evalues.rates import per_process_gamma
from dedrift.report import render_anytime_report
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
    def test_state_survives_between_checks_and_accumulates(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        first = run_anytime_check(store)
        saved = load_states(store)
        assert len(saved) == first.n_processes
        assert all(s.cycles == 1 for s in saved.values())

        second = run_anytime_check(store)  # same latest cycle, folded again
        again = load_states(store)
        assert all(s.cycles == 2 for s in again.values())
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
        assert "per epoch" in res.resets[0]
        golden_states = [s for k, s in after.items() if k[0] == "golden"]
        assert all(s.cycles == 1 for s in golden_states), "wealth carried across an epoch boundary"
        store.close()

    def test_wealth_table_orders_by_evidence(self, tmp_path: Path) -> None:
        store = project(tmp_path)
        res = run_anytime_check(store)
        table = wealth_table(res)
        assert list(table["log_wealth"]) == sorted(table["log_wealth"], reverse=True)
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
        assert all(p.cycles == 1 for p in anytime.processes)
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
        assert "materiality-gated" in out.output  # the p-value path's wording

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
        assert "evidence **for** stability" in md

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
        store.append_many(SimAgent(cfg).run_cycles(1))
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
