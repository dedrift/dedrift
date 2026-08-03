"""E-value tests: harness self-validation first, then the constructions.

The ordering is deliberate. A martingale test that passes tells you nothing
unless you have first shown it *fails* on something known to be broken. So
:class:`TestHarnessSelfValidation` establishes both directions on
constructions whose expectations we know analytically, and only then do the
production constructions get tested with the same machinery.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from dedrift.evalues import (
    EProcessState,
    PriorState,
    clopper_pearson,
    ebh,
    epoch_fingerprint,
    geometric_allocation,
    log_tilt_evalue,
    rate_evalue,
    symmetric_grid,
    tilt_from_materiality,
    update_process,
    ville_threshold,
    worst_case_log_evalue,
)
from dedrift.evalues.rates import (
    frozen_reference_hypergeometric_INVALID,
    hypergeometric_tilt,
)

#: Operating scale: 3 canaries x 7 reps = 21 current trials per family per
#: cycle, against a 5-cycle golden window (105 trials).
N_CUR, N_REF = 21, 105


def sup_expected_e(e_fn, n: int, p_grid) -> tuple[float, float]:
    """Return ``(sup_p E_p[E], argsup)`` by exact binomial enumeration.

    Exact rather than Monte Carlo: the whole point is to detect a violated
    inequality, and enumeration removes sampling noise from the verdict.
    """
    ss = np.arange(n + 1)
    vals = np.array([e_fn(int(s)) for s in ss], dtype=float)
    best, arg = -np.inf, np.nan
    for p in p_grid:
        got = float(np.sum(binom.pmf(ss, n, p) * vals))
        if got > best:
            best, arg = got, float(p)
    return best, arg


class TestHarnessSelfValidation:
    """Prove the harness detects both validity and its absence."""

    def test_conditional_tilt_is_exact_when_both_margins_random(self) -> None:
        """The fixture: averaging over both margins gives exactly 1.

        This is the positive control. If this drifts from 1, the harness
        itself is broken and no other result in this file means anything.
        """
        for m, n, psi in [(1, 1, 3.0), (105, 21, 2.5), (147, 7, 2.0)]:
            for p in (0.05, 0.2, 0.5):
                total = 0.0
                for r in range(m + 1):
                    pr = float(binom.pmf(r, m, p))
                    if pr < 1e-14:
                        continue
                    ss = np.arange(n + 1)
                    vals = np.array([hypergeometric_tilt(int(x), n, r, m, psi) for x in ss])
                    total += pr * float(np.sum(binom.pmf(ss, n, p) * vals))
                assert total == pytest.approx(1.0, abs=1e-9), f"m={m} n={n} psi={psi} p={p}"

    def test_harness_rejects_frozen_reference_construction(self) -> None:
        """The negative control, and the reason to trust the positive one.

        With the reference frozen, conditioning on the total is a bijection
        with the current count, so the hypergeometric is no longer the
        conditional law and the nuisance parameter returns. The violation is
        not subtle — three orders of magnitude — and the harness must see it.
        """
        r, m, psi = 5, N_REF, 2.5
        sup, arg = sup_expected_e(
            lambda s, r=r, m=m, psi=psi: frozen_reference_hypergeometric_INVALID(
                s, N_CUR, r, m, psi
            ),
            N_CUR,
            np.linspace(0.01, 0.6, 200),
        )
        assert sup > 100.0, f"harness failed to detect a known-invalid construction (sup={sup})"
        assert arg > 0.2, "violation should be worst where r/m is least representative of p"


class TestTiltExactness:
    def test_known_p_tilt_has_expectation_exactly_one(self) -> None:
        """``E(S;n,p,psi) = psi^S / (1-p+psi p)^n`` — the m.g.f. identity."""
        for n, p, psi in [(21, 0.05, 2.0), (21, 0.30, 0.5), (147, 0.05, 3.0), (7, 0.5, 1.5)]:
            sup, _ = sup_expected_e(
                lambda s, n=n, p=p, psi=psi: np.exp(log_tilt_evalue(s, n, p, psi)), n, [p]
            )
            assert sup == pytest.approx(1.0, abs=1e-10), f"n={n} p={p} psi={psi}"

    def test_mixture_of_evalues_is_an_evalue(self) -> None:
        """Arithmetic mean over a symmetric grid: valid by linearity."""
        grid = symmetric_grid((1.5, 2.0, 3.0))
        assert grid == tuple(sorted(grid))
        for p in (0.05, 0.2):
            sup, _ = sup_expected_e(
                lambda s, p=p: np.exp(worst_case_log_evalue(s, N_CUR, (p, p), grid)), N_CUR, [p]
            )
            assert sup <= 1.0 + 1e-9, f"mixture at known p exceeded 1 (p={p}, sup={sup})"


class TestWorstCaseConstruction:
    """The production construction: valid on the coverage event, and only there."""

    @pytest.mark.parametrize(
        ("m", "r", "n"), [(105, 5, 21), (147, 7, 21), (210, 11, 42), (105, 21, 21)]
    )
    def test_valid_for_every_p_inside_the_interval(self, m: int, r: int, n: int) -> None:
        gamma = 0.01
        lo, hi = clopper_pearson(r, m, gamma)
        grid = symmetric_grid((2.0,))
        sup, _ = sup_expected_e(
            lambda s, n=n, lo=lo, hi=hi: np.exp(worst_case_log_evalue(s, n, (lo, hi), grid)),
            n,
            np.linspace(lo, hi, 120),
        )
        assert sup <= 1.0 + 1e-6, f"conditional validity violated inside CI: sup={sup}"

    def test_conservatism_is_the_documented_price(self) -> None:
        """Worst-casing leaves budget on the table; quantify rather than hide."""
        lo, hi = clopper_pearson(5, N_REF, 0.01)
        grid = symmetric_grid((2.0,))
        sup, _ = sup_expected_e(
            lambda s, lo=lo, hi=hi: np.exp(worst_case_log_evalue(s, N_CUR, (lo, hi), grid)),
            N_CUR,
            np.linspace(lo, hi, 120),
        )
        assert 0.5 < sup < 1.0, f"expected measurable conservatism, got {sup}"

    def test_validity_fails_outside_coverage_so_gamma_is_load_bearing(self) -> None:
        """Documents *why* gamma is paid: off the coverage event, nothing holds."""
        lo, hi = clopper_pearson(5, N_REF, 0.01)
        grid = symmetric_grid((2.0,))
        sup, _ = sup_expected_e(
            lambda s, lo=lo, hi=hi: np.exp(worst_case_log_evalue(s, N_CUR, (lo, hi), grid)),
            N_CUR,
            np.linspace(0.001, 0.6, 250),
        )
        assert sup > 10.0, "if this were bounded, gamma would be unnecessary"

    def test_no_bet_when_data_missing(self) -> None:
        prior = PriorState(reference_successes=5, reference_trials=N_REF)
        assert rate_evalue(0, 0, prior).log_e == 0.0  # suppressed family
        assert not rate_evalue(0, 0, prior).placed
        assert rate_evalue(3, N_CUR, PriorState()).log_e == 0.0  # no reference
        assert not rate_evalue(3, N_CUR, PriorState()).placed


class TestPredictability:
    """The highest-risk bug class, closed structurally rather than by review."""

    def test_bet_cannot_depend_on_current_cycle_data(self) -> None:
        """Mutation test: perturb only the current cycle, bet must not move."""
        prior = PriorState(
            n_cycles=3, successes=4, trials=63, reference_successes=5, reference_trials=N_REF
        )
        bets = {rate_evalue(s, N_CUR, prior).bet for s in range(N_CUR + 1)}
        assert len(bets) == 1, "bet varied with current-cycle data"

    def test_interval_depends_only_on_prior_state(self) -> None:
        prior = PriorState(reference_successes=5, reference_trials=N_REF)
        details = {rate_evalue(s, N_CUR, prior).detail for s in (0, 5, 21)}
        assert len(details) == 1, "nuisance interval varied with current-cycle data"


class TestEProcess:
    def test_wealth_accumulates_and_crosses(self) -> None:
        alpha_prime = 0.04
        st = EProcessState(key=("golden", "f", "refusal", "rate"), fingerprint="fp1")
        threshold = ville_threshold(alpha_prime)
        for _ in range(40):
            out = rate_evalue(12, N_CUR, PriorState(reference_successes=5, reference_trials=N_REF))
            upd = update_process(
                st, out, fingerprint="fp1", alpha_prime=alpha_prime, successes=12, trials=N_CUR
            )
            st = upd.state
            if upd.crossed_now:
                break
        assert st.log_wealth >= threshold
        assert st.crossed_at is not None
        assert st.rise_cycle == 1

    def test_skipped_cycle_leaves_wealth_exactly_unchanged(self) -> None:
        st = EProcessState(
            key=("golden", "f", "refusal", "rate"), fingerprint="fp1", log_wealth=1.23
        )
        out = rate_evalue(0, 0, PriorState(reference_successes=5, reference_trials=N_REF))
        upd = update_process(st, out, fingerprint="fp1", alpha_prime=0.04, successes=0, trials=0)
        assert upd.state.log_wealth == 1.23
        assert upd.state.cycles == 1
        assert upd.state.bets_placed == 0

    def test_fingerprint_change_resets_and_announces(self) -> None:
        st = EProcessState(
            key=("golden", "f", "refusal", "rate"), fingerprint="fp1", log_wealth=2.5, epoch=0
        )
        out = rate_evalue(3, N_CUR, PriorState(reference_successes=5, reference_trials=N_REF))
        upd = update_process(
            st, out, fingerprint="fp2", alpha_prime=0.04, successes=3, trials=N_CUR
        )
        assert upd.was_reset
        assert upd.state.epoch == 1
        assert upd.state.log_wealth == pytest.approx(out.log_e)
        assert "per epoch" in (upd.reset_notice or "")

    def test_fingerprint_covers_every_invalidating_input(self) -> None:
        base = dict(
            suite_version="v1",
            embedder="hash",
            golden_cycles=("c1", "c2"),
            extractor_version="1",
        )
        fp = epoch_fingerprint(**base)
        for field, value in [
            ("suite_version", "v2"),
            ("embedder", "st:all-MiniLM-L6-v2"),
            ("golden_cycles", ("c1", "c3")),
            ("extractor_version", "2"),
            ("judge_version", "j1"),
        ]:
            assert epoch_fingerprint(**{**base, field: value}) != fp, field
        # order-insensitive in the golden set: same baseline, same epoch
        assert epoch_fingerprint(**{**base, "golden_cycles": ("c2", "c1")}) == fp

    def test_geometric_allocation_is_summable(self) -> None:
        total = sum(geometric_allocation(0.05, e) for e in range(200))
        assert total <= 0.05 + 1e-12


class TestEBH:
    def test_rejects_obvious_case(self) -> None:
        res = ebh([100.0, 80.0, 0.5, 0.1], alpha=0.05)
        assert res.n_rejected == 2
        assert res.rejected == [True, True, False, False]

    def test_rejects_nothing_when_no_evidence(self) -> None:
        res = ebh([1.0, 1.0, 1.0], alpha=0.05)
        assert res.n_rejected == 0
        assert res.threshold == float("inf")
        assert all(a == 1.0 for a in res.e_adjusted)

    def test_threshold_matches_the_rule(self) -> None:
        e = [100.0, 80.0, 0.5, 0.1]
        res = ebh(e, alpha=0.05)
        m, k = len(e), res.n_rejected
        assert res.threshold == pytest.approx(m / (0.05 * k))
        assert sorted(e, reverse=True)[k - 1] >= res.threshold

    def test_adjusted_levels_agree_with_decisions(self) -> None:
        e = [60.0, 25.0, 3.0, 0.2]
        alpha = 0.05
        res = ebh(e, alpha=alpha)
        for i, adj in enumerate(res.e_adjusted):
            assert res.rejected[i] == (adj <= alpha + 1e-9), i

    def test_nan_is_no_evidence_not_contagion(self) -> None:
        res = ebh([100.0, float("nan"), 80.0], alpha=0.05)
        assert res.rejected[1] is False
        assert res.n_rejected >= 1

    def test_empty(self) -> None:
        res = ebh([], alpha=0.05)
        assert res.n_rejected == 0


class TestMaterialityDerivedBets:
    def test_tilt_from_materiality_is_monotone_and_above_one(self) -> None:
        psi_small = tilt_from_materiality(0.05, 2.0)
        psi_big = tilt_from_materiality(0.05, 10.0)
        assert 1.0 < psi_small < psi_big

    def test_grid_is_symmetric_so_both_directions_are_covered(self) -> None:
        grid = symmetric_grid((tilt_from_materiality(0.05, 2.0),))
        assert len(grid) == 2
        assert grid[0] * grid[1] == pytest.approx(1.0)
