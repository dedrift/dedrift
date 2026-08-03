"""E-value contracts: what makes a construction admissible here.

An e-value for a null ``H0`` is a non-negative random variable ``E`` with
``sup_{P in H0} E_P[E] <= 1``. Two consequences we rely on:

* **Validity (Markov).** ``P(E >= 1/alpha) <= alpha`` — rejecting when
  ``E >= 1/alpha`` is a level-alpha test with no distributional theory.
* **Anytime-validity (Ville).** If each ``E_t`` satisfies
  ``E[E_t | F_{t-1}] <= 1``, then ``M_T = prod_{t<=T} E_t`` is a
  non-negative supermartingale with ``M_0 = 1`` and
  ``P(exists T : M_T >= 1/alpha) <= alpha`` over an unbounded horizon.

The second property is the whole point: it is what lets an operator monitor
forever and stop whenever they like. It also imposes the constraint that
governs this package's design — **predictability**. The bet at cycle ``t``
must be chosen from data strictly before ``t``. Using current-cycle data to
choose the current bet destroys the supermartingale property while leaving
every number looking plausible, which is the most dangerous failure mode
available to this project.

We enforce predictability structurally rather than by review: bet selection
receives a :class:`PriorState`, which by construction carries only
summaries of completed cycles. There is no code path by which the current
observation reaches a bet. ``tests/test_evalues.py`` asserts this with a
mutation test (perturb the current cycle, assert the bet is unchanged).

Conditional vs marginal validity — the trap we already fell into
----------------------------------------------------------------
A construction can satisfy ``E[E] <= 1`` when averaged over *all* the data
it uses, yet fail ``E[E_t | F_{t-1}] <= 1`` once part of that data is fixed
and known. Fisher's conditioning for a 2x2 table is exactly such a case: it
averages over both margins as random. Freeze the reference count and
conditioning on the total becomes a bijection with the current count, the
hypergeometric stops being the conditional law, and the nuisance parameter
returns — measured at ``sup_p E[E] = 2132`` at our operating scale, against
``1.000000`` when both margins are random. Marginal validity is not enough;
the martingale needs the conditional statement. See
:func:`dedrift.evalues.rates.hypergeometric_tilt` (valid, fixture) and
:func:`dedrift.evalues.rates.frozen_reference_hypergeometric_INVALID`
(retained deliberately so the test harness can prove it detects failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

#: Smallest log-wealth increment we bother recording; guards log(0).
LOG_FLOOR = -50.0


@dataclass(frozen=True)
class PriorState:
    """Summaries of *completed* cycles — the only input a bet may see.

    Frozen and deliberately minimal. If a field would be needed that can
    only be computed from the current cycle, that is a design error, not a
    missing field.

    Attributes:
        n_cycles: Completed cycles contributing to this process.
        successes: Successes observed in completed current-side cycles.
        trials: Trials observed in completed current-side cycles.
        reference_successes: Successes in the reference window.
        reference_trials: Trials in the reference window.
    """

    n_cycles: int = 0
    successes: int = 0
    trials: int = 0
    reference_successes: int = 0
    reference_trials: int = 0

    def with_cycle(self, successes: int, trials: int) -> PriorState:
        """Return the state after folding in a now-completed cycle."""
        return PriorState(
            n_cycles=self.n_cycles + 1,
            successes=self.successes + successes,
            trials=self.trials + trials,
            reference_successes=self.reference_successes,
            reference_trials=self.reference_trials,
        )


@dataclass(frozen=True)
class EValueOutcome:
    """One cycle's e-value, in logs.

    Attributes:
        log_e: ``log E_t``. Zero means "no bet placed" (``E_t = 1``), which
            is the correct contribution for a suppressed or degenerate
            cycle: it preserves the supermartingale exactly.
        placed: False when no bet was possible (insufficient data,
            degenerate margin, suppressed family).
        detail: Human-readable note for the report.
        bet: The predictable bet parameters actually used.
    """

    log_e: float = 0.0
    placed: bool = False
    detail: str = "no bet"
    bet: tuple[float, ...] = field(default_factory=tuple)

    @property
    def e(self) -> float:
        """``E_t`` on the natural scale (for small batteries / reports)."""
        return float(np.exp(self.log_e))


class EValueConstruction(Protocol):
    """A per-cycle e-value construction.

    Implementations must satisfy ``E[E_t | prior] <= 1`` under their stated
    null, and must not read anything beyond ``prior`` when choosing a bet.
    """

    def __call__(
        self, successes: int, trials: int, prior: PriorState
    ) -> EValueOutcome:  # pragma: no cover - protocol
        """Return this cycle's e-value, betting only on ``prior``."""
        ...
