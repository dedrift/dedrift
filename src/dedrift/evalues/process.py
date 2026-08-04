"""E-process accumulation: log-wealth, epochs, and the Ville threshold.

An e-process is the running product of per-cycle e-values. We store its
logarithm, because the product overflows on any horizon worth monitoring.

Epoch semantics — the part to read
----------------------------------
An e-process is only meaningful while the hypothesis and the measuring
instrument are fixed. Change the canary suite, the embedder, the golden
baseline, or the signature extractor and the null being tested is a
different null; wealth accumulated under the old one is not evidence about
the new one. So the process resets, and the report says so.

The consequence is that the guarantee is **alpha per epoch**, where an
epoch is a maximal stretch with a fixed fingerprint. This is not a weakened
guarantee to be engineered around: a guarantee spanning a hypothesis change
would be meaningless, not stronger. Operators who want a genuine
unbounded-horizon bound across arbitrarily many epochs can opt into a
summable allocation (:func:`geometric_allocation`), which spends
``alpha * 2**-(e+1)`` on epoch ``e`` so the total is at most ``alpha`` however
many epochs occur.

Skipped cycles contribute ``E_t = 1`` (log-wealth unchanged). That is the
exact and correct handling for a suppressed family — not an approximation,
and importantly not the same as carrying a stale value forward.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

from dedrift.evalues.base import EValueOutcome, PriorState


def epoch_fingerprint(
    *,
    suite_version: str,
    embedder: str,
    golden_cycles: tuple[str, ...],
    extractor_version: str,
    judge_version: str = "",
) -> str:
    """Hash of everything whose change invalidates accumulated evidence.

    Anything that alters what is being measured, or what it is measured
    against, belongs here. Silently continuing an e-process across such a
    change would state a guarantee that does not hold, which is the worst
    failure this project can ship.
    """
    payload = json.dumps(
        {
            "suite": suite_version,
            "embedder": embedder,
            "golden": sorted(golden_cycles),
            "extractor": extractor_version,
            "judge": judge_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def ville_threshold(alpha_prime: float) -> float:
    """``log(1/alpha')`` — the crossing level for the wealth process."""
    if not 0 < alpha_prime < 1:
        msg = f"alpha' must be in (0,1), got {alpha_prime}"
        raise ValueError(msg)
    return float(-math.log(alpha_prime))


def geometric_allocation(alpha: float, epoch: int) -> float:
    """``alpha * 2**-(epoch+1)``: summable, so lifetime spend <= alpha.

    Preferable to splitting ``alpha`` across an anticipated epoch count,
    which requires knowing that count and wastes budget when resets are
    rare. Offered as a mode, not a default — per-epoch is the honest
    default because a reset means the hypothesis changed.
    """
    return float(alpha * 2.0 ** -(epoch + 1))


@dataclass(frozen=True)
class EProcessState:
    """Persistent state for one ``(baseline, family, signature, channel)``.

    Attributes:
        key: Identity tuple.
        log_wealth: Running ``log M_T``; 0 at epoch start.
        cycles: Cycles folded in (bets placed or not).
        bets_placed: Cycles where an admissible bet existed.
        epoch: Epoch index; incremented on every reset.
        fingerprint: Epoch fingerprint this wealth was accumulated under.
        prior: Completed-cycle summaries feeding the next bet.
        peak_log_wealth: Running maximum, for reporting.
        rise_cycle: Cycle index at which the current run-up began — a
            natural onset estimate, and a better one than the
            Page-Hinkley change point it replaces.
        crossed_at: Cycle index at which the threshold was first crossed.
    """

    key: tuple[str, str, str, str]
    log_wealth: float = 0.0
    cycles: int = 0
    bets_placed: int = 0
    epoch: int = 0
    fingerprint: str = ""
    prior: PriorState = field(default_factory=PriorState)
    peak_log_wealth: float = 0.0
    rise_cycle: int | None = None
    crossed_at: int | None = None

    def reset_for(self, fingerprint: str) -> EProcessState:
        """Start a fresh epoch: wealth to 0, prior cleared, epoch + 1."""
        return EProcessState(
            key=self.key,
            log_wealth=0.0,
            cycles=0,
            bets_placed=0,
            epoch=self.epoch + 1,
            fingerprint=fingerprint,
            prior=PriorState(),
            peak_log_wealth=0.0,
            rise_cycle=None,
            crossed_at=None,
        )


@dataclass(frozen=True)
class EProcessUpdate:
    """Result of folding one cycle into a process."""

    state: EProcessState
    outcome: EValueOutcome
    was_reset: bool
    crossed_now: bool

    @property
    def reset_notice(self) -> str | None:
        """Report line when an epoch boundary occurred, else None."""
        if not self.was_reset:
            return None
        return (
            f"e-process reset (epoch {self.state.epoch}): suite, embedder, golden "
            "baseline or extractor changed, so accumulated evidence no longer "
            "concerns the same null. Guarantee is alpha per epoch."
        )


def update_process(
    state: EProcessState,
    outcome: EValueOutcome,
    *,
    fingerprint: str,
    alpha_prime: float,
    successes: int,
    trials: int,
    reference: tuple[int, int] | None = None,
) -> EProcessUpdate:
    """Fold one cycle's e-value into the process.

    Order matters: the reset check happens *before* accumulation, so a
    cycle observed under a new fingerprint contributes to the new epoch
    rather than the old one.

    Args:
        state: Prior state.
        outcome: This cycle's e-value (``log_e = 0`` if no bet).
        fingerprint: Current epoch fingerprint.
        alpha_prime: Per-process crossing budget (total ``alpha`` minus the
            coverage budget ``gamma`` spent on nuisance handling).
        successes: Current-cycle successes, folded into the prior *after*
            this cycle's bet — never before.
        trials: Current-cycle trials, likewise.
        reference: Optional ``(successes, trials)`` for the next cycle's
            reference window.

    Returns:
        The update, including whether a reset and/or crossing occurred.
    """
    was_reset = bool(state.fingerprint) and state.fingerprint != fingerprint
    work = state.reset_for(fingerprint) if was_reset else state
    if not work.fingerprint:
        work = replace(work, fingerprint=fingerprint)

    log_wealth = work.log_wealth + outcome.log_e
    threshold = ville_threshold(alpha_prime)
    cycles = work.cycles + 1

    # Onset estimate: where the current unbroken run-up started.
    rise = work.rise_cycle
    if outcome.log_e > 0:
        rise = cycles if rise is None else rise
    elif log_wealth <= 0:
        rise = None

    crossed_now = work.crossed_at is None and log_wealth >= threshold
    prior = work.prior.with_cycle(successes, trials)
    if reference is not None:
        prior = replace(prior, reference_successes=reference[0], reference_trials=reference[1])

    new = EProcessState(
        key=work.key,
        log_wealth=log_wealth,
        cycles=cycles,
        bets_placed=work.bets_placed + (1 if outcome.placed else 0),
        epoch=work.epoch,
        fingerprint=fingerprint,
        prior=prior,
        peak_log_wealth=max(work.peak_log_wealth, log_wealth),
        rise_cycle=rise,
        crossed_at=cycles if crossed_now else work.crossed_at,
    )
    return EProcessUpdate(new, outcome, was_reset, crossed_now)


def state_to_row(state: EProcessState) -> dict[str, Any]:
    """Flatten for the ``eprocess_state`` table."""
    b, f, s, c = state.key
    return {
        "baseline": b,
        "family": f,
        "signature": s,
        "channel": c,
        "log_wealth": state.log_wealth,
        "cycles": state.cycles,
        "bets_placed": state.bets_placed,
        "epoch": state.epoch,
        "fingerprint": state.fingerprint,
        "peak_log_wealth": state.peak_log_wealth,
        "rise_cycle": state.rise_cycle,
        "crossed_at": state.crossed_at,
        "prior_successes": state.prior.successes,
        "prior_trials": state.prior.trials,
        "reference_successes": state.prior.reference_successes,
        "reference_trials": state.prior.reference_trials,
    }
