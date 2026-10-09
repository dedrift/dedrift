"""Withdrawn beta-binomial predictive ratio retained for audit reproducibility.

Replaces the frozen-reference Clopper-Pearson worst-case construction for
one rate process. Motivation, measured on the independent audit: the CP
interval at the battery-wide coverage split (gamma_i = gamma_total / K,
8e-4 at K=24) is so wide that at canary scale it CONTAINS the alternative
for any moderate shift; the worst-case e-value then cannot grow (median
log-wealth DECAYED ~0.07/cycle under a +20pp shift) and detection power at
+5/+10/+20pp over 60 cycles measured 0/30 on every arm.

This module shipped in 0.4/0.5 under an incorrect SAFE/e-value claim. The
ratio is a Bayes factor against a *prior-mixture* null. It is not an e-value
for every fixed value of the composite-null nuisance rate, either conditional
on a realized frozen reference or even after averaging over that reference.
For example, with a fixed reference r=0 of m=5 and one future Bernoulli trial
at p=0.5, its expected value is 2.0417, not at most one.

The implementation remains so old results and the exact counterexample can
be reproduced. :func:`twosample_rate_evalue` now refuses operational use;
the production pipeline accepts only the paid frozen-CP construction. For a
block of s
successes in n trials with past counts (S_1, F_1) on the current side and
(S_0, F_0) in the frozen reference:

    log E = [betaln(s+a1, f+b1) - betaln(a1, b1)]
            - [betaln(s+ap, f+bp) - betaln(ap, bp)]

with a1 = 1+S_1, b1 = 1+F_1, ap = 1+S_0+S_1, bp = 1+F_0+F_1.

Do not use this quantity for sequential testing or alerting.
"""

from __future__ import annotations

from scipy.special import betaln

from dedrift.evalues.base import EValueOutcome, PriorState


def log_beta_binomial_evalue(
    successes: int,
    trials: int,
    prior: PriorState,
) -> float:
    """One cycle's log e-value for a two-sample rate process.

    Args:
        successes: Successes in the current cycle's block.
        trials: Trials in the current cycle's block.
        prior: Completed-cycle sufficient statistics (current side) plus
            the frozen reference counts.

    Returns:
        ``log E_t``; 0.0 (no bet) when there is nothing to compare against.
    """
    if trials <= 0:
        return 0.0
    if prior.reference_trials <= 0:
        return 0.0  # no reference: E = 1, the honest "no bet"
    s1 = prior.successes
    f1 = prior.trials - prior.successes
    s0 = prior.reference_successes
    f0 = prior.reference_trials - prior.reference_successes
    a1, b1 = 1.0 + s1, 1.0 + f1
    ap, bp = 1.0 + s0 + s1, 1.0 + f0 + f1
    s = float(successes)
    f = float(trials - successes)
    log_own = float(betaln(s + a1, f + b1) - betaln(a1, b1))
    log_pooled = float(betaln(s + ap, f + bp) - betaln(ap, bp))
    return log_own - log_pooled


def twosample_rate_evalue(
    successes: int,
    trials: int,
    prior: PriorState,
) -> EValueOutcome:
    """Refuse the withdrawn construction instead of emitting unsafe evidence."""
    del successes, trials, prior
    raise RuntimeError(
        "the beta-binomial 'twosample' construction was withdrawn because it is not "
        "an e-value for the composite null; use dedrift.evalues.rate_evalue"
    )
