"""Two-sample SAFE e-value for rate streams (beta-binomial predictive ratio).

Replaces the frozen-reference Clopper-Pearson worst-case construction for
one rate process. Motivation, measured on the independent audit: the CP
interval at the battery-wide coverage split (gamma_i = gamma_total / K,
8e-4 at K=24) is so wide that at canary scale it CONTAINS the alternative
for any moderate shift; the worst-case e-value then cannot grow (median
log-wealth DECAYED ~0.07/cycle under a +20pp shift) and detection power at
+5/+10/+20pp over 60 cycles measured 0/30 on every arm.

Construction (the SAFE 2x2 e-value in sequential form; Grunwald, de Heide
& Koolen, "Safe Testing", JRSSB 2024; Turner, Ly & Grunwald,
arXiv:2106.02693): for two Bernoulli streams with independent Beta(1,1)
priors under the alternative and a shared Beta(1,1) prior under the null,
the ratio of the current stream's own posterior predictive to the pooled
posterior predictive for each new block is an e-value, and the product
over blocks is an e-process under optional continuation. For a block of s
successes in n trials with past counts (S_1, F_1) on the current side and
(S_0, F_0) in the frozen reference:

    log E = [betaln(s+a1, f+b1) - betaln(a1, b1)]
            - [betaln(s+ap, f+bp) - betaln(ap, bp)]

with a1 = 1+S_1, b1 = 1+F_1, ap = 1+S_0+S_1, bp = 1+F_0+F_1.

Three design notes, all measured before shipping:
* No nuisance interval: the shared null parameter is integrated against a
  prior, so no coverage budget (gamma) is spent, and the e-value grows at
  the posterior-predictive odds rate — not the KL gap to the far edge of a
  99.9% interval. That is what fixes the frozen-CP interval trap (measured
  0/30 detections at +20pp over 60 cycles on the independent audit).
* The pooled denominator LEARNS the alternative as the current stream
  accumulates, so per-cycle growth decays ~1/t after onset and cumulative
  wealth grows only ~log t late on: strong shifts cross in a few cycles,
  mid-rate shifts (p0=0.30) take tens of cycles at canary block sizes.
  The reference-anchored alternative (null predictive from the frozen
  reference only) was implemented and REJECTED: it does not stall, but its
  per-block expectation exceeds 1 when the reference posterior misses the
  true rate (measured E[E] up to 5.8 under the null at p=0.9) — invalid.
* Validity rests on the iid-block null; heavier-than-binomial within-cycle
  overdispersion and persistent (AR(1)) cycle offsets are model deviations
  whose measured null behavior is published (assumption-plus-measurement,
  same stance as the rest of the anytime path).
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
    """EValueOutcome wrapper used by the anytime driver."""
    if trials <= 0 or prior.reference_trials <= 0:
        return EValueOutcome(0.0, False, "no reference or no trials: no bet", ())
    log_e = log_beta_binomial_evalue(successes, trials, prior)
    return EValueOutcome(
        log_e,
        True,
        "two-sample beta-binomial predictive ratio (SAFE 2x2)",
        (),
    )
