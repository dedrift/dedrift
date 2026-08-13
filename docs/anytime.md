# Anytime-valid mode

Every calibration number elsewhere in these docs is **per check**. Monitoring
runs forever, so per-check control is the wrong guarantee: at the measured
3.2% per-check rate, an unchanged agent checked hourly accrues roughly 23
false alerts a month, and nothing in the fixed-sample theory bounds that
accumulation.

Anytime mode replaces it with lifetime-oriented rate e-processes:

!!! info "The target and its boundary"
    Per-process optional-stopping control and per-check e-BH control are
    proven. The full repeated dependent battery targets an unbounded-horizon
    false-alert budget of **α = 0.05 per epoch**, but that trajectory-wide
    statement relies on the causal condition described below. It is measured,
    not presented as an unconditional theorem.

```bash
dedrift check --inference anytime
dedrift report --inference anytime --out report.md
```

Default is `--inference fixed`. Read the rest of this page before switching,
because lifetime-oriented control is bought with detection power and the exchange rate at
canary scale is steep.

## Measured, both directions

Same seeded stable-agent histories, both inference paths, default suite scale
(18 canaries × 7 repetitions; 24 declared rate e-processes = 6 families × 4 rate
signatures on the **golden** baseline), 500 runs × 2000 cycles.

The streams are simulated *dependent* — signatures of a family are computed
from the same records — because independent streams would measure the one
case in which the governing result already holds, and so would prove
nothing about ours.

| | ever raised a false alert |
|---|---|
| **anytime-valid** (two-sample e-processes + e-BH) | **2/500 (0.4%)** — Wilson upper bound 0.015 |
| fixed-sample (BH per check) | **419/500 (84%)** by cycle 100; **500/500** by cycle 1000 |

The anytime column is the `twosample` rate model (the default) at the
2000-cycle horizon; the fixed column is the record-level battery,
`cycle_effect = "off"` — also the default. Two events in 500 runs is
consistent with a true lifetime rate up to about 1.5% — inside the
α = 0.05 budget,
and not a claim of exact flatness. What the contrast establishes is the
difference in kind — a fixed-sample guarantee decays with
use, and this one has not been observed to breach its budget.

And the cost, on the same machinery — cycles from shift onset to alert.
Only the 6 refusal streams shift (the realistic case, since a regression
rarely moves every channel at once); onset at cycle 10, 100 runs × 400
cycles:

| shift | anytime-valid (twosample, pooled battery) | fixed-sample |
|---|---|---|
| +2 pp | **0/100 — never detected** | 100%, median 17 cycles |
| +5 pp | 1/100, median 52 | 100%, median 7 |
| +10 pp | **89/100**, median 50 cycles | 100%, median 2 |
| +20 pp | 100/100, median 17 cycles | 100%, median 1 |

Percentages are the fraction of runs that detect **at all** within 400
cycles; medians are conditional on detecting. Below about +10 pp the
wealth process grows too slowly to cross inside any practical horizon —
the right word is *inconsistency*, not delay. The battery includes one
**pooled process per rate signature** (the product of the six per-family
e-values — an e-value under cross-family independence, inside e-BH's
arbitrary-dependence guarantee; battery 24 → 28): it is what roughly
doubles family-wide drift detection (the same study without it measures
49/100, median 66, at +10 pp), while the per-family processes keep
single-family resolution.

**Read that table before switching modes.** Anytime-valid mode is for
catching real degradation over weeks without accumulating false alarms; it is
not for catching a 2 pp shift this afternoon. If small shifts matter to you
more than lifetime error control, `fixed` is the honest choice — which is
why it is the default.

The mechanism behind the cost is visible in the construction: the pooled
denominator *learns the alternative* as the current stream accumulates, so
per-cycle growth decays ~1/t after onset and cumulative wealth grows only
~log t late on. Strong shifts cross in a few cycles; mid-rate ones
(p₀ = 0.30) take tens of cycles at canary block sizes — at a 60-cycle
harness horizon +10 pp at p₀ = 0.30 was never caught (0/20), while +20 pp
went 4/20 (median 26) and a low-baseline +10 pp 5/20 (median 28).

## The rate model: twosample (default) vs the Clopper–Pearson construction (frozen_cp)

Each rate process is a **two-sample SAFE e-value** — the SAFE 2×2
construction in sequential form (Grunwald, de Heide & Koolen, [Safe
Testing, JRSSB 2024](https://academic.oup.com/jrsssb/article/84/3/822/7056146);
Turner, Ly & Grunwald, [arXiv:2106.02693](https://arxiv.org/abs/2106.02693)).
Two Bernoulli streams carry independent Beta(1,1) priors under the
alternative and a shared Beta(1,1) prior under the null; the ratio of the
current stream's own posterior predictive to the pooled posterior
predictive for each new cycle block is an e-value, and the product over
blocks is an e-process under optional continuation. For a block of s
successes in n trials, with (S₁, F₁) the current stream's completed-cycle
counts and (S₀, F₀) the frozen reference counts:

```text
log E = [betaln(s + a1, f + b1) − betaln(a1, b1)]
      − [betaln(s + ap, f + bp) − betaln(ap, bp)]

a1 = 1 + S₁,  b1 = 1 + F₁,   ap = 1 + S₀ + S₁,  bp = 1 + F₀ + F₁
```

No nuisance interval is needed: the shared null parameter is integrated
against the prior, so **no coverage budget is spent** and the e-value
grows at the posterior-predictive odds rate rather than the KL gap to the
far edge of a worst-case interval.

**The alternative construction, and its measured limit.** `frozen_cp`
worst-cases over a frozen Clopper–Pearson interval for the unknown
baseline rate, with the
coverage budget split across the battery (γᵢ = γ_total ÷ K = 8×10⁻⁴ at
K = 24). At canary scale that interval is so wide that it *contains the
alternative* for any moderate shift, so the worst-case e-value cannot
grow: the independent audit measures **0/30 detections at +5/+10/+20 pp
over 60 cycles at p₀ = 0.30**, with median log-wealth *decaying* ≈0.07
per cycle under a +20 pp shift. The construction ships as a documented
ablation — its measured power profile is the evidence behind the
twosample default, not a mode to run in production — and `gamma_total`
and `tilts` are meaningful only there.

**The saturation caveat.** The pooled denominator learns the alternative
as drifted cycles accumulate, so per-cycle growth decays ~1/t after onset
— fast crosses for strong shifts, tens of cycles for mid-rate ones, and a
+10 pp shift at p₀ = 0.30 is beyond a 60-cycle horizon entirely (the
400-cycle table above is the operative measurement). The
reference-anchored alternative that would avoid the stall was implemented
and **rejected**: it is not an e-value when the reference posterior misses
the true rate (measured E[E] up to 5.8 under the null).

**The validity boundary, measured.** Validity rests on the iid-block null,
and two different things fail at two different points.

The **e-value property itself** — E[E_t | F_{t-1}] ≤ 1, without which
Ville's inequality has no premise — is what fails first. Under a shared
per-cycle offset it is **already unresolved against 1 at σ = 0.10**
(E[M_T] = 1.10 over 45,000 replicated trajectories) and **clearly fails by
σ = 0.25**. Canary heterogeneity is safe in the other direction: unequal
fixed rates within a family make the block Poisson-binomial, which is
underdispersed relative to the binomial the null assumes, so the
construction only gets more conservative.

The **observed alert rate** lags that failure, which is why watching it is
not enough: under iid cycle wobble it stays within the α = 0.05 budget up
to σ = 0.25 (8/500 = 1.6%, Wilson upper 3.1%) — a regime in which the
guarantee has already lapsed. The persistent AR(1) regime σ = 0.25,
φ = 0.9 measures **36/500 = 7.2% (upper 9.8%) — above the 5% budget**.

So: if your endpoint wobbles at σ ≳ 0.10, do not read the anytime bound as
covering you, whatever the alert rate looks like. Details under
[Robustness to provider-side
wobble](#robustness-to-provider-side-wobble).

## How the budget decomposes

With the default `twosample` rate model there is no nuisance interval, so
**no coverage budget is spent and the e-BH level is the full per-epoch
alpha: α′ = α = 0.05**. The split below applies only to the `frozen_cp`
construction, where the unknown baseline rate is paid for out
of α:

| Component | frozen_cp default | Pays for |
|---|---|---|
| α — lifetime, battery-wide | 0.05 | the whole claim |
| α′ — e-BH level | 0.03 | multiplicity across the processes |
| γ total — coverage | 0.02 | the unknown baseline rate |
| γ per process | γ_total ÷ K | one process's interval |

**Why γ is divided by the pool size.** e-BH controls FDR only if every input
is a genuinely valid e-value. A process whose coverage interval misses the
true rate is not one, so coverage failures union-bound across the battery:

> P(ever a false alert) ≤ α′ + Σᵢ γᵢ

At 24 processes, using γ = 0.02 *per process* would claim 0.05 while
delivering 0.51. dedrift computes γᵢ = γ_total ÷ K from the live pool, so
changing your suite cannot silently stale the arithmetic.

The frozen_cp default γ_total = 0.02 came from sweeping allocations whose
claim is actually 0.05. Within that valid region the trade is one-sided —
detection at +10 pp rises from 24% to 90% as γ_total goes 0.005 → 0.03,
with the measured null rate 0 throughout — so the choice was made on power
alone, since validity does not discriminate. We stop at 0.02 rather than
0.03 because α′ *is* the battery's FDR level, and spending most of the
lifetime budget insuring against a coverage failure never once observed
buys little. Both are configurable — and both are inert under `twosample`,
which the audit's 0/30 measurement of the alternative construction makes
the recommended default.

## "Per epoch" is the part to understand

An **epoch** is a stretch during which the hypothesis and the measuring
instrument are fixed. It ends when any of these change:

- the canary suite
- the pinned embedder
- the golden baseline definition
- the signature extractor
- the anytime α/γ allocation, tilt grid, or allocation mode

At an epoch boundary every e-process resets to zero wealth, and the report
says so prominently. Epoch instances are globally monotone: changing A→B→A
creates three epochs, never reuses A's old pool, ledger, or budget. A changed
epoch begins *after* the cycle visible when it is declared; historical cycles
are not replayed under a monitoring configuration chosen later.

This is the correct semantics, not a weakened guarantee. Change the suite and
you are testing a different null; evidence accumulated under the old one is
not evidence about the new one, so a guarantee spanning that change would be
*meaningless* rather than stronger. If you re-baseline monthly, your claim is
α per month — and you should know that rather than assume otherwise.

For a summable cross-epoch target, `epoch_allocation = "geometric"` allocates
both α′ and γ_total by `2⁻⁽ᵉ⁺¹⁾` on globally monotone epoch *e*. Allocating
only α′ while re-spending γ would not be summable. It costs power and is off
by default.

## What is proven and what is measured

Kept separate deliberately.

| Statement | Status |
|---|---|
| Per process, P(ever crossing) ≤ α′ at any stopping time | **proven** (Ville) |
| Per check, FDR ≤ α′ under arbitrary dependence among e-values | **proven** ([Wang & Ramdas](https://academic.oup.com/jrsssb/article/84/3/822/7056146)) |
| Over the trajectory, FDR ≤ α′ applying e-BH every cycle | **assumed + measured** |

That last row is the honest gap. Applying e-BH repeatedly to running
e-processes is anytime-valid under a causal condition — no unobserved
confounding from the past — established by the [stopped e-BH
result](https://arxiv.org/abs/2502.08539). Our streams are dependent (two
baselines share the current cycle; all channels share the golden sample), so
the condition is not automatic. It is plausible here, because every stream is
a function of the same observable cycle history, but we have not proven it for
this battery. The candidate violation worth naming is unobserved
provider-side state — a rolling deployment, load variation — correlating
streams through time. Instead of asserting the guarantee, the realised rate
is measured: 2/500 above, with the wobble boundary below.

## Configuration

```toml
[detection]
inference = "fixed"        # or "anytime"

[anytime]
alpha = 0.05               # lifetime, battery-wide
rate_model = "twosample"   # default; "frozen_cp" is the Clopper–Pearson interval construction (an ablation)
gamma_total = 0.02         # frozen_cp only: coverage budget; alpha_prime = alpha - gamma_total
tilts = [1.5, 2.0, 3.0]    # frozen_cp only: symmetrised to {psi, 1/psi}
epoch_allocation = "per_epoch"   # or "geometric"
```

The tilt grid is symmetrised so drift in either direction is covered, and its
support is a **fixed configured constant**. Predictability is therefore
structural — a bet can never depend on the cycle it is betting on, because it
comes from configuration. A helper (`tilt_from_materiality`) exists to derive
a tilt from a materiality band, which would aim the bets at effects you have
declared you care about, but the shipped default does not call it. The grid
is a constant, not an aimed one. (All of this configures `frozen_cp`; the
twosample e-value has no tilt grid at all.)

Anytime mode has **no post-hoc materiality gate**. Its
alerts test rate stability with configured bets, while fixed-mode alerts also
apply observed-effect thresholds. The two modes do not currently represent
the same practical hypothesis; use anytime for sequential rate surveillance
and fixed mode for the broader, effect-screened diagnostic battery.

## Current scope

Only the **rate channel** (refusal, format validity, schema conformance,
error rate) is implemented in anytime mode. A verified single channel is worth
more than four unverified ones; the scalar and semantic channels need
constructions of their own rather than a reuse of this one, and until they
exist, anytime mode monitors less than fixed mode does. Run both.

Two further limitations, stated rather than discovered:

- **Pool membership freezes at epoch start.** Only combinations with
  reference data at that moment are admitted. A signature that gains data
  mid-epoch waits for the next epoch; one that loses data stays in the pool
  contributing exactly `E_t = 1`. This is a consequence of predictability,
  not an oversight: membership decided from the current cycle would make the
  bet depend on the data it is betting on.
- **A suppressed cycle contributes `E_t = 1`** — not a skipped
  update, and not a stale value carried forward. That preserves the
  supermartingale *under a non-informative-dropout condition*: suppression is
  decided after the cycle is observed, so `E[E_t | past] <= 1` requires that
  whether a cycle was suppressed carries no information about its outcome.
  For `had_error` those are nearly the same event — a real caveat, not a
  formality.

## Reading a wealth trajectory

Log-wealth is accumulated evidence since the epoch began.

- **Rising** — evidence against stability. The cycle where the rise began is
  reported as the onset estimate, and it feeds attribution better than the
  Page–Hinkley change point it replaces.
- **Negative** — the configured bets have lost. This is not affirmative
  evidence of stability and must not be presented as certification.
- **Crossing log(1/α′)** — the process alone would reject; whether it alerts
  is decided by e-BH across the battery.

## Robustness to provider-side wobble

The anytime path was run on stable agents while a shared per-cycle
offset was injected — the configured stack unchanged, but the per-record law
drifting between cycles, which is what a hosted endpoint does on its own.
`phi` is the AR(1) coefficient of that offset: `phi = 0` is memoryless,
`phi > 0` makes it persist. Ever-alerted, 500 runs × 2000 cycles, twosample
rate model (the fixed path degrades far faster under the same injection —
its ladder is on [the statistics page](statistics.md#cycle-effects)):

| σ | φ | ever alerted |
|---|---|---|
| 0.00 | 0 | 2/500 — 0.4%, Wilson upper 1.5% |
| 0.10 | 0 | 2/500 |
| 0.25 | 0 | 8/500 — 1.6%, Wilson upper 3.1% |
| 0.10 | 0.90 | 4/500 |
| 0.25 | 0.90 | **36/500 — 7.2%, Wilson upper 9.8%** |

Harness cross-checks at a 60-cycle horizon agree at small n: 0/40 with no
wobble (upper 8.8%), 1/30 at σ = 0.15 (upper 16.7%). (Published here at
first as 0/60 and 1/40; those denominators did not match the run the
figure and the paper report, and are superseded.)

Read the last row against its matched controls. At σ = 0.25 the rate goes
8/500 (φ = 0) → 36/500 (φ = 0.90): monotone in persistence at fixed
magnitude, which is what the theory predicts if the causal condition is
what binds — persistent offsets are exactly the configuration in which
unobserved provider-side state can correlate streams through time. And
7.2% **exceeds the α = 0.05 budget**. That is the measured boundary of the
guarantee, published rather than smoothed over: the observed rate stays
inside budget under memoryless wobble up to σ = 0.25, and does not under
strongly persistent wobble of that magnitude. Read that alongside the
validity boundary above — the observed rate staying inside α up to
σ = 0.25 is not the same as the guarantee holding there, and it does not:
the e-value property is already unresolved at σ = 0.10.

Why persistence and not just magnitude: an offset redrawn independently each
cycle has no past, so however large it is it cannot violate a condition
about confounding *from the past*. Measuring that and calling it reassurance
would be measuring the wrong thing.

## The assumption behind the reference

Under `frozen_cp`, γ buys a Clopper–Pearson interval that covers the true
reference rate, and the per-process construction is valid *on that coverage
event*. Clopper–Pearson coverage assumes the reference is a binomial sample
from the rate you are trying to bound. The twosample default spends no
coverage budget — but it compares against the same frozen reference counts,
and the selection problem is identical either way: a golden baseline is a
set of cycles **you declared known-good** — a selected sample. Selecting
toward quiet cycles moves the reference away from the true stable rate in
exactly the direction that turns "drift vs reference" into "reference vs
truth", and no rate model repairs that.

Practical consequence: freeze the baseline on the *first* cycles you
collected rather than the ones that looked best. `dedrift baseline set
--first 3` exists for that. A hand-picked baseline weakens the guarantee by
an amount nobody can quantify, and the 2/500 measurement above says nothing
about it — that study draws its reference as a clean binomial sample, so
the reference is unbiased there by construction.
