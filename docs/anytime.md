# Anytime-valid mode

Every calibration number elsewhere in these docs is **per check**. Monitoring
runs forever, so per-check control is the wrong guarantee: at the measured
1.4% per-check rate, an unchanged agent checked hourly accrues roughly ten
false alerts a month, and nothing in the fixed-sample theory bounds that
accumulation.

Anytime-valid mode replaces it with a statement that holds at every stopping
time:

!!! success "The guarantee"
    Over an **unbounded** monitoring horizon, the probability of ever raising
    a false alert on a stable agent is at most **α = 0.05** — **per epoch**.

```bash
dedrift check --inference anytime
dedrift report --inference anytime --out report.md
```

Default is `--inference fixed`. Read the rest of this page before switching,
because the guarantee is bought with detection power and the exchange rate at
canary scale is steep.

## Measured, both directions

Same seeded stable-agent histories, both inference paths, default suite scale
(18 canaries × 7 repetitions, 24 rate e-processes), 500 runs × 2000 cycles:

| | ever raised a false alert |
|---|---|
| **anytime-valid** (e-processes + e-BH) | **0/500** — Wilson upper bound 0.0076 |
| fixed-sample (BH per check) | **500/500 (100%)**, already by cycle 100 |

The anytime rate is flat at 100, 500, 1000 and 2000 cycles. That flatness is
the whole point: a fixed-sample guarantee decays with use, an anytime-valid
one does not.

And the cost, on the same machinery — cycles from shift onset to alert:

| shift | anytime-valid | fixed-sample |
|---|---|---|
| +2 pp | **not detected** | 100%, median 4 cycles |
| +5 pp | 17%, median 124 cycles | 100%, median 2 |
| +10 pp | ~38%, median 53 cycles | 100%, median 1 |
| +20 pp | 100%, median 6 cycles | 100%, median 1 |

**Read that table before switching modes.** Anytime-valid mode is for
catching real degradation over weeks without accumulating false alarms; it is
not for catching a 2 pp shift this afternoon. If small shifts matter to you
more than lifetime error control, `fixed` is the honest choice — which is why
it remains the default.

The mechanism behind the cost is visible in the construction: handling the
unknown baseline rate requires worst-casing over a coverage interval, which
makes each bet conservative (measured: E[E_t] ≈ 0.73–0.91 per step under the
null). Log-wealth therefore drifts *down* on a stable agent, and small
effects never accumulate enough to cross.

## How the budget decomposes

α splits into two parts, and the split is not cosmetic:

| Component | Default | Pays for |
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

The default γ_total = 0.02 came from sweeping allocations whose claim is
actually 0.05. Within that valid region the trade is one-sided — detection at
+10 pp rises from 13% to 47% as γ_total goes 0.005 → 0.03, with the measured
null rate 0 throughout — so the choice was made on power alone, since
validity does not discriminate. We stop at 0.02 rather than 0.03 because α′
*is* the battery's FDR level, and spending most of the lifetime budget
insuring against a coverage failure never once observed buys little. Both are
configurable.

## "Per epoch" is the part to understand

An **epoch** is a stretch during which the hypothesis and the measuring
instrument are fixed. It ends when any of these change:

- the canary suite
- the pinned embedder
- the golden baseline definition
- the signature extractor
- the judge version (if the judged tier contributes)

At an epoch boundary every e-process resets to zero wealth, and the report
says so prominently.

This is the correct semantics, not a weakened guarantee. Change the suite and
you are testing a different null; evidence accumulated under the old one is
not evidence about the new one, so a guarantee spanning that change would be
*meaningless* rather than stronger. If you re-baseline monthly, your claim is
α per month — and you should know that rather than assume otherwise.

For a genuine unbounded-epoch bound, `epoch_allocation = "geometric"` spends
α·2⁻ᵉ on epoch *e*, so the total across arbitrarily many epochs stays ≤ α. It
costs power, and it is off by default because per-epoch is the honest reading.

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
is measured: 0/500 above.

## Configuration

```toml
[detection]
inference = "fixed"        # or "anytime"

[anytime]
alpha = 0.05               # lifetime, battery-wide
gamma_total = 0.02         # coverage budget; alpha_prime = alpha - gamma_total
tilts = [1.5, 2.0, 3.0]    # symmetrised to {psi, 1/psi}
epoch_allocation = "per_epoch"   # or "geometric"
```

The tilt grid is symmetrised so drift in either direction is covered, and its
support is derived from your materiality bands — which means the bets are
aimed at effects you have declared you care about, and predictability is
structural (a bet can never depend on the cycle it is betting on, because it
comes from configuration).

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
- **A suppressed cycle contributes `E_t = 1` exactly** — not a skipped
  update, and not a stale value carried forward. That is what preserves the
  supermartingale.

## Reading a wealth trajectory

Log-wealth is accumulated evidence since the epoch began.

- **Rising** — evidence against stability. The cycle where the rise began is
  reported as the onset estimate, and it feeds attribution better than the
  Page–Hinkley change point it replaces.
- **Negative** — the bets have lost, which is evidence *for* stability.
  Worth pausing on: the fixed-sample path has no way to express "the agent
  looks actively fine", only "not significant".
- **Crossing log(1/α′)** — the process alone would reject; whether it alerts
  is decided by e-BH across the battery.
