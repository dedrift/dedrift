# Experimental anytime-valid rate monitoring

`dedrift check --inference anytime` monitors rate signatures over an open-ended
sequence of finalized canary cycles. It is intentionally narrower and more
conservative than the fixed-sample path: only rate channels are implemented,
only a frozen golden baseline is admissible, and the decision rule controls the
probability of any false crossing rather than optimizing discovery power.

This is not a live-production or general two-sample monitor. The guarantee is
conditional on the fixed-rate Bernoulli model and the prospective workflow
below; no corrected end-to-end power table is published. A valid but powerless
process can remain quiet through a real regression. Treat it as an experimental
release-gate signal, not a pager.

## Important correction in 0.5.1

Versions 0.4.0 and 0.5.0 made an incorrect claim about the default
`twosample` beta-binomial predictive ratio. That ratio is a Bayes factor for a
prior-mixture null; it is **not** an e-value pointwise for every unknown null
rate after the reference is observed. An exact one-trial counterexample is:

- frozen reference: 0 successes in 5 trials;
- current observation: one Bernoulli trial under the stable null `p = 0.5`;
- expected claimed e-value: `2.0417`, where a valid e-value requires at most 1.

Realistic fixed-reference cases also exceeded the advertised lifetime error
budget. The 0.5.1 release therefore:

- withdraws and rejects `rate_model = "twosample"`;
- removes products of per-family e-values, which were invalid without the
  claimed cross-family independence;
- removes repeated e-BH from the operational decision;
- restores the frozen Clopper--Pearson construction as the only supported
  rate model; and
- uses persistent Ville--Bonferroni crossings over the fixed process pool.

The old 0.4/0.5 null-rate and power headlines for `twosample` are retracted.
They must not be used to size or approve any deployment.

## Current construction

Suppose the epoch contains `K` rate processes. The configured budget is

```text
alpha = alpha_prime + gamma_total
```

The two parts pay for different uncertainty.

1. `gamma_total / K` is the coverage-failure budget for each process's frozen
   reference interval.
2. `alpha_prime / K` is the lifetime crossing budget for each process.

For a rate process, dedrift computes a Clopper--Pearson interval `I` from the
frozen golden counts. For each new cycle it evaluates a symmetric mixture of
Bernoulli likelihood-ratio bets and takes a certified lower bound on its worst
value over `p in I`.
On the event that the true stable rate is in `I`, the resulting wealth is
dominated by a non-negative supermartingale. Ville's inequality then gives

```text
P(process ever crosses K / alpha_prime) <= alpha_prime / K.
```

A union bound covers all `K` processes without assuming they are independent.
A second union bound pays for all reference intervals missing their true rate.
Under the stated stable-rate Bernoulli model, the epoch-wide result is

```text
P(any false alert at any checked cycle) <= alpha_prime + gamma_total = alpha.
```

Crossings are durable for the rest of an epoch. This is both operationally
useful and mathematically necessary: a batched check that folds several unseen
cycles must not erase a crossing that occurred mid-backlog merely because
wealth later fell.

For the general conditional e-value requirement behind this construction, see
[Safe Testing](https://academic.oup.com/jrsssb/article/86/5/1091/7623686).

## Assumptions and boundary

The guarantee is not distribution-free. For each monitored rate it assumes:

- the golden reference is fixed before monitoring;
- the golden cycles are selected by an outcome-independent rule fixed without
  inspecting their observed rates (prefer prospectively designated first
  cycles; never cherry-pick cycles because they looked stable or "good");
- under the null, current observations follow the same fixed Bernoulli rate;
- the bet for a cycle is determined before that cycle is observed; and
- whether a rate observation is available is fixed independently of its value;
- finalized cycles are immutable and processed exactly once.

Hidden provider state, within-cycle correlation, overdispersion, or a rate
that changes while the operator still calls the agent “stable” can violate the
per-process model. Cross-family dependence alone is fine for the Bonferroni
decision, but temporal dependence can still invalidate the underlying wealth
process. Treat robustness simulations as measurements outside the theorem,
not as a replacement theorem.

## Required prospective workflow

The epoch and process pool must exist before any monitored outcome is
observed. Use this order:

```bash
# 1. Collect/finalize reference cycles chosen by a predeclared rule.
dedrift baseline set --first 3

# 2. Declare the epoch and frozen pool. Exit 3 / NO CURRENT OBSERVATION is expected.
dedrift check --inference anytime

# 3. Only now collect the first monitored cycle, then check it.
dedrift canary run --suite canaries.yaml --agent myagent:agent_fn --model provider/model@v1
dedrift check --inference anytime
```

The declaration check intentionally folds no history. Every finalized or open
canary cycle first observed before that declaration watermark is permanently
excluded from the epoch, even if it is finalized later. A later invocation may
fold several unseen cycles only because its epoch, pool, and bets were already
durable before all of them were observed.

`dedrift canary run` and `CanaryRunner.run_cycle(store=...)` durably reserve the
cycle with `Store.begin_cycle` before the first agent call, checkpoint every
record, and finalize only after the expected coordinates are present. A custom
collector must call `Store.begin_cycle(cycle_id, expected_records=...)` before
the first agent output exists. Direct post-hoc `append_many`/`dedrift log`
imports created after epoch declaration are rejected as anytime evidence; they
can remain historical data only after a new baseline and epoch.

Checkpointing prevents completed calls from disappearing after a crash, but
this release does not automatically resume an interrupted runner. The open
cycle deliberately blocks later anytime cycles until an outcome-independent,
auditable abort policy exists; do not delete an inconvenient partial cycle.

## Why the reference is golden-only

A frozen reference creates one confidence event for the whole epoch. A rolling
reference creates a new interval at every cycle. Paying `gamma` once for those
repeated intervals would be incorrect; the failure probability can grow with
the horizon. A future rolling implementation needs a time-uniform confidence
sequence. Until then, rolling comparisons remain available only through the
fixed-sample `dedrift check` path.

## Power: the honest limitation

Worst-casing over a high-coverage interval and splitting two budgets across the
battery is conservative. At ordinary canary scale, the interval can contain a
moderate alternative, in which case wealth will not grow. More monitoring time
does not fix an alternative trapped inside the frozen interval.

Use anytime mode when the lifetime false-alert target is more important than
sensitivity to small shifts. Increase *independent golden reference trials* to
narrow the interval; do not tune away the coverage or multiplicity split.
Run the fixed path alongside it for per-cycle sensitivity, and validate both
modes on your own stable histories and incident replays before using either as
a release gate.

The configured tilts choose the alternatives targeted by the predictable
bets and therefore affect power. They are **not** an observed-effect
materiality gate: an anytime crossing does not assert that a separately
measured practical-null threshold was exceeded. Use the fixed-sample path
when that two-gate interpretation is required.

No replacement power table is published in 0.5.1. A table will return only
after the corrected operational decision is measured end to end at stated
sample sizes.

## Configuration

```toml
[detection]
inference = "fixed"             # opt in with "anytime"

[anytime]
alpha = 0.05                    # epoch-wide lifetime false-alert budget
gamma_total = 0.02              # total frozen-reference coverage budget
tilts = [1.5, 2.0, 3.0]         # symmetrized internally
epoch_allocation = "per_epoch" # or "geometric"
rate_model = "frozen_cp"        # only supported construction
```

`per_epoch` spends a fresh `alpha` after the monitored hypothesis or
instrument changes. `geometric` spends `alpha * 2^-(epoch+1)` and the same
fraction of `gamma_total`, producing a summable bound across arbitrarily many
epochs at a substantial power cost.

An epoch changes when the suite identity, golden baseline, embedder,
signature/inference version, or inference configuration changes. Accumulated
wealth is reset because it concerns a different hypothesis.

## Operational semantics

- Only finalized cycles are visible to inference.
- Cycles are folded in durable first-observation order. If an earlier cycle is
  still open, later finalized cycles wait behind it and the check stays
  non-green (`NO OBSERVATION`) until the contiguous prefix can be processed.
  This prevents completion order from changing the wealth path.
- The process pool is declared once per epoch from reference data and frozen.
- The first check in an epoch is declaration-only; it never replays outcomes
  already present in finalized or open cycles.
- Degraded cycles are still evaluated: choosing whether to bet after seeing
  their outcomes would invalidate the e-process. The degraded label remains
  visible, and a prior crossing remains visible.
- A composition mismatch fails closed without updating state. It cannot be
  converted to `E_t = 1` after observation; repair collection or re-baseline.
- A genuinely unavailable rate contributes `E_t = 1` only under the stated
  outcome-independent availability assumption.
- Every processed cycle and process update commits in one SQLite transaction.
- Re-running without a new finalized cycle is a read-only operation.
- Reports distinguish `NO REFERENCE`, `NO OBSERVATION`, `PARTIAL`, and `NONE`
  from a supported `OK` result.

## Why no production claim is made

Anytime mode still covers rate channels only. It has no scalar or semantic
e-process, no rolling confidence sequence, and no calibrated corrected power
table yet. Each check currently reads/parses the full finalized history, so
runtime is O(history) per check and cumulative compute can become O(cycles²)
over a long deployment. Check rows now retain a compact finalized-revision
watermark rather than duplicating record-ID manifests, but windowed indexed
reads, checkpoints, retention, and scale benchmarks are required before a
production claim of any scale.
Storage retention/redaction are also deployment responsibilities in this
pre-1.0 package. These are explicit limitations, not implied roadmap guarantees.
