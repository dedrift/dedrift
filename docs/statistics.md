# The statistics

This page is the reason dedrift exists. Every claim below is enforced by a
test in CI; where a guarantee is approximate or a procedure is heuristic, the
docs say so plainly.

## The detector battery

One **primary** test per channel enters the FDR pool and can alert;
redundant tests run as **corroboration** only, so multiplicity is spent on
distinct questions rather than three answers to the same one.

| Channel | Primary test | Materiality gate | Corroboration |
|---|---|---|---|
| Location / shape | Two-sample Kolmogorov–Smirnov | KS statistic D ≥ 0.15 (sup-norm CDF distance — catches shape changes with equal means, where Cohen's d ≈ 0). **D is also the reported effect** for this channel, so what gates is what you read; Cohen's d appears on the Welch corroboration row as a location diagnostic | Anderson–Darling, Welch's t (raw p shown, never alert) |
| Tool-call order | Two-sample KS on `tool_order_inversions`, like any other scalar | Same scalar gates (D ≥ 0.15, …) | — |
| Dispersion | Brown–Forsythe (Levene, median-centered) | Robust dispersion ratio (mean abs deviation from the median) ≥ 1.5× either way | — |
| Tail | P95 permutation test (pooled labels, seeded, add-one p) | Relative P95 shift ≥ 10% | — |
| Rates | Two-proportion z, continuity-corrected | Percentage-point thresholds (refusal ≥ 2 pp, …; refusal is pattern-matched phrasing — see fine print) | — |
| Semantic | MMD² (RBF), seeded permutation null ≥ 500 perms, **pooled** median-heuristic bandwidth (permutation-invariant, so the kernel is fixed across relabellings) | Auto-calibrated MMD² floor, same bandwidth as the observation | — |
| Sequential per-cycle means | Page–Hinkley | Flag + onset localizer, not a p-value | — |
| Industry heuristic | PSI, 10 bins frozen from golden | Labeled heuristic (0.1 / 0.25); **never** a test; emitted only above its validity scale (see fine print) | — |

**The tool-call-order channel takes the battery to m ≈ 336 — and the
headline null is family-wise in m.**
`tool_order_inversions` counts Kendall-tau inversions of the tool-call
name sequence against its alphabetical sort — 0 for records with fewer
than two calls. Counts and schema validity say nothing about tool-call
*order*; this is the only Tier-1 channel that sees workflow reordering,
and a battery without it is blind to order reversal by construction (the
independent audit measures 0/30 for such a battery). The alphabetical
reference is arbitrary by
design: a stable agent's inversion count is stationary under any fixed
convention, and the tests compare distributions, so only *changes* in
ordering register. The scalar puts the primary pool at
**m ≈ 336** at the default suite — and the any-alert rate is family-wise:
it grows with m even when every per-test FDR statement is valid, so the
headline null below is always stated at its battery size.

## The gating pipeline — order matters

1. Run all tests; collect p-values. PSI and Page–Hinkley produce flags, not
   p-values, and never enter step 2. Corroboration tests never enter it
   either.
2. **Benjamini–Hochberg adjustment at q = 0.05** across the primary tests in the
   check — one multiplicity family spanning both baselines.
3. Survivors pass a **materiality gate** (per-channel thresholds in the
   table above) — all configurable. Corroboration tests carry no
   materiality verdict at all: the gates are defined on the primary effect
   scales, and e.g. the Anderson–Darling statistic is not commensurable
   with a sup-norm distance, so labeling it "material" would be
   meaningless.
4. Only primary tests passing **both** gates become alerts. Everything else
   appears in the report as "significant, below materiality", "not
   significant", or "corroboration".

Alert fatigue kills monitoring tools; the double gate is deliberate.

This is a prioritization policy, not a formal practical-null FDR procedure.
Ordinary BH needs independence or PRDS, which is not established for this
dependent battery. Filtering equality-null rejections by an observed effect
does not inherit FDR control for “effect is inside the materiality band.” The
default simulation below is the evidence offered here; near-threshold alerts
need confirmation and production suites need their own calibration.

## Calibration is enforced, not asserted

The simulation harness generates agent logs with known ground truth, and CI
runs (on every commit and inside the release pipeline):

- **Per-detector null calibration** — every p-valued detector in the battery
  (KS, AD, Welch, Levene — including under heavy t(3) tails — the P95
  permutation test, the two-proportion z, and MMD) has its false-alarm rate
  measured over hundreds of null simulations against a stated acceptance
  band.
- **Pipeline-level null calibration** — 500 seeded stable-agent runs through
  the full pipeline; the per-check probability of any alert must have a
  Wilson 95% upper bound below **0.065**. The measured rate is **16/500 =
  3.2%, Wilson upper 5.1%**, at m ≈ 336 primaries. The band sits above the
  per-test nominal for exactly the battery-size reason above: the any-alert
  rate is family-wise, and a gate frozen at a smaller battery's rate — the
  same study measures 10/500, upper 3.6%, at m ≈ 300 primaries — would
  fail a correctly behaving m ≈ 336 battery. (A pass/fail over 20 runs —
  the naive version of this gate — would pass 39% of the time even with a
  true 10% alert rate; 500 runs with a reported bound is the honest
  version of the claim.)
  Scale caveat: these runs use 12 canaries × 5 repetitions, not the 18 × 7
  default, for CI runtime. Under a calibrated null the alert rate is
  approximately scale-free, but the materiality gates interact with N — so
  the bound is established, and claimed, at that stated scale.
- **Power checks** — injected shifts (mean, variance, distribution swap,
  equal-mean shape change, scripted model swap) must be detected at
  documented rates, with correct config attribution.

![Measured null false-alarm rates with Wilson intervals and acceptance bands](assets/fig2_calibration.png)

## Detection power: the honest table

Power comes from samples, and canary suites are small. For a rate signature
at a 5% baseline in a family of 30 canaries (two-sided α = 0.05 per test,
*before* FDR, which reduces power further):

| Repetitions N | n per window | +2 pp | +5 pp | +10 pp | +15 pp |
|---|---|---|---|---|---|
| 5 | 150 | 0.07 | 0.28 | 0.78 | 0.97 |
| 7 (default) | 210 | 0.11 | 0.42 | 0.91 | 1.00 |
| 10 | 300 | 0.13 | 0.60 | 0.99 | 1.00 |

**A 2 pp refusal shift is essentially undetectable at this scale.** The
default 2 pp materiality gate is a floor on what may alert, not a promise of
what will be detected. If small shifts matter, grow the relevant canary
family or raise N.

![Power curves for the rate channel](assets/fig3_power.png)

## Cycle effects

Balanced windows give equal *composition*. Exchangeability needs more: the
per-record law must be constant across cycles, which is strictly stronger
than "nothing in the configured stack changed". A hosted model varies
within a fixed version alias, and the reference pools several cycles
against a current window of one, so a shared per-cycle offset shows up as
location and dispersion movement no record-level two-sample test can
distinguish from drift.

Measured on stable agents with a latent per-cycle offset of magnitude σ —
per-check false-alert rate with Wilson 95% intervals, audit-harness scale
(30 canaries × 7 reps, 3-cycle golden), all three columns measured on the
current build. Each project contributes five checks, so the denominators
below are 300 checks at σ = 0 and 200 above it for the record-level
battery (`detection.cycle_effect = "off"`, the default), 150 for `auto`,
and 100 for `auto + persistence=2`:

| σ | record-level battery (default) | cluster-aware (`auto`) | `auto` + persistence = 2 |
|---|---|---|---|
| 0.00 | 3.0% [1.6, 5.6] — 9/300 | 3.3% [1.4, 7.6] — 5/150 | 0.0% [0.0, 3.7] — 0/100 |
| 0.05 | 36.5% [30.1, 43.4] — 73/200 | 34.0% [26.9, 41.9] — 51/150 | 6.0% [2.8, 12.5] — 6/100 |
| 0.10 | 71.0% [64.4, 76.8] — 142/200 | 64.0% [56.1, 71.2] — 96/150 | 31.0% [22.8, 40.6] — 31/100 |
| 0.15 | 89.0% [83.9, 92.6] — 178/200 | 81.3% [74.3, 86.8] — 122/150 | 47.0% [37.5, 56.7] — 47/100 |
| 0.25 | 98.5% [95.7, 99.5] — 197/200 | 94.0% [89.0, 96.8] — 141/150 | 66.0% [56.3, 74.5] — 66/100 |

Each cell is the observed rate `k/n` with its Wilson 95% interval, and the
counts are printed so the rate can be checked against them.

(Two earlier revisions of this table are superseded. A pre-`tool_order`
build measured the record-level column at 2.3% / 33.5% / 70.8% / 87.5% /
97.8%; the rise is expected rather than noise, since the any-alert rate is
family-wise and `tool_order_inversions` takes the battery from m ≈ 300 to
m ≈ 336 primaries. Separately, every cell here was previously printed as
the *centre* of its Wilson interval rather than the observed `k/n`. The
centre is shrunk toward one half, so 0/100 read as 1.8% and 197/200 as
97.6%. The counts above are the measurements.)

At the CI scale (12 × 5, 500 runs, σ = 0) the modes measure 16/500
(3.2%, upper 5.1%) off and 19/500 (3.8%, upper 5.9%) auto — consistent
with the harness-scale σ = 0 row.

The correction (`detection.cycle_effect = "auto"`, opt-in) engages per
channel when the estimated intraclass correlation ρ crosses
`detection.cycle_effect_icc` (default 0.02). ρ is estimated from history
cycles only — never the cycle under test — on the log scale for positive
channels, and is capped at 0.15 so a drifting history cannot inflate the
estimate. On quasi-continuous channels, shape is tested by KS on
within-cycle-standardized values (a shared per-cycle offset cancels
exactly, so record-level power survives), disjoined with a
design-effect-inflated Welch location test and combined by Bonferroni-min,
which is valid under any dependence of the two members. Rate channels
switch to a design-effect-inflated two-proportion z. Dispersion and P95
move to Student-t summaries over per-cycle statistic values — cycle, not
record, is the unit under clustering — with discreteness guards that keep
the record-level p-value when the per-cycle series is near-constant. When
a channel engages, the corrected p replaces the record-level one (each
member is valid in both regimes, so the replacement is never
anticonservative); engagement is reported per channel in the check report.

Read the ladder plainly: the correction **reduces but does not restore
calibration** under wobble, and `detection.alert_persistence = 2` — an
alert must repeat on a fresh current cycle before it fires — roughly
halves the residual again. Neither reaches nominal at canary scale, and
that limit is information-theoretic: a per-cycle offset at these sample
sizes *is* a real distributional change, not noise a record-level test can
see through. The 16/500 headline is measured at σ = 0 and describes that
regime only. Pinned or self-hosted models: leave the default `off`. For
mildly wobble-prone hosted models the better option is [anytime-valid
mode](anytime.md), whose observed ever-alert rate stays within its
lifetime budget under iid wobble up to σ = 0.25 (measured 8/500;
persistent AR(1) offsets at σ = 0.25, φ = 0.9 exceed it at 7.2% — the
published boundary). But its *guarantee* lapses earlier than its alert
rate does — the e-value property is already unresolved at σ = 0.10 — so
above that magnitude neither path is covered by a bound, and `auto` +
`alert_persistence = 2` is the configuration whose behaviour is measured
there, at the rates above.

The correction is bought with power — detection within 3 post-drift
checks, drift injected at cycle 5, audit harness (record-level cells at
30–60 reps, cluster-aware cells at 10–20; small-n, stated rather than
smoothed over):

| injected shift | record-level (`off`) | cluster-aware (`auto`) |
|---|---|---|
| gross model swap, multi-channel | 30/30 | 10/10 |
| output length ×1.10 (d ≈ 0.77) | 31/60 (52%) | 8/20 (40%) |
| refusal +20 pp | 24/60 (40%) | 4/20 (20%) |
| variance only | 43/50 (86%) | 18/20 (90%) |
| shape skew, moments matched | 14/50 (28%) | 6/20 (30%) |

The gross swap is certain under either battery; moderate single-channel
shifts are where the cluster-aware battery pays. Tool-call order reversal
is the order channel's own measurement: **8/20 first-check channel
alerts** in the same harness — at a scale where "detected" means about
two first checks in five — while a battery without the channel measures
**0/30**, blind to order by construction. That hole is what the channel
exists to close. (This probe was rerun at 20 replicates for the full
adversarial matrix; the 5/10 published with the 0.4.0 release notes was
the earlier 10-replicate run and is superseded.)

## Fine print, stated plainly

- **The 500-run pipeline bound covers the structural channels only.**
  Embeddings are excluded from those runs for runtime, so the semantic
  channel (family MMD and semantic displacement) has per-detector
  calibration but **no pipeline-level measurement** — and that is the
  channel carrying the largest effect in the [case study](case-study.md).
  The gap is material and stated here rather than left to be inferred from
  m = 336 versus m = 384.
- **Suppressed evidence fails closed.** A fully suppressed check reports
  `NO VALID COMPARISON`; partial suppression reports `PARTIAL COVERAGE`.
  Both return CLI exit 3, never the green exit 0.

- **Balance is checked; exchangeability is measured, not assumed.**
  Every check
  verifies this per (baseline, family): if a canary's records vanish from
  one window (a timeout, a partial run, a suite edit), the family mixture
  shifts and KS would fire on a missing-data artifact — so the comparison
  is **suppressed and reported as COMPOSITION MISMATCH** instead of drift.
  Power against shifts confined to a few canaries remains lower than for
  family-wide shifts. Exchangeability across cycles is the stronger
  requirement, and its measured failure modes — with the cluster-aware
  correction and its costs — are in [Cycle effects](#cycle-effects) above.
- **The guarantee on this page is per check, not per lifetime.**
  Everything above bounds the probability that *one* check on a stable
  agent raises an alert. Monitoring runs continuously, and expected
  counts add regardless of how checks depend on each other (linearity of
  expectation; only the "at least one" probability needs independence).
  At the measured 3.2% per-check rate, a stable agent checked hourly
  accrues roughly **0.8 false alerts per day, ~23 per month** (720 ×
  0.032). BH adjusts multiplicity *within* a check under its assumptions;
  nothing in the batch machinery controls accumulation *across*
  sequence of them.

    **Sequential control exists**: [anytime-valid mode](anytime.md)
    replaces the per-check statement with a lifetime one — measured 2
    false alerts in 500 stable-agent runs of 2000 cycles each (0.4%,
    Wilson upper 1.5%), against 100% for this path on identical
    histories. It is not
    the default, because it buys that guarantee with detection power: a
    +2 pp shift stays undetectable (0/100) and +10 pp is caught in 89/100
    runs, at a median of 50 cycles. Both figures are on that page. If you
    stay on the fixed-sample
    path, the honest operating advice is unchanged — treat one alert as
    evidence to investigate rather than an incident, require persistence
    across consecutive cycles before paging anyone
    (`detection.alert_persistence = 2` makes that the checker's
    semantics), and choose your check
    frequency knowing the arithmetic above.
- **Where the KS gate binds.** The two-sample critical value at raw
  α = 0.05 is D_crit ≈ 1.36·√((n+m)/(n·m)). For equal arms the default
  gate (D ≥ 0.15) sits below D_crit until n ≳ 165 per arm — and BH only
  raises the bar. At typical per-family scale (say 21 current vs 105
  pooled reference, D_crit ≈ 0.32) significance is therefore the stricter
  filter, and the KS gate cannot be the one that fires; its job is to stop
  trivially significant D from alerting at large n. A config-aware test
  recomputes both claims from the shipped default, so a threshold change
  that breaks this description fails CI.
- **The flag channel (Page–Hinkley, PSI) is uncalibrated and carries no
  multiplicity control — treat flags as diagnostics, never as alerts.**
  PH runs on every (family, signature) stream (48 at the measured scale: 8 scalar signatures x 6 families), so
  per-stream rates compound: the 500-run null study measures, at the
  48-stream battery, **69.8% of stable checks showing at least one flag**
  (per-stream rate 2.7%, Wilson [2.5%, 2.9%] over 24,000 draws;
  1−(1−0.027)⁴⁸ = 0.73 against the observed 0.698 — the residual gap is
  the streams sharing records). That
  number is printed here deliberately — flags never alert, they exist to
  localize onsets for attribution, and the report labels them as such.
  Cross-stream correction for the flag channel is on the roadmap.
- **PSI is refused where it cannot mean anything.** PSI between two finite
  samples of the *same* distribution is not zero — to first order
  E[PSI] ≈ (B−1)·(1/n_ref + 1/n_cur) — and that asymptotic figure is itself optimistic: simulating the shipped implementation at (30, 10) gives a measured E[PSI] of **3.10**, with PSI calling unchanged data a "major shift" **100%** of the time. At canary scale (10 bins,
  tens of records) exceeds the 0.25 "major" folk threshold from sampling
  noise alone; before this guard, PSI flagged **100% of stable checks**.
  The pipeline therefore emits PSI only when its null expectation is below
  half the "moderate" threshold (roughly n ≳ 360 equal-arm samples). PSI
  remains what it always was — a large-sample production-traffic index —
  and simply does not pretend to work below its domain of validity.
- **The refusal signature is pattern-matched phrasing, not a behavior
  class.** The extractor matches refusal *wording*; a model that
  paraphrases its refusals can move the measured rate in either direction
  while true refusal behavior moves the other way — the independent audit
  measured a **−27 pp** change in the measured rate on a **+20 pp** true
  shift toward paraphrased refusals. Treat the alert's direction as a
  pointer to read the outputs, not as a verdict on what the model is
  doing.
- **Anderson–Darling and Welch are corroboration, not evidence.** They test
  the same location/shape hypothesis as KS on the same data; admitting them
  to the FDR pool would roughly double m — halving every BH threshold — for
  no informational gain. Their raw p-values are printed for context and can
  never alert. (AD's p is additionally capped by SciPy's asymptotics to
  [0.001, 0.25].)
- **BH is used without a PRDS claim.** PRDS is not available here:
  Dobriban (2026, arXiv:2607.12208) constructs
  correlated two-sided Gaussian p-values for which BH provably exceeds its
  nominal level, disproving a twenty-year-old conjecture that would
  otherwise have covered positively correlated two-sided tests. Our primaries are
  two-sided and computed on shared data — the exact configuration. The
  pipeline-level measured alert rate is therefore the operative guarantee
  for the default path; Benjamini–Yekutieli is on the roadmap for users who
  want a theorem, and `--inference anytime` uses e-BH, which holds under
  arbitrary dependence.
- **Page–Hinkley** (λ = 12, δ = 0.3 in robust-scale units, centre and scale
  estimated **causally** from an expanding window of data strictly before
  each step): the idealized null crossing bound is ~0.15% per stream, but
  because centre and scale are estimated from few observations the measured
  rate is **8.5% per stream on 30-cycle histories and 11.3% on 60-cycle
  ones** (8000 draws). The causal estimation is load-bearing: an estimator
  that standardises with the whole stream, including cycles *after* the
  alarm, reads its own future and reports a falsely reassuring rate near
  1.5%. This is why PH is a labelled diagnostic that can never alert.
  Two edge cases are engineered explicitly: a missing (family, cycle)
  would reindex the stream with NaN from which the running mean never
  recovers, so streams mask to the family's observed cycles; and
  near-constant discrete streams can produce absurd statistics
  (~2.6e15), held at ~0 by a scale floor at the window's
  floating-point resolution.
PH alarms localize onsets for attribution;
  they never alert: only primary tests can.
- **MMD² materiality floor** is auto-calibrated per (baseline, family) as the
  95th percentile of MMD² between pairs of the baseline's own cycles — an
  empirical null from known-same-distribution data. With fewer than five
  reference cycles the auto-floor is `UNCALIBRATED`; MMD remains visible but
  cannot alert. The config accepts an explicit non-negative override.
- **Permutation resolution is sized for the BH family.** Add-one Monte Carlo
  p-values have minimum `1/(B+1)`. A configured `B=500` cannot reach the
  rank-one BH cutoff in a 336-test family, so the checker automatically raises
  `B` to at least `ceil(m_upper/q)-1` and records both configured and effective
  counts. P95 permutations are streamed in bounded-memory chunks.
- **Reproducibility.** All randomness is seeded; permutation seeds are
  recorded in the report. Same logs + same config ⇒ the same report up to
  the recorded check timestamp: every statistic, p-value, effect, flag,
  and verdict is byte-identical; the timestamp line is the sole
  run-dependent field.
