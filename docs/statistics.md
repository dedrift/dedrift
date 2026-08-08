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
| Tool-call order (new in v0.4.0) | Two-sample KS on `tool_order_inversions`, like any other scalar | Same scalar gates (D ≥ 0.15, …) | — |
| Dispersion | Brown–Forsythe (Levene, median-centered) | Robust dispersion ratio (mean abs deviation from the median) ≥ 1.5× either way | — |
| Tail | P95 permutation test (pooled labels, seeded, add-one p) | Relative P95 shift ≥ 10% | — |
| Rates | Two-proportion z, continuity-corrected | Percentage-point thresholds (refusal ≥ 2 pp, …; refusal is pattern-matched phrasing — see fine print) | — |
| Semantic | MMD² (RBF), seeded permutation null ≥ 500 perms, **pooled** median-heuristic bandwidth (permutation-invariant, so the kernel is fixed across relabellings) | Auto-calibrated MMD² floor, same bandwidth as the observation | — |
| Sequential per-cycle means | Page–Hinkley | Flag + onset localizer, not a p-value | — |
| Industry heuristic | PSI, 10 bins frozen from golden | Labeled heuristic (0.1 / 0.25); **never** a test; emitted only above its validity scale (see fine print) | — |

**The battery grew in v0.4.0, and the headline null moved with it.**
`tool_order_inversions` counts Kendall-tau inversions of the tool-call
name sequence against its alphabetical sort — 0 for records with fewer
than two calls. Counts and schema validity say nothing about tool-call
*order*; this is the only Tier-1 channel that sees workflow reordering,
which the independent audit measured as completely invisible to v0.3.1
(0/30, by construction). The alphabetical reference is arbitrary by
design: a stable agent's inversion count is stationary under any fixed
convention, and the tests compare distributions, so only *changes* in
ordering register. The new scalar takes the primary pool from m ≈ 300 to
**m ≈ 336** at the default suite — and the any-alert rate is family-wise:
it grows with m even when every per-test FDR statement is valid. That
growth, not a calibration slip, is why the headline null below moved from
its v0.3.1 value.

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
  Wilson 95% upper bound below **0.065**. v0.4.0 measures **16/500 = 3.2%,
  Wilson upper 5.1%**. The v0.3.1 measurement — 10/500, upper 0.0364 —
  stays on the record as the historical value at m ≈ 300 primaries; the
  gate was re-banded from 0.05 for exactly the battery-growth reason above,
  since a gate frozen at the m ≈ 300 rate would fail a correctly behaving
  m ≈ 336 battery. (A pass/fail over 20 runs — the
  original spec — would pass 39% of the time even with a true 10% alert
  rate; 500 runs with a reported bound is the honest version of the claim.)
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
(30 canaries × 7 reps, 3-cycle golden), 30 reps per level for `auto` and
20 for `auto + persistence=2`; the `off` column is the original v0.3.1
audit at 100–300 reps per level, and `detection.cycle_effect = "off"` (the
default) reproduces that record-level battery exactly:

| σ | off (v0.3.1 battery) | auto | auto + persistence = 2 |
|---|---|---|---|
| 0.00 | 2.3% [1.6, 3.1] | 4.5% [1.4, 7.6] | 1.8% [0.0, 3.7] |
| 0.05 | 33.5% [29.4, 37.6] | 34.4% [26.9, 41.9] | 7.6% [2.8, 12.5] |
| 0.10 | 70.8% [66.9, 74.8] | 63.7% [56.1, 71.2] | 31.7% [22.8, 40.6] |
| 0.15 | 87.5% [85.4, 89.5] | 80.6% [74.3, 86.8] | 47.1% [37.5, 56.7] |
| 0.25 | 97.8% [96.6, 99.1] | 92.9% [89.0, 96.8] | 65.4% [56.3, 74.5] |

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
wobble-prone hosted models the valid default is [anytime-valid
mode](anytime.md), which stays within its lifetime budget under iid wobble
up to σ = 0.25 (measured 5/500; persistent AR(1) offsets at σ = 0.25,
φ = 0.9 exceed it at 7.2% — the published boundary). `auto` + persistence
is the fixed-path option, at the rates above.

The correction is bought with power — detection within 3 post-drift
checks, drift injected at cycle 5, audit harness (v0.3.1 cells at 30–60
reps, v0.4.0 cells at 10–20; small-n, stated rather than smoothed over):

| injected shift | off (v0.3.1) | auto (v0.4.0) |
|---|---|---|
| gross model swap, multi-channel | 30/30 | 10/10 |
| output length ×1.10 (d ≈ 0.77) | 31/60 (52%) | 8/20 (40%) |
| refusal +20 pp | 24/60 (40%) | 4/20 (20%) |
| variance only | 43/50 (86%) | 18/20 (90%) |
| shape skew, moments matched | 14/50 (28%) | 6/20 (30%) |
| tool-call order reversal | 0/30 (blind) | 5/10 first-check alerts |

The gross swap remains certain; moderate single-channel shifts are where
the cluster-aware battery pays. The last row is the new channel: order
drift was invisible to v0.3.1 by construction and is now detected — at a
scale where "detected" means half of first checks.

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
  requirement, and its measured failure modes — with the v0.4.0 correction
  and its costs — are in [Cycle effects](#cycle-effects) above.
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

    **Sequential control now exists**: [anytime-valid mode](anytime.md)
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
  per-stream rates compound: measured on the 500-run null study at the
  v0.3.1 battery (42 streams), **68.6% of stable checks showed at least one flag** (up from 56% once Page-Hinkley stopped standardising itself with data from after the point it was judging). With 48 streams the figure can only be higher; it has not been re-measured. That
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
- **BH is used without a PRDS claim.** This page previously said PRDS was
  "believed to cover" positively correlated two-sided tests. That belief is
  no longer available: Dobriban (2026, arXiv:2607.12208) constructs
  correlated two-sided Gaussian p-values for which BH provably exceeds its
  nominal level, disproving a twenty-year-old conjecture. Our primaries are
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
  ones** (8000 draws). An earlier version of this page said 1.5%; that came
  from an estimator that used the whole stream, including cycles *after* the
  alarm — a sequential detector reading its own future. Removing the
  look-ahead raised the honest rate sixfold. This is why PH is a labelled
  diagnostic that can never alert.
  v0.4.0 fixed two further PH defects the audit surfaced: a missing
  (family, cycle) used to reindex the stream with NaN and the running mean
  never recovered, so streams now mask to the family's observed cycles; and
  near-constant discrete streams used to render absurd statistics
  (~2.6e15), now held at ~0 by a scale floor at the window's
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
