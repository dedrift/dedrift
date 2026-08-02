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
| Location / shape | Two-sample Kolmogorov–Smirnov | KS statistic D ≥ 0.15 (sup-norm CDF distance — catches shape changes with equal means, where Cohen's d ≈ 0) | Anderson–Darling, Welch's t (raw p shown, never alert) |
| Dispersion | Levene (median-centered) | Variance ratio ≥ 1.5× either way | — |
| Tail | P95 permutation test (pooled labels, seeded, add-one p) | Relative P95 shift ≥ 10% | — |
| Rates | Two-proportion z, continuity-corrected | Percentage-point thresholds (refusal ≥ 2 pp, …) | — |
| Semantic | MMD² (RBF), seeded permutation null ≥ 500 perms, **pooled** median-heuristic bandwidth (permutation-invariant, so the p-value is exact-level) | Auto-calibrated MMD² floor, same bandwidth as the observation | — |
| Sequential per-cycle means | Page–Hinkley | Flag + onset localizer, not a p-value | — |
| Industry heuristic | PSI, 10 bins frozen from golden | Labeled heuristic (0.1 / 0.25); **never** a test | — |

## The gating pipeline — order matters

1. Run all tests; collect p-values. PSI and Page–Hinkley produce flags, not
   p-values, and never enter step 2. Corroboration tests never enter it
   either.
2. **Benjamini–Hochberg FDR at q = 0.05** across the primary tests in the
   check — one multiplicity family spanning both baselines.
3. Survivors pass a **materiality gate** (per-channel thresholds in the
   table above) — all configurable.
4. Only primary tests passing **both** gates become alerts. Everything else
   appears in the report as "significant, below materiality", "not
   significant", or "corroboration".

Alert fatigue kills monitoring tools; the double gate is deliberate.

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
  Wilson 95% upper bound below 0.05. (A pass/fail over 20 runs — the
  original spec — would pass 39% of the time even with a true 10% alert
  rate; 500 runs with a reported bound is the honest version of the claim.)
- **Power checks** — injected shifts (mean, variance, distribution swap,
  equal-mean shape change, scripted model swap) must be detected at
  documented rates, with correct config attribution.

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

## Fine print, stated plainly

- **Balanced-design exchangeability.** Both comparison windows contain the
  same canaries at the same repetition count, so under the strong null ("no
  change anywhere in the stack") pooled per-family samples are exchangeable
  across windows and the two-sample tests apply. Power against shifts
  confined to a few canaries is correspondingly lower than for family-wide
  shifts.
- **Anderson–Darling and Welch are corroboration, not evidence.** They test
  the same location/shape hypothesis as KS on the same data; admitting them
  to the FDR pool would roughly double m — halving every BH threshold — for
  no informational gain. Their raw p-values are printed for context and can
  never alert. (AD's p is additionally capped by SciPy's asymptotics to
  [0.001, 0.25].)
- **BH validity for this battery is asserted under PRDS, not proven.** The
  primary tests are positively correlated on the same data, which PRDS is
  believed to cover, but Levene-vs-KS dependence under non-normality is not
  a theorem we can cite. The pipeline-level calibration measures the
  realized alert rate directly, which is the operative guarantee; a
  Benjamini–Yekutieli option is on the roadmap for the cautious.
- **Page–Hinkley** (λ = 12, δ = 0.3 in reference-SD units, scale from the
  median absolute successive difference): the idealized null crossing bound
  is ~0.15% per stream, but because centering and scale are estimated, the
  **measured** rate is ~1.5% per stream over 30-cycle horizons — the
  calibration test enforces < 3%. PH alarms localize onsets for attribution;
  they only alert through the same materiality gating as batch tests.
- **MMD² materiality floor** is auto-calibrated per (baseline, family) as the
  95th percentile of MMD² between pairs of the baseline's own cycles — an
  empirical null from known-same-distribution data. With fewer than three
  reference cycles the floor is uncalibratable (0), and the config accepts an
  explicit override.
- **Reproducibility.** All randomness is seeded; permutation and bootstrap
  seeds are recorded in the report. Same logs + same config ⇒ identical
  report, byte for byte.
