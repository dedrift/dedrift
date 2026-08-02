# The statistics

This page is the reason dedrift exists. Every claim below is enforced by a
test in CI; where a guarantee is approximate or a procedure is heuristic, the
docs say so plainly.

## The detector battery

| Signature type | Test | Notes |
|---|---|---|
| Scalars, per family | Two-sample Kolmogorov–Smirnov and Anderson–Darling | Effect: Cohen's d + raw shift in original units |
| Scalar location | Welch's t | Secondary — only ever alongside KS, never alone |
| Scalar dispersion | Levene (median-centered), bootstrap P95 shift | Agents often go erratic before they go wrong on average |
| Rates (refusal, validity, exact-match) | Two-proportion z with continuity correction | Effect: percentage-point shift |
| Embeddings, per family | MMD² (RBF), seeded permutation null ≥ 500 perms | Bandwidth: median heuristic on the **reference window only** |
| Sequential per-cycle means | Page–Hinkley | A flag and onset-localizer, not a p-value |
| Industry heuristic | PSI, 10 bins frozen from golden | Labeled heuristic (0.1 / 0.25); **never** presented as a test |

## The gating pipeline — order matters

1. Run all tests; collect p-values. PSI and Page–Hinkley produce flags, not
   p-values, and never enter step 2.
2. **Benjamini–Hochberg FDR at q = 0.05** across *all* p-valued tests in the
   check — one multiplicity family spanning both baselines. BH is valid under
   independence and positive regression dependence (PRDS), which covers the
   positively correlated tests run on the same data.
3. Survivors pass a **materiality gate**: per-channel effect-size thresholds
   (e.g. refusal shift ≥ 2 pp, |Cohen's d| ≥ 0.5, variance ratio ≥ 1.5×,
   relative P95 shift ≥ 10%) — all configurable.
4. Only tests passing **both** gates become alerts. Everything else appears
   in the report as "significant, below materiality" or "not significant".

Alert fatigue kills monitoring tools; the double gate is deliberate.

## Calibration is enforced, not asserted

The simulation harness generates agent logs with known ground truth, and CI
runs (on every commit and inside the release pipeline):

- **Null calibration** — 20 seeded no-change runs through the full pipeline
  must produce (within documented tolerance) zero alerts; per-test
  false-alarm rates for KS, the two-proportion z, and MMD are measured
  against their nominal levels with stated acceptance bands.
- **Power checks** — injected shifts (mean, variance, distribution swap,
  scripted model swap) must be detected at documented rates, with correct
  config attribution.

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
- **Anderson–Darling p-values are capped** by SciPy's asymptotic
  approximation to [0.001, 0.25]. Consequence: AD can corroborate KS but can
  never be the sole FDR survivor in a large battery.
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
