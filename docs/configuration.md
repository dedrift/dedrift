# Configuration reference

Everything lives in `.dedrift/config.toml`, created by `dedrift init`.
Every key below is optional; omitted keys use the documented default. Configuration is
fail-closed: unknown sections, misspelled keys, wrong primitive types, non-finite numbers, and
values outside the domains below stop the command with exit code 1 instead of being ignored.

```toml
[project]
name = "my-agent"
canary_repetitions = 7        # N runs per canary per cycle
rolling_window_cycles = 5     # K cycles in the rolling reference

[detection]
fdr_q = 0.05                  # Benjamini–Hochberg level over primary tests
permutations = 500            # resamples for permutation tests (seeded)
seed = 1729                   # global seed, recorded in every report
ph_lambda = 12.0              # Page–Hinkley alarm threshold (robust-scale units)
ph_delta = 0.3                # Page–Hinkley dead-zone
inference = "fixed"           # "fixed" | "anytime" (see docs/anytime.md)
cycle_effect = "off"          # "off" (default) | "auto" — the cluster-aware battery
cycle_effect_icc = 0.02       # ICC threshold for engaging the correction
alert_persistence = 1         # checks a channel must alert in a row to fire

[materiality]
refusal_rate_pp = 2.0         # min refusal-rate shift, percentage points
format_validity_pp = 1.0      # min format-validity shift, pp
rate_default_pp = 2.0         # other rate signatures, pp
ks_distance = 0.15            # min KS statistic D for KS alerts (see note)
scalar_cohen_d = 0.5          # reserved compatibility key; not an alert gate
dispersion_ratio = 1.5        # dispersion gate on the ROBUST scale the test
                              # uses (mean abs deviation from the median),
                              # not the sample variance
p95_relative = 0.10           # tail gate: min relative P95 shift
embedding_mmd2_floor = -1.0   # -1 = auto-calibrate; 0 = off; >0 = explicit

[embeddings]
model = ""                    # set via `dedrift embedder pin`, not by hand
```

## Notes that matter

Counts are validated before use: `canary_repetitions >= 2`,
`rolling_window_cycles >= 1`, `permutations >= 100`, and `seed >= 0`.
Probability budgets are strict interior probabilities (`0 < fdr_q < 1`,
`0 < gamma_total < alpha < 1`). Percentage-point materiality gates are in
`[0, 100]`, `ks_distance` is in `[0, 1]`, dispersion ratios are at least 1,
and other scalar/tail gates are non-negative. The legacy `variance_ratio`
name is accepted by itself, but setting it together with `dispersion_ratio`
is an error.

### `canary_repetitions` (N)

The single biggest power lever. The
[power table](statistics.md#detection-power-the-honest-table) shows what
each N buys; cost grows linearly.

This project value is authoritative for both `dedrift canary run` and
`dedrift sim`. A legacy `--repetitions N` argument is accepted only when it
matches the project value; edit `config.toml` to change the experiment design.

### `fdr_q`

Applied **once per check** across all primary tests (both baselines
together). Corroboration tests (Anderson–Darling, Welch) never enter the
pool and never alert.

### `permutations`

This is a requested minimum, not always the effective count. Add-one
permutation p-values cannot be smaller than `1/(B+1)`, so the checker raises
`B` to at least `ceil(m_upper/fdr_q)-1`, where `m_upper` conservatively bounds
the full primary BH family. Both counts are persisted and printed. P95
permutations run in bounded-memory chunks; semantic MMD still has quadratic
kernel memory in the number of records and should be load-tested at your
intended scale.

### `cycle_effect`, `cycle_effect_icc`, `alert_persistence`

`cycle_effect = "off"` (the default) is the exact record-level battery:
every channel compares records directly against the reference windows.
`"auto"` engages the cluster-aware correction on
channels whose golden-window/history cycle means show a shared per-cycle
latent offset (hosted-model wobble): within-cycle-standardized KS for
shape, design-effect Welch for location, design-effect rate z, and
Student-t cycle-level summaries for dispersion/P95.
`cycle_effect_icc` is the engagement
threshold on the estimated ICC. `alert_persistence = 2` holds a first-time
alert until the same channel alerts at the next check — wobble alerts are
transient, drift persists; measured false-alert rates by mode are in
docs/statistics.md#cycle-effects, including the power cost (one cycle of
delay).

### `ks_distance`

The KS channel gates — and reports — the KS statistic D (sup-norm CDF
distance), because a shape change with equal means has Cohen's d ≈ 0 and is
still real drift. Binding scale, stated plainly: the raw-α=0.05 critical
value is ≈ 1.36·√((n+m)/nm), which for equal arms stays above 0.15 until
n ≳ 165 per arm — so at typical per-family scale, significance is the
stricter filter and this gate exists to stop trivially significant tiny D
from alerting at large n.

### `embedding_mmd2_floor`

Left at `-1`, the MMD² materiality floor is auto-calibrated per
(baseline, family) as the 95th percentile of MMD² between pairs of the
baseline's own cycles — an empirical null from known-same-distribution
data, computed under the same kernel bandwidth as the observed statistic.
Needs ≥ 5 reference cycles. Below that the report says `UNCALIBRATED` and
the MMD result cannot alert unless you configured an explicit floor.

### `ph_lambda`, `ph_delta`

Page–Hinkley is a **diagnostic**, not an alert: it localizes drift onsets
for attribution. The measured per-stream false-flag rate at the defaults is
**8.5%** on 30-cycle histories and 11.3% on 60-cycle ones (8000 draws); CI
asserts a Wilson interval inside [0.06, 0.12]. Centre and scale are
estimated causally, from data strictly before each step — a sequential
detector that standardised on cycles *after* the alarm would read its own
future and understate the rate roughly sixfold. Flags compound
across streams — a stable agent shows ≥1 flag on
[68.6% of checks](statistics.md) — which is why flags never page anyone.

### Embeddings

```bash
dedrift embedder pin hash                    # zero-dependency, always available
dedrift embedder pin st:all-MiniLM-L6-v2    # requires dedrift[embeddings]
```

Pinning is **forever per project**: changing the embedder invalidates all
semantic history, so dedrift refuses to compare across embedder versions
rather than produce quietly meaningless numbers. Pick deliberately, then
leave it.

### `[anytime]` — anytime-valid mode

```toml
[detection]
inference = "fixed"        # or "anytime"

[anytime]
alpha = 0.05               # lifetime, battery-wide false-alert budget
rate_model = "twosample"   # or "frozen_cp" (the Clopper–Pearson interval construction)
gamma_total = 0.02         # frozen_cp only: coverage budget
tilts = [1.5, 2.0, 3.0]    # frozen_cp only: symmetrised to {psi, 1/psi}
epoch_allocation = "per_epoch"   # or "geometric"
```

With `rate_model = "twosample"` (default) each rate process is a two-sample
SAFE e-value against the frozen reference counts; there is no nuisance
interval, so no coverage budget is spent and the e-BH level is the full
`alpha`. `gamma_total` and `tilts` apply only to `"frozen_cp"`.

Switches the inference layer from per-check FDR to lifetime-oriented rate
e-processes. Per-process optional-stopping and per-check e-BH control are
proven; repeated dependent e-BH has the causal assumption documented on the
linked page. The power trade is quantified in
[anytime-valid mode](anytime.md). Default is `fixed`.

## Exit codes

| Command | Exit | Meaning |
|---|---|---|
| `dedrift check` | 0 | OK with valid evidence coverage |
| `dedrift check` | 2 | DRIFT DETECTED — wire this to alerting |
| `dedrift check` | 3 | Inconclusive: degraded, missing/invalid reference, or partial coverage |
