# Configuration reference

Everything lives in `.dedrift/config.toml`, created by `dedrift init`.
Every key below is optional; omitted keys use the documented default.

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

[materiality]
refusal_rate_pp = 2.0         # min refusal-rate shift, percentage points
format_validity_pp = 1.0      # min format-validity shift, pp
rate_default_pp = 2.0         # other rate signatures, pp
ks_distance = 0.15            # min KS statistic D for KS alerts (see note)
scalar_cohen_d = 0.5          # location gate for non-KS scalar channels
dispersion_ratio = 1.5        # dispersion gate on the ROBUST scale the test
                              # uses (mean abs deviation from the median),
                              # not the sample variance
p95_relative = 0.10           # tail gate: min relative P95 shift
embedding_mmd2_floor = -1.0   # -1 = auto-calibrate; 0 = off; >0 = explicit

[embeddings]
model = ""                    # set via `dedrift embedder pin`, not by hand
```

## Notes that matter

### `canary_repetitions` (N)

The single biggest power lever. The
[power table](statistics.md#detection-power-the-honest-table) shows what
each N buys; cost grows linearly.

### `fdr_q`

Applied **once per check** across all primary tests (both baselines
together). Corroboration tests (Anderson–Darling, Welch) never enter the
pool and never alert.

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
Needs ≥ 3 reference cycles; below that the floor is 0 and the report says
"uncalibratable".

### `ph_lambda`, `ph_delta`

Page–Hinkley is a **diagnostic**, not an alert: it localizes drift onsets
for attribution. Defaults are calibrated to ≈1.5% per-stream false-flag
rate over 30-cycle horizons (measured, enforced < 3% in CI). Flags compound
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
gamma_total = 0.02         # coverage budget; alpha_prime = alpha - gamma_total
tilts = [1.5, 2.0, 3.0]    # symmetrised to {psi, 1/psi}
epoch_allocation = "per_epoch"   # or "geometric"
```

Switches the inference layer from per-check FDR to e-processes with a
lifetime guarantee. The trade — validity over an unbounded horizon, paid for
in detection power — is quantified in
[anytime-valid mode](anytime.md). Default stays `fixed`.

## Exit codes

| Command | Exit | Meaning |
|---|---|---|
| `dedrift check` | 0 | OK (or NO REFERENCE) |
| `dedrift check` | 2 | DRIFT DETECTED — wire this to alerting |
