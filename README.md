# dedrift

**Agents don't throw errors when they degrade. They keep confidently producing worse outputs. dedrift catches it.**

dedrift is an open-source Python package that detects silent behavioral drift in AI agents. It logs agent interactions, runs a frozen canary suite repeatedly, extracts behavioral signatures, and applies statistically rigorous drift detection with config-change attribution ("behavior shifted within 6h of the model version change").

## Why dedrift

Model updates, prompt edits, tool-schema changes, RAG index refreshes, and provider-side silent updates all shift agent behavior without throwing a single error. Existing monitoring catches crashes, not character changes.

dedrift's differentiation is statistical correctness:

- Every p-valued detector's false-alarm rate is *measured* by simulation tests in CI
  against stated acceptance bands — and the full pipeline's null alert rate is bounded
  (Wilson 95% upper bound < 5%) over 500 seeded stable-agent runs at a stated scale
  (12 canaries × 5 repetitions; see [the statistics page](https://dedrift.ai/statistics/)).
- All alerting passes through FDR control (Benjamini–Hochberg) over one primary test per
  channel; redundant tests run as corroboration outside the pool. No raw per-test
  p-values dressed up as alerts.
- Every alert requires both statistical significance and a configurable effect-size (materiality) threshold. Fewer, higher-confidence alerts.
- LLM outputs are stochastic: canaries run N times per cycle and we compare distributions, never single outputs.
- Dual baselines: every check runs against a rolling recent window (sudden breaks) and a frozen golden baseline (boiling-frog drift).
- Honest about power: small N means low detection power, and the docs show you the math instead of hiding it.
- **Anytime-valid mode** (`--inference anytime`): swaps per-check FDR for a
  *lifetime* guarantee — over an unbounded horizon, P(ever falsely alerting on
  a stable agent) ≤ α, per epoch. Measured 0 false alerts across 500
  stable-agent runs of 2000 cycles each with dependent streams (no false alert at any
  measured horizon), against 100% for the per-check path on identical histories. It
  costs detection power, and the cost is inconsistency rather than delay: a
  +10 pp shift on one channel is caught in only 23% of runs. Opt-in, golden
  baseline only, and both numbers are published.

## Status

Pre-alpha, under active development. Working today: logging schema + store, canary runner
(N repetitions per cycle), Tier-1 structural signatures, Tier-2 semantic signatures
(pinned embedder, semantic displacement, MMD-RBF with a seeded permutation null and an
auto-calibrated materiality floor), the full detector battery
(KS/Levene/permutation-P95/two-proportion z/MMD as primaries; AD and Welch as
corroboration; PSI and Page–Hinkley as labeled diagnostics) with BH-FDR over primaries
and materiality gating, dual baselines, config-change attribution, and deterministic
markdown reports — all with calibration and power tests enforced in CI. Plus an
opt-in **anytime-valid** inference path (`--inference anytime`): e-values,
e-processes and e-BH giving a lifetime rather than per-check guarantee, with
per-epoch semantics and persisted process state. Rate channel only so far —
run both modes.

## Install

```bash
pip install dedrift              # core: zero ML dependencies
pip install "dedrift[embeddings]"  # + semantic signatures (sentence-transformers)
pip install "dedrift[judge]"       # + LLM-judge tier
```

For development: `pip install -e ".[dev]"`.

## Quickstart — a full simulated drift incident in five commands

No API keys needed: `dedrift sim` ships a seeded synthetic agent whose "model
version" is swapped mid-history, shifting output length, refusal rate, and
format validity — the classic silent degradation.

```bash
pip install dedrift
mkdir drift-demo && cd drift-demo

dedrift init                                   # create the project
dedrift embedder pin hash                      # optional: enable Tier-2 semantic signatures
dedrift sim --cycles 8 --change-cycle 7        # 8 canary cycles; model swap at cycle 7
dedrift baseline set cycle-0000 cycle-0001 cycle-0002   # freeze known-good cycles
dedrift check                                  # exits 2: DRIFT DETECTED (both baselines)
dedrift report --out report.md                 # deterministic markdown report
```

The report shows what shifted in plain units (e.g. refusal +21 pp, output
variance ratio ~9x), BH-adjusted p-values, and attribution: "nearest config
event: model fingerprint change, 0.0 h before onset." With your own agent,
replace `sim` with `dedrift canary run --suite canaries.yaml --agent
yourmodule:agent_fn --model 'provider/model@version'` on a schedule.

## Detection power: the honest table

Statistical power depends on sample size, and canary suites are small. For a
rate signature (e.g. refusal) at a 5% baseline in a family of 30 canaries,
two-sided α=0.05 per test (before FDR, which reduces power further), simulated
power to detect a shift of the given size:

| Repetitions N | n per window | +2 pp | +5 pp | +10 pp | +15 pp |
|---|---|---|---|---|---|
| 5  | 150 | 0.07 | 0.28 | 0.78 | 0.97 |
| 7 (default) | 210 | 0.11 | 0.42 | 0.91 | 1.00 |
| 10 | 300 | 0.13 | 0.60 | 0.99 | 1.00 |

Read the first column honestly: **a 2 pp refusal shift is essentially
undetectable at this scale.** dedrift's default materiality gate (2 pp) is a
floor on what may alert, not a promise of what will be detected. If small rate
shifts matter to you, grow the refusal-boundary family or raise N — power
comes from samples, not from wishful thresholds.

## dedrift Pro

A commercial tier with advanced inference is in development and lives outside
this repository. Email [support@dedrift.ai](mailto:support@dedrift.ai) to hear
when it ships.

## Contact

Questions, bug reports, or interest in being a design partner:
[open an issue](https://github.com/dedrift/dedrift/issues) or email
[support@dedrift.ai](mailto:support@dedrift.ai).

## License

AGPL-3.0-only. See `LICENSE`.
