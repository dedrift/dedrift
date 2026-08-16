# dedrift

**Agents don't throw errors when they degrade. They keep confidently producing worse outputs. dedrift catches it.**

dedrift is an open-source Python package that detects silent behavioral drift in AI agents. It logs agent interactions, runs a frozen canary suite repeatedly, extracts behavioral signatures, and applies statistically rigorous drift detection with config-change attribution ("behavior shifted within 6h of the model version change").

## Why dedrift

Model updates, prompt edits, tool-schema changes, RAG index refreshes, and provider-side silent updates all shift agent behavior without throwing a single error. Existing monitoring catches crashes, not character changes.

dedrift's differentiation is statistical correctness:

- Every p-valued detector's false-alarm rate is *measured* by simulation tests in CI
  against stated acceptance bands — and the full pipeline's null alert rate is bounded
  over 500 seeded stable-agent runs at a stated scale
  (12 canaries × 5 repetitions): **16/500 = 3.2%, Wilson 95% upper
  5.1%**, against a CI gate banded at 6.5%. That rate is family-wise and
  grows with battery size m even under valid per-test FDR — the
  `tool_order_inversions` signature takes the pool to m ≈ 336 primaries,
  and at m ≈ 300 the same study measures 10/500 (upper 3.6%) — so the
  headline number is always published at its battery size
  (see [the statistics page](https://dedrift.ai/statistics/)).
- Primary equality-test p-values pass through Benjamini–Hochberg adjustment;
  redundant tests run as corroboration outside the pool. PRDS is not proven for this
  shared battery, and the later observed-effect filter is not a practical-null FDR
  procedure. The release therefore relies on stated default-scenario simulations,
  not a universal production FDR claim.
- Every alert requires both statistical significance and a configurable effect-size (materiality) threshold. Fewer, higher-confidence alerts.
- LLM outputs are stochastic: canaries run N times per cycle and we compare distributions, never single outputs.
- Dual baselines: every check runs against a rolling recent window (sudden breaks) and a frozen golden baseline (boiling-frog drift).
- Honest about power: small N means low detection power, and the docs show you the math instead of hiding it.
- **Anytime mode** (`--inference anytime`): swaps per-check FDR for
  lifetime-oriented rate e-processes targeting P(ever falsely alerting on a
  stable agent) ≤ α, per epoch. Per-process optional-stopping control and
  per-check e-BH are proven; the repeated dependent battery has a documented
  causal assumption and is measured rather than presented as an unconditional
  theorem. The default rate model is the two-sample SAFE e-value: measured 2
  false alerts across 500 stable-agent runs of 2000 cycles each with
  dependent streams (0.4%, Wilson upper 1.5%), against 100% for the
  per-check path on identical histories. Power on refusal shifts: +20 pp
  detected in 100/100 runs (median 17 cycles), +10 pp in 89/100 (median 50
  cycles, 400-cycle horizon). The measured validity boundary: persistent
  AR(1) cycle offsets (σ = 0.25, φ = 0.9) push the ever-alert rate to 7.2%,
  above the 5% budget — published as the boundary of the guarantee. An
  alternative Clopper–Pearson construction (`anytime.rate_model =
  "frozen_cp"`) ships as a documented ablation; its coverage interval traps
  the alternative at canary scale (audit: 0/30 detections at +5/+10/+20 pp
  over 60 cycles). Opt-in,
  golden baseline only, and both numbers are published.

## Status

Pre-alpha, under active development. Working today: logging schema + store, canary runner
(N repetitions per cycle), Tier-1 structural signatures, Tier-2 semantic signatures
(pinned embedder, semantic displacement, MMD-RBF with a seeded permutation null and an
auto-calibrated materiality floor), the full detector battery
(KS/Levene/permutation-P95/two-proportion z/MMD as primaries; AD and Welch as
corroboration; PSI and Page–Hinkley as labeled diagnostics) with BH adjustment over primaries
and materiality gating, dual baselines, config-change attribution, and deterministic
markdown reports — all with calibration and power tests enforced in CI. Plus an
opt-in **anytime** inference path (`--inference anytime`): e-values,
e-processes and e-BH targeting lifetime rather than per-check control, with
per-epoch semantics, persisted exactly-once process state, and explicit
coverage status. Rate channel only so far — run both modes.

### Agents that don't emit text

The eight built-in scalar signatures measure properties of generated text. If your agent
returns a number — a scorer, a ranker, a classifier emitting a calibrated probability —
seven of them are constant by construction and the eighth is latency. Declare your own
numeric channels and they enter the same battery as a built-in signature, with the same
gates and the same BH adjudication:

```toml
[project]
custom_scalars = ["score", "top_prob"]   # read from output.structured; max 16
deterministic = true                      # exactly reproducible ⇒ N = 1 is honest
```

Published false-alarm rates are measured on the **default** battery; your own channels
enlarge m and bring their own correlation structure. See
[configuration](https://dedrift.ai/configuration/).

## Install

```bash
pip install dedrift              # core: zero ML dependencies
pip install "dedrift[embeddings]"  # + semantic signatures (sentence-transformers)
```

`rubric_id` is currently preserved as provenance only. No LLM judge is
executed or advertised by this release.

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

The report shows what shifted in plain units (e.g. refusal +21 pp, latency
dispersion ratio 2.4x), BH-adjusted p-values, and attribution: "nearest config
event: model fingerprint change, 0.0 h before onset." With your own agent,
replace `sim` with `dedrift canary run --suite canaries.yaml --agent
yourmodule:agent_fn --model 'provider/model@version'` on a schedule.

`project.canary_repetitions` in `.dedrift/config.toml` is the authoritative
sample design for both commands. `dedrift check` exits 0 only for a fully
supported `OK`, 2 for detected drift, and 3 when evidence is degraded,
missing, or only partially comparable.

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
