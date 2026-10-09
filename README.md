# dedrift

**An experimental behavioral regression gate for repeatable AI-agent workflows.**

dedrift is an open-source Python package for comparing repeated runs of a
frozen canary suite across agent releases. It logs interactions, extracts
behavioral signatures, and flags statistically significant, materially large
distribution changes with correlational config-change context.

It is **not a passive production monitor, a production pager, or evidence that a
model change caused a regression**. It can miss important changes, and its
published calibration applies only to the stated synthetic scenarios and
battery size. The useful current role is a release/upgrade experiment whose
result a human reviews alongside task-specific evals.

## Why dedrift

Model updates, prompt edits, tool-schema changes, RAG index refreshes, and
provider-side updates can change agent behavior without throwing an error.
Frozen canaries hold the inputs constant so a team can ask a narrower question:
did this build behave differently from a predeclared reference?

What the current implementation actually supports:

- Every p-valued detector's false-alarm rate is *measured* by simulation tests in CI
  against stated acceptance bands — and the full pipeline's null alert rate is bounded
  over 500 seeded stable-agent runs at a stated scale
  (12 canaries × 5 repetitions): **16/500 = 3.2%, Wilson 95% upper
  5.1%**, against a CI gate banded at 6.5%. That rate is family-wise and
  grows with battery size m even under valid per-test FDR — the
  `tool_order_inversions` signature takes the pool to m ≈ 336 primaries,
  and at m ≈ 300 the same study measures 10/500 (upper 3.6%) — so the
  headline number is always published at its battery size
  (see [the statistics page](https://dedrift.ai/docs/statistics/)).
- Primary equality-test p-values pass through Benjamini–Hochberg adjustment;
  redundant tests run as corroboration outside the pool. PRDS is not proven for this
  shared battery, and the later observed-effect filter is not a practical-null FDR
  procedure. The release therefore relies on stated default-scenario simulations,
  not a universal production FDR claim.
- On the default fixed-sample path, every alert requires both statistical
  significance and a configurable observed effect-size (materiality) threshold.
  This is a prioritization rule, not a proof that each surviving alert is a
  product regression.
- LLM outputs are stochastic: canaries run N times per cycle and we compare distributions, never single outputs.
- Fixed-sample checks use dual baselines: a rolling recent window (sudden
  breaks) and a frozen golden baseline (boiling-frog drift).
- Honest about power: small N means low detection power, and the docs show you the math instead of hiding it.
- **Anytime mode** (`--inference anytime`): swaps per-check FDR for
  lifetime-oriented rate e-processes targeting P(ever falsely alerting on a
  stable agent) ≤ α, per epoch, under the stated stable-rate Bernoulli
  model. Version 0.5.1 uses the frozen Clopper–Pearson construction
  (`anytime.rate_model = "frozen_cp"`) as the only supported rate model. It
  splits `α′ = α - gamma_total` across a fixed process pool and makes
  crossings persistent, so Ville's inequality plus a Bonferroni union bound
  handles cross-process dependence without e-BH. The former `twosample`
  prior-predictive ratio, dependent pooled products, repeated e-BH decision,
  and their published null/power headlines are withdrawn. The corrected mode
  is conservative and can have very low power at canary scale; no replacement
  power table is claimed in 0.5.1. It is opt-in and golden-baseline only; read
  the [assumptions and limitations](https://dedrift.ai/docs/anytime/) before
  using it in a release decision. Its configured tilts aim the predictable bets and affect power; they
  are not an observed-effect materiality gate, so the fixed path's practical-
  significance claim does not transfer to anytime alerts.

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
frozen-reference e-processes and persistent Ville–Bonferroni crossings
targeting lifetime rather than per-check control, with per-epoch semantics,
persisted exactly-once process state, and explicit coverage status. Rate
channel only so far.

This is pre-alpha research software, not a production alerting system. Checks currently scan
the full finalized history, making each check O(history) and long runs
potentially O(cycles²) in cumulative compute. Check rows use the finalized
revision rather than duplicating full record-ID manifests, but bounded indexed
windows/checkpoints, retention/redaction, privacy controls, and deployment-scale
benchmarks remain pre-1.0 blockers. There is no supported pager integration,
service-level objective, or evidence that the detector stays useful across
frequent releases. Use one `Store` instance per thread, back up the entire
`.dedrift` directory, and validate both false-alert rate and detection power on
your own replay set before treating a result as a release signal.

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
monitor_latency = false                   # host jitter is not scorer behavior
```

Published false-alarm rates are measured on the **default** battery; your own channels
enlarge m and bring their own correlation structure. See
[configuration](https://dedrift.ai/docs/configuration/).

### Typed evaluators, including Jev

A typed evaluator can turn qualities such as policy compliance or task success
into numeric probabilities, which dedrift can ingest through `custom_scalars`.
That is a potentially useful composition: the evaluator supplies a semantic
measurement and dedrift looks for changes in its distribution over releases.

There is currently **no built-in or validated [Jev](https://docs.typesafe.ai/models)
integration**. Jev/RLCD does
not replace baselines, repeated measurements, or longitudinal change detection,
and [aggregate calibration](https://docs.typesafe.ai/confidence) is not a
per-decision guarantee. If you prototype
this path, pin an exact evaluator model version (not a moving alias), record it
in the agent config fingerprint, keep the raw decision/probability, and measure
calibration and order/prompt sensitivity on your own labeled data before adding
the channel to a gate.

## Install

```bash
pip install dedrift              # core: zero ML dependencies
pip install "dedrift[embeddings]"  # + semantic signatures (sentence-transformers)
```

`rubric_id` is currently preserved as provenance only. No LLM judge is
executed or advertised by this release.

For development: `pip install -e ".[dev]"`.

## Quickstart — a controlled regression demonstration

No API keys needed: `dedrift sim` ships a seeded synthetic agent whose "model
version" is swapped mid-history, shifting output length, refusal rate, and
format validity. This scenario is deliberately large and engineered to be
detectable; it is a pipeline smoke test, not evidence that arbitrary real-world
regressions will be caught.

```bash
pip install dedrift
mkdir drift-demo && cd drift-demo

dedrift init                                   # create the project
dedrift embedder pin hash                      # optional: enable Tier-2 semantic signatures
dedrift sim --cycles 8 --change-cycle 7        # 8 canary cycles; model swap at cycle 7
dedrift baseline set --first 3   # predeclare by position; never cherry-pick outcomes
dedrift check                                  # this seeded demo is expected to exit 2
dedrift report --out report.md                 # also exits 2 when its check detects drift
```

The report shows measured shifts in the channel's units, BH-adjusted p-values,
and the nearest preceding config event as correlational context. With your own agent,
replace `sim` with `dedrift canary run --suite canaries.yaml --agent
yourmodule:agent_fn --model 'provider/model@version'` in a test environment.

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

## Commercial status

There is no hosted service or paid production tier today. A commercial product
should not be inferred from this repository or roadmap language. The immediate
goal is to learn whether teams repeatedly use the narrower release-gate workflow
and whether it saves them from real regressions.

## Contact

Questions, bug reports, or interest in being a design partner:
[open an issue](https://github.com/dedrift/dedrift/issues) or email
[support@dedrift.ai](mailto:support@dedrift.ai).
Report suspected vulnerabilities privately as described in
[`SECURITY.md`](SECURITY.md), not in a public issue.

## License

AGPL-3.0-only. See `LICENSE`.
