# dedrift

**Agents don't throw errors when they degrade. They keep confidently producing
worse outputs. dedrift catches it.**

dedrift is an open-source Python package that detects silent behavioral drift
in AI agents. It logs agent interactions, runs a frozen canary suite
repeatedly, extracts behavioral signatures, and applies statistically rigorous
drift detection with config-change attribution — *"behavior shifted within 6
hours of the model version change."*

```bash
pip install dedrift
```

## Why another monitoring tool?

Model updates, prompt edits, tool-schema changes, RAG index refreshes, and
provider-side silent updates all shift agent behavior without a single error.
Existing observability catches crashes, not character changes — and most
drift tooling raises alerts from raw per-test p-values, which guarantees
alert fatigue.

dedrift's entire differentiation is statistical correctness:

- **Calibrated false-alarm rates.** Every detector's false-alarm rate is
  validated by simulation tests that run in CI on every commit — and in the
  release pipeline, so a release cannot ship if the statistics fail their own
  audit.
- **Multiplicity is always handled.** All alerting passes through
  Benjamini–Hochberg FDR control. Raw per-test p-values are never surfaced as
  alerts.
- **Significance AND materiality.** Every alert requires both statistical
  significance and a configurable effect-size threshold. Fewer, higher-confidence
  alerts.
- **Stochasticity ≠ drift.** LLM outputs are random. Canaries run N times per
  cycle and dedrift compares distributions, never single outputs.
- **Dual baselines.** Every check runs against a rolling recent window (sudden
  breaks) and a frozen golden baseline (slow boiling-frog drift).
- **Honest about power.** Small canary suites have limited detection power.
  [The docs show you the math](statistics.md#detection-power-the-honest-table)
  instead of hiding it.

## Try it in two minutes — a simulated drift incident

No API keys needed: the built-in simulator plays an agent whose "model
version" is swapped mid-history.

```bash
mkdir drift-demo && cd drift-demo
dedrift init
dedrift embedder pin hash                    # optional: Tier-2 semantic signatures
dedrift sim --cycles 8 --change-cycle 7      # model swap at cycle 7
dedrift baseline set cycle-0000 cycle-0001 cycle-0002
dedrift check                                # exit 2: DRIFT DETECTED
dedrift report --out report.md
```

The report shows what shifted in plain units (refusal +21 pp, output variance
ratio ~9x), BH-adjusted p-values, and attribution: *nearest config event —
model fingerprint change, 0.0 h before onset.*

## With your own agent

Write one function and a YAML file of frozen canary inputs:

```python
# myagent.py
def agent_fn(input: dict) -> dict:
    response = my_agent.run(input["text"])
    return {"text": response.text, "structured": response.json, ...}
```

```bash
dedrift canary run --suite canaries.yaml --agent myagent:agent_fn \
    --model 'anthropic/claude-sonnet-5@2026-05-01'
dedrift check && dedrift report
```

Run it on a schedule; dedrift fingerprints your agent config on every record,
so when behavior shifts, the report correlates the onset with the nearest
config change.

## dedrift Pro

Anytime-valid sequential inference (e-processes), conditional
production-traffic drift, and importance weighting are part of a separate
commercial tier — contact [support@dedrift.ai](mailto:support@dedrift.ai).
