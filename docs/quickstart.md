# Quickstart

Two paths: a two-minute simulated incident (no API keys), then your own
agent.

## 1. The two-minute demo

The built-in simulator plays an agent whose "model version" is silently
swapped mid-history — shifting output length, refusal rate, and format
validity.

```bash
pip install dedrift
mkdir drift-demo && cd drift-demo

dedrift init                                   # creates .dedrift/
dedrift embedder pin hash                      # optional: Tier-2 semantic signatures
dedrift sim --cycles 8 --change-cycle 7        # 8 canary cycles; swap at cycle 7
dedrift baseline set cycle-0000 cycle-0001 cycle-0002
dedrift check                                  # exit code 2: DRIFT DETECTED
dedrift report --out report.md
```

The report shows what shifted in plain units, BH-adjusted p-values, and the
attribution: *"nearest config event: model fingerprint change, 0.0 h before
onset."*

## 2. Your own agent

dedrift needs exactly one function — text in, behavior out:

```python
# myagent.py
def agent_fn(input: dict) -> dict:
    response = my_agent.run(input["text"])
    return {
        "text": response.text,  # required
        "structured": response.json,  # optional: enables format/exact-match signatures
        "tokens_out": response.usage.output_tokens,  # optional
    }
```

…and a frozen [canary suite](canaries.md). Then, on a schedule (cron, CI,
anything):

```bash
dedrift canary run --suite canaries.yaml --agent myagent:agent_fn \
    --model 'anthropic/claude-sonnet-5@2026-05-01'
dedrift check     # exit 0 = OK, 2 = drift, 3 = inconclusive, 1 = operational error
```

**Cycle lifecycle.** Checks consume only *finalized* cycles, so a
partially-written cycle can never be mistaken for a complete one. `dedrift sim`
and `dedrift canary run` finalize as they go; `dedrift log` deliberately leaves
imported cycles **open** unless you pass `--finalize-cycles`, so import then
finalize explicitly:

```bash
dedrift log records.jsonl --finalize-cycles
# or, per cycle, with an exact expected count:
dedrift cycle finalize cycle-0007 --expected-records 126
```

Histories created before 0.3.1 migrate as open and must be finalized once before
they will check.

The `--model` string feeds the config fingerprint — record it accurately,
because it is what attribution correlates behavioral onsets against.

### Ready-made adapters

The repository ships two Anthropic adapters used in the
[real-world case study](case-study.md):

- **`examples/anthropic_agent.py`** — API key via the `anthropic` SDK
  (recommended: proper sampling control, low latency overhead).
- **`examples/claude_code_agent.py`** — no API key needed; drives the
  Claude Code CLI headlessly on a Claude subscription. Caveats documented
  in the file: CLI latency overhead, no sampling control, subscription
  limits.

```bash
DEDRIFT_DEMO_MODEL=claude-haiku-4-5 dedrift canary run \
    --suite examples/canaries_real.yaml \
    --agent examples.anthropic_agent:agent_fn \
    --model 'anthropic/claude-haiku-4-5'
```

## 3. Freeze a golden baseline

After you've collected a few cycles you trust:

```bash
dedrift baseline set --first 3     # or list explicit cycle IDs
```

Every check thereafter compares against **both** the rolling window (sudden
breaks) and this frozen baseline (slow boiling-frog drift). Without a golden
baseline, slow drift gets absorbed into an adaptive reference — that's the
failure mode, not a configuration choice.

## 4. Read a check like an operator

- **Alerts** passed BH adjustment at q=0.05 *and* an observed materiality gate — these are
  worth waking up for.
- **"Significant, below materiality"** — real but small; watch it.
- **Flags** (PSI, Page–Hinkley) are uncalibrated diagnostics that localize
  onsets for attribution. A stable agent shows occasional flags —
  [measured: 69.8% of stable checks](statistics.md) — so never page on flags.
- **COMPOSITION MISMATCH** means a canary's records went missing and the
  comparison was suppressed: fix collection, don't chase drift.
- **DEGRADED DATA** means too many current-cycle errors for any drift
  conclusion.

### Two inference modes

`dedrift check` controls false alarms *per check*. Because monitoring runs
forever, that rate compounds — measured 3.2% per check on stable agents,
about 23 false alerts a month at hourly checks. Two answers ship:
`detection.alert_persistence = 2` requires an alert to repeat on a fresh
cycle before it fires (wobble-induced false alerts are transient, drift
persists), and `--inference anytime` uses lifetime-oriented rate e-processes
instead (measured: 2 in 500 stable runs of 2000 cycles, Wilson upper 1.5%),
at a real cost in detection power. Per-process and per-check results are
proven; the repeated dependent battery relies on a documented causal
assumption. Read
[anytime-valid mode](anytime.md) before switching; `fixed` is the default.

```bash
dedrift check --inference anytime     # lifetime-oriented rate monitoring
```

Defaults are documented in the [configuration reference](configuration.md);
what the suite should contain is in [writing canary suites](canaries.md);
the math behind the verdicts is in [the statistics](statistics.md).
