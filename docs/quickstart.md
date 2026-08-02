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
dedrift check     # exit 0 = OK, exit 2 = drift: wire it to your alerting
```

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

- **Alerts** passed BH-FDR at q=0.05 *and* a materiality gate — these are
  worth waking up for.
- **"Significant, below materiality"** — real but small; watch it.
- **Flags** (PSI, Page–Hinkley) are uncalibrated diagnostics that localize
  onsets for attribution. A stable agent shows occasional flags —
  [measured: 56% of stable checks](statistics.md) — so never page on flags.
- **COMPOSITION MISMATCH** means a canary's records went missing and the
  comparison was suppressed: fix collection, don't chase drift.
- **DEGRADED DATA** means too many current-cycle errors for any drift
  conclusion.

Defaults are documented in the [configuration reference](configuration.md);
what the suite should contain is in [writing canary suites](canaries.md);
the math behind the verdicts is in [the statistics](statistics.md).
