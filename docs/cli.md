# CLI reference

Every command takes `--project PATH` (default: current directory). The
library API in `dedrift.*` mirrors the CLI one-to-one.

## `dedrift init`

Create a project: `.dedrift/` with `config.toml` (thresholds, FDR level,
seeds, materiality gates — all documented inline), an append-only JSONL log,
and a SQLite index. Everything runs on a laptop; there are no servers.

## `dedrift log FILE`

Ingest `InteractionRecord`s from a JSONL file — the generic adapter for any
agent framework. Config fingerprints are computed per record and fingerprint
changes become config events for attribution.

Imported canary cycles stay **open** by default so streaming a partial file
can never become inference-ready accidentally. Use `--finalize-cycles` (and
prefer `--expected-records N`) when the file contains complete cycles, or run
`dedrift cycle finalize CYCLE_ID --expected-records N` after the final chunk.
Finalized cycles are immutable; checks, reports, signatures, and baselines
ignore open cycles. `dedrift canary run` and `dedrift sim` finalize their own
complete batches atomically.

## `dedrift canary run`

```bash
dedrift canary run --suite canaries.yaml --agent mymodule:agent_fn \
    --model 'provider/model@version' [--cycles 1]
```

Runs every canary in the suite N times against your callable
(`def agent_fn(input: dict) -> dict`). Repetitions below 2 are rejected:
single runs cannot support distributional comparison. N comes from the
project's `canary_repetitions`; a supplied `--repetitions` must match that
value so CLI flags cannot silently change the monitoring design.

## `dedrift baseline set`

```bash
dedrift baseline set cycle-0000 cycle-0001 cycle-0002   # explicit
dedrift baseline set --last 3                            # last N cycles
```

Freezes finalized cycles as the golden baseline. It is never auto-updated; drifting baselines are
how boiling frogs get cooked.

## `dedrift embedder pin NAME` / `dedrift embedder show`

Pin the project's embedding model **forever** (`hash` or `st:<model>`).
Re-pinning to a different embedder is refused — it would invalidate all
history.

## `dedrift sim`

```bash
dedrift sim --cycles 8 --change-cycle 7 [--canaries 30] [--seed 1729]
```

Seeded synthetic agent with a scriptable mid-history config change. Powers
the demo and the CI calibration/power suites.

## `dedrift check`

Runs the full gated pipeline for the latest cycle against both baselines.
Prints both verdicts and the alert list; **exits 2 when drift is detected**,
so it drops straight into cron/CI. Exit 0 is reserved for an overall `OK`.
Exit 3 means the result is inconclusive (`DEGRADED DATA`, no valid reference,
or partial evidence coverage); automation must not interpret it as healthy.

## `dedrift report [--out FILE]`

Renders the deterministic markdown report: verdicts, alert table in plain
units, attribution, heuristic flags (clearly labeled), per-family sparklines,
config timeline, and an appendix of every non-alerting result.

## `dedrift signatures [--by family|canary]`

Prints the aggregated signature tables (mean / variance / P95 per signature
per cycle) for eyeballing trajectories.
