# dedrift

**Agents don't throw errors when they degrade. They keep confidently producing worse outputs. dedrift catches it.**

dedrift is an open-source Python package that detects silent behavioral drift in AI agents. It logs agent interactions, runs a frozen canary suite repeatedly, extracts behavioral signatures, and applies statistically rigorous drift detection with config-change attribution ("behavior shifted within 6h of the model version change").

## Why dedrift

Model updates, prompt edits, tool-schema changes, RAG index refreshes, and provider-side silent updates all shift agent behavior without throwing a single error. Existing monitoring catches crashes, not character changes.

dedrift's differentiation is statistical correctness:

- Every detector controls its documented false-alarm rate — validated by simulation tests in CI.
- All alerting passes through FDR control (Benjamini–Hochberg). No raw per-test p-values dressed up as alerts.
- Every alert requires both statistical significance and a configurable effect-size (materiality) threshold. Fewer, higher-confidence alerts.
- LLM outputs are stochastic: canaries run N times per cycle and we compare distributions, never single outputs.
- Dual baselines: every check runs against a rolling recent window (sudden breaks) and a frozen golden baseline (boiling-frog drift).
- Honest about power: small N means low detection power, and the docs show you the math instead of hiding it.

## Status

Pre-alpha. Phase 0 (scaffold, logging schema, storage, simulator) in progress. See `ROADMAP.md`.

## Install

```bash
pip install -e .          # core: zero ML dependencies
pip install -e ".[embeddings]"   # + semantic signatures (sentence-transformers)
pip install -e ".[judge]"        # + LLM-judge tier
pip install -e ".[dev]"          # development tooling
```

## Quickstart (target v0 walkthrough)

```bash
dedrift init                 # create a project
# ... log agent interactions, run canaries ...
dedrift check                # drift detection with FDR + materiality gating
dedrift report               # deterministic markdown report with attribution
```

A full simulated demo (synthetic agent, mid-log model swap, detection + attribution) ships with v0.1.0.

## dedrift Pro

Anytime-valid sequential inference (e-processes), conditional production-traffic drift, and importance weighting are part of a separate commercial tier and are not in this repository.

## License

AGPL-3.0-only. See `LICENSE`.
