# How it works

## The core loop

1. **Log** every agent interaction as an `InteractionRecord` — input, output,
   tool calls, latency, tokens, errors, and a full config block (model,
   prompt hash, tool-schema hash, RAG index version, agent version). The
   config block is hashed into a fingerprint; any fingerprint change becomes
   a **config event** on the project timeline.
2. **Run canaries.** A canary suite is a frozen YAML set of 50–200 inputs,
   stratified by family: `happy_path`, `edge_case`, `refusal_boundary`,
   `tool_heavy`, `adversarial`, `long_context`. Each canary runs **N times
   per cycle** (default 7) against your agent via a one-function adapter.
   dedrift never implements your agent; it only calls it.
3. **Extract signatures.** Per record: output length, format validity,
   refusal flag, tool-call counts and schema conformity, steps, retries,
   errors, latency, exact-match on expected fields (Tier 1 — always on, zero
   ML dependencies). With a pinned embedder: semantic displacement and pooled
   embeddings per family (Tier 2 — optional extra).
4. **Check.** The current cycle is compared against **two references at
   once**: the rolling window of recent cycles (catches sudden breaks) and a
   frozen golden baseline you declared known-good (catches slow boiling-frog
   drift). Both verdicts are always reported.
5. **Attribute.** For each alert, dedrift estimates the drift onset (the
   Page–Hinkley change-point where available, else the first alerting cycle)
   and ranks config events by temporal proximity. Attribution is
   correlational and the report says so — "consistent with", never
   "caused by".
6. **Report.** Deterministic markdown: same logs + same config produce an
   identical report. Seeds for every resampling procedure are recorded in it.

## Why fixed canaries?

With production traffic, inputs drift too, so separating input drift from
behavior-given-input drift requires conditional comparison — a genuinely
harder problem. A frozen canary suite removes input variation by
construction: any output-distribution shift is attributable to the stack.
That is the honest v0 scope. (Conditional production-traffic drift is on the
roadmap as a separate tier.)

## Cycles are the unit of comparison

One cycle = one full suite execution. Records carry an explicit `cycle_id`,
and every statistical comparison is between the current cycle's distribution
and a reference window's distribution — never between individual outputs,
because individual LLM outputs are random.

## The embedder is pinned forever

Changing the embedding model invalidates every historical semantic
comparison. dedrift therefore pins the embedder per project on first use and
**refuses** to compare across embedder identities — re-pinning to a different
model or mixing caches raises an error rather than silently producing
garbage. The zero-dependency `hash` embedder ships for demos and tests;
`st:<model>` uses any sentence-transformers model via the
`dedrift[embeddings]` extra.
