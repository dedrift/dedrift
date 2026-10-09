# How it works

## The core loop

1. **Log** every agent interaction as an `InteractionRecord` — input, output,
   tool calls, latency, tokens, errors, and a full config block (model,
   prompt hash, tool-schema hash, RAG index version, agent version). The
   config block is hashed into a fingerprint; any fingerprint change becomes
   a **config event** on the project timeline.
2. **Run canaries.** A canary suite is a frozen YAML set of inputs,
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
   once**: the rolling window of recent cycles (aimed at sudden breaks) and a
   frozen golden baseline you declared known-good (aimed at cumulative
   movement). Both verdicts are reported when they have valid references.
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
construction. It narrows the suspects, but it does not prove that the recorded
stack change caused an output-distribution shift: provider state, sampling,
tools, external data, and collection defects can still move. Frozen-canary
release comparison is the honest v0 scope. Conditional production-traffic
drift is not implemented.

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

## Optional typed-evaluator channels

For qualities that the built-in structural signatures cannot observe, an
external typed evaluator can emit a numeric score or probability into
`output.structured`, and `project.custom_scalars` can admit that value to the
battery. A [Jev/Noul decision](https://docs.typesafe.ai/models) is one possible
source of such a channel.

This is an architectural option, **not a shipped or validated Jev
integration**. [RLCD calibration](https://docs.typesafe.ai/confidence) is
aggregate evidence about an evaluator, not
a guarantee for each decision or for your domain. The evaluator also becomes
part of the measurement instrument: model, prompt, option order, and vendor
changes can look like agent drift. Pin its exact version, fingerprint its
configuration, retain the raw output, and validate calibration and power on
labeled project data before using it in a release gate. A moving
`jev-latest` alias would make the measuring stick move with the system being
measured.
