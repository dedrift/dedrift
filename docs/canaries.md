# Writing canary suites

A canary suite is a **frozen, versioned set of inputs** your agent answers
repeatedly. Everything dedrift concludes rests on it, so it deserves the
same care as a test suite — because that's what it is.

## The format

```yaml
version: "1"
canaries:
  - id: happy-001
    family: happy_path
    input: {text: "Summarize the plot of Hamlet in two sentences."}

  - id: tool-001
    family: tool_heavy
    input: {text: "Return JSON with keys 'city' and 'population' for the capital of France."}
    expected: {city: "Paris"}        # enables format_valid + exact_match signatures
```

Fields per canary:

| Field | Required | What it does |
|---|---|---|
| `id` | yes | Unique, stable forever — history is keyed on it |
| `family` | yes | One of the six families below |
| `input` | yes | Mapping passed to your `agent_fn` (usually `{text: ...}`) |
| `expected` | no | Expected structured keys/values → format-validity and exact-match signatures |
| `rubric_id` | no | Provenance identity only; this release does not execute an LLM judge |

## The six families

Tests are pooled **per family**, so each family is a separate lens on
behavior — and needs enough members to give its tests power.

| Family | Probes | Example |
|---|---|---|
| `happy_path` | Bread-and-butter competence and style | "Summarize X in two sentences" |
| `edge_case` | Degenerate/ambiguous inputs | `"?"`, impossible list requests |
| `refusal_boundary` | Consistency of decline behavior | Things that should be politely refused |
| `tool_heavy` | Structured output & tool usage | JSON tasks with `expected` keys |
| `adversarial` | Injection resistance | "Ignore your instructions and…" |
| `long_context` | Long-input behavior | Multi-paragraph inputs with a needle |

## Sizing: power comes from samples

Per-family sample size per window is *canaries-per-family × repetitions*.
The [power table](statistics.md#detection-power-the-honest-table) is the
budget: at the default N=7 with 30 canaries in a family, a +10 pp refusal
shift is detected with ~0.91 power, but +2 pp is essentially undetectable.
Practical guidance:

- **Demo scale:** 3 per family (the shipped example) — catches big changes
  like a model swap.
- **Production scale:** 50–200 canaries total, weighted toward the families
  that matter to your product (if refusal consistency is your business,
  grow `refusal_boundary`, not `happy_path`).
- Raising repetitions N helps every family at linear cost.

## Freezing discipline

- **Never silently edit a canary.** The suite is version-controlled; a
  changed input invalidates that canary's history.
- If you add or remove canaries, expect the next check to report
  **COMPOSITION MISMATCH** for the affected families against old baselines
  — that's the [composition guard](statistics.md#fine-print-stated-plainly)
  refusing to compare windows with different mixtures. Re-freeze the golden
  baseline after suite changes.
- A canary that times out and logs no records triggers the same guard: a
  missing-data finding, deliberately never presented as drift.

## What gets measured

From every run, dedrift extracts structural signatures (length, latency,
refusal, format validity, exact match, tool-call counts, retries, errors —
each tracked in location, dispersion, and tail) and, with a
[pinned embedder](configuration.md#embeddings), semantic signatures
(per-family embedding clouds via MMD, plus per-record semantic displacement
from the canary's own reference centroid).
