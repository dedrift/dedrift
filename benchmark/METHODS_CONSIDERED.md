# Methods considered for the null-calibration benchmark

Every candidate was installed or fetched and inspected before a decision.
Measured methods are in `README.md`; this file records the candidates that
were **attempted and excluded**, with the exact, checkable reason in each
case. Exclusion here is not a negative judgment of a tool: several were
excluded precisely because measuring them would require using them for
something their authors did not design them to do.

## Driftbase 0.15.1 — excluded: public write path is disconnected from the read path

Attempted integration: `driftbase.backends.sqlite.SQLiteBackend.write_runs()`
accepts our interaction records mapped onto its run schema (tool sequence,
output length, latency, error/retry counts, token counts — all fields our
records genuinely carry), but `driftbase.engine.compute_drift()` never sees
those rows:

- `write_runs()` writes the legacy `agent_runs_local` table.
- `compute_drift()` reads via `backends/sqlite_reader.py`, which queries the
  newer `runs_raw`/`runs_features` tables and, by default, filters to
  `ingestion_source = 'connector'` — i.e. traces imported from Langfuse /
  LangSmith / OTEL. `compute_drift()` exposes no flag to widen this.
- The one-time v0.11 migration copies `agent_runs_local` into `runs_raw`
  only at backend open; rows written afterwards are never migrated, and
  re-opening does not re-run it (verified: write 1 row, re-open,
  `get_runs()` returns 0).

Measuring Driftbase would therefore require either running a third-party
tracing server or inserting rows into internal tables by hand — both are
distortions this study's protocol forbids. We verified the finding on a
clean install before excluding. Driftbase's documentation describes
sample-size gating (its "confidence tiers" suppress verdicts below minimum
run counts), which is exactly the kind of validity-scale refusal this study
argues for; we could not measure its false-alarm rate at defaults and say
nothing about it.

## Nautilus Compass 2.3.0 — excluded: estimand mismatch (within-session persona drift)

The package (Wang; the project cited as `wang2026compass` in the paper)
detects *within-session* drift of an assistant's reply sequence: language
switching, style/length drift, and persona drift as cosine distance from a
first-turn anchor (`output_drift.py`, `PERSONA_COS_MIN = 0.60`, triggered on
sessions of five or more replies). Our histories are independent
single-turn canary repetitions on frozen inputs — there are no sessions and
no persona to drift from. Running Compass would require fabricating
sessions out of canary repetitions, which measures a question Compass does
not ask. (Practical note: 2.3.0 also requires `sentence-transformers`,
which conflicts with the benchmark environment; this was secondary to the
estimand mismatch.)

## UpTrain 0.7.1 — excluded: the drift operator cannot run under any installable pydantic

`uptrain.operators.drift.ConceptDrift` (DDM/ADWIN over a streamed measure
column) is the one candidate whose streaming estimand would fit a per-family
canary stream. It cannot be executed at its pinned release:

- Python 3.13: the dependency chain fails to build (`litellm`).
- Python 3.12 with `--no-deps` + `polars`, `river`, `loguru`: the operator
  mixes pydantic v1 and v2 APIs — `@root_validator` without
  `skip_on_failure` (a hard `PydanticUserError` on every pydantic 2.x we
  tested: 2.13, 2.5.3, 2.1.1) and v2-only `.model_dump()` (absent in
  pydantic 1.10).

Patching the library would violate the defaults-only protocol, so it is
excluded with the failure recorded rather than repaired.

## whylogs 1.6.4 — excluded: no programmatic drift-alarm API at defaults

Profiles compute per-feature distribution summaries and the library ships
drift *visualization* utilities; the alarming decision (thresholds,
monitors) lives in the hosted WhyLabs configuration surface. There is no
documented, callable, default-thresholded drift verdict in the open-source
API to measure against, and inventing one would be our configuration, not
theirs.

## llm-drift, drift_orchestrator — excluded: not pip-installable

Both are cited in the paper as emerging agent-drift tooling
(GitHub-only projects). The study's inclusion rule is "pip-installable with
a callable API", so histories can be consumed without distortion; neither
has a package release, and wrapping a repository snapshot would not be a
pinned, reproducible default configuration.

## DeDrift (Baranchuk et al., ICCV 2023) — namesake, not a candidate

"DeDrift: Robust Similarity Search under Content Drift" (arXiv:2308.02752)
*adapts* ANN index quantizers as embedding distributions shift. It is not a
drift *detector*: no tests, no alerts, no false-alarm notion, and no public
code. It shares only the name with this package. The paper carries a
one-line disambiguation footnote; there is nothing here to measure.
