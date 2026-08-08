# Right-of-reply drafts — DO NOT SEND automatically

These are drafts for the owner to review and send. Nothing here has been
sent. Publication of the benchmark page and the paper's validity-scale
section waits for the owner's decision after the stated window.

---

## Draft 1 — Evidently maintainers (right of reply, 14-day window)

**From:** Ali Mahmoudi <alimahmoudidev26@gmail.com>
**To:** Evidently AI team (hello@evidentlyai.com, or a GitHub discussion on
evidentlyai/evidently if you prefer)
**Subject:** Measured null false-alarm rates at canary scale in an upcoming
paper — scripts and right of reply before publication

Hi Evidently team,

I'm the author of dedrift (github.com/dedrift/dedrift), an open-source
behavioral drift detector for LLM agents, preparing a methods paper for
arXiv. One section is a **validity-scale study**: we measure how often
common drift-detection configurations alert on data with *no drift at all*,
at the window sizes agent canary suites actually produce (tens to ~a
hundred rows per window). It is explicitly framed as not a head-to-head:
the paper states that your estimand (input-distribution drift over
production tables) differs from ours (behavior-given-fixed-input under a
designed experiment) and that the two are complementary. Because Evidently
is the reference open-source framework, and because your documented route
for LLM-output drift (per-output descriptors + drift methods over the
descriptor table) has exactly the shape of our tables, its default preset
is the realistic instance of the misapplication.

What was measured — **Evidently 0.7.21, `DataDriftPreset`, all defaults** —
on 500 seeded synthetic stable-agent histories (identical generators to my
own tool's published null suite; scripts and raw results attached,
everything reproducible from fixed seeds 0–499):

| Granularity | 18×7 scale | 12×5 scale |
|---|---|---|
| Pooled table, ≥ 1 drifted column | 356/500 = 71.2% [67.1, 75.0] | 329/500 = 65.8% [61.5, 69.8] |
| Pooled table, dataset verdict (share ≥ 0.5) | 2/500 = 0.4% [0.1, 1.4] | 0/500 = 0.0% [0.0, 0.8] |
| Per canary family, ≥ 1 drifted column | 498/500 = 99.6% [98.6, 99.9] | 500/500 = 100% [99.2, 100] |

(Wilson 95% intervals in brackets.)

The per-column measurement attributes the excess over the 1 − 0.95¹²
independence arithmetic precisely: the test-selection heuristic routes the
three low-cardinality integer columns (tool_call_count, steps,
tool_order_inversions) to a chi-square test whose small-cell validity
condition is violated at these window sizes, and those columns reject at
**28–47%** each on unchanged data, while every column routed to a p-value
test (K-S, z) sits at 2–5%. The dataset-level verdict is quiet on stable
agents; we also note plainly that conservatism says nothing about power,
which this study does not measure.

The section makes the same critique of the PSI metric in general and of my
own tool's uncalibrated diagnostic flag channel, whose measured false-flag
rate is printed in the same table. It also states that your LLM evaluation
surface (judge- and rule-based scoring) answers a different question and
was not tested.

Three things I'd specifically welcome correction on:

1. Is there any configuration in current Evidently that applies
   multiplicity control across per-column drift decisions? I found none in
   the surface I exercised; the paper scopes the claim to exactly that.
2. For drift over LLM-output descriptor tables at this scale, is
   `DataDriftPreset` at defaults a fair representation of the documented
   route, or is there a configuration you would regard as the intended
   baseline?
3. The chi-square routing for low-cardinality integer columns at n ≈ 60–126
   per window: is the ~30–47% null rejection we measure a known small-n
   behavior of the test-selection heuristic? If so I'd welcome the
   reference; if it is surprising, the reproduction is attached.

I'm happy to include a response or correction from you in the paper and on
the benchmark page, credited and unedited, beside the table it addresses.
If I have misconfigured something or misread a default, I would much rather
fix it now than in a v2. **The window I am holding before publication is 14
days from your receipt of this mail**; silence is not taken as endorsement,
only as no-comment.

Best,
Ali Mahmoudi
dedrift.ai · github.com/dedrift/dedrift

*Attachments: benchmark/ scripts (methods.py, run.py), benchmark/results/
(raw per-run JSON, both scales), benchmark/METHODS_CONSIDERED.md*

---

## Draft 2 — Driftbase (integration finding; not a benchmark item)

**From:** Ali Mahmoudi <alimahmoudidev26@gmail.com>
**To:** Driftbase team (info@driftbase.io)
**Subject:** Integration report: compute_drift() cannot see locally written
runs (0.15.1)

Hi Driftbase team,

While preparing a null-calibration benchmark of drift-detection tooling I
attempted to include Driftbase 0.15.1 and could not wire it end-to-end at
its public API; reporting in case it is useful. We excluded Driftbase from
the measurement rather than work around this — recording the reason
publicly, in a "methods considered" note, so the exclusion is checkable.

What I found, on a clean install (Python 3.13, driftbase 0.15.1):

1. `SQLiteBackend.write_runs()` writes the legacy `agent_runs_local` table.
2. `compute_drift()` reads through `backends/sqlite_reader.py`, which
   queries `runs_raw`/`runs_features` and, by default, filters to
   `ingestion_source = 'connector'` (Langfuse/LangSmith/OTEL imports).
   `compute_drift()` exposes no flag to widen that filter.
3. The v0.11 migration copies `agent_runs_local` → `runs_raw` only at
   backend open; rows written afterwards are never migrated (verified:
   write one row, re-open, `get_runs()` returns 0).

Net effect: with only local writes (no external tracing backend),
`compute_drift()` raises "No runs found". If there is an intended offline
path I missed, I'd genuinely like to know — the benchmark page is versioned
and re-run, and a corrected integration would be included in the next
revision, credited.

Two positive notes that are also in the public record: the documented
confidence tiers (suppressing verdicts below minimum run counts) are
exactly the kind of validity-scale guard the benchmark argues for; and the
run schema mapped cleanly onto our interaction records — the blocker is
strictly the write/read split above.

Best,
Ali Mahmoudi
dedrift.ai · github.com/dedrift/dedrift

*Attachment: benchmark/METHODS_CONSIDERED.md (the public note as shipped)*
