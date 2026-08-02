# Below the validity scale

Most drift tooling was built for production data tables with thousands of
rows per window. Agent behavioral monitoring runs on **tens** of rows per
window — repeated canary runs, small by economic necessity. This page shows,
with measurements you can reproduce, what happens when standard drift
machinery is applied below the scale it was designed for — and why dedrift's
answer is to **refuse** that regime rather than emit noise into it.

!!! note "This is not a head-to-head"
    The tools measured here target a different estimand than dedrift.
    Evidently asks whether a *data distribution* changed — observational
    monitoring of production tables. dedrift holds inputs fixed by design
    and asks whether *behavior given those inputs* changed. The two are
    complementary. What follows is a **misapplication study**: the failure
    modes of general-purpose drift workflows in the small-n behavioral
    regime, measured because that misapplication is realistic — it is the
    default thing a team does when no behavioral tool is in place.

## Exhibit 1: PSI, an index that cannot say "stable" at small n

The Population Stability Index is the industry's favorite drift score, with
folk thresholds of 0.1 ("moderate") and 0.25 ("major"). PSI between two
finite samples of the **same** distribution is not zero — to first order,

> E[PSI] ≈ (B − 1) · (1/n + 1/m)

for B bins and window sizes n, m. At canary scale — 10 bins, 30
reference rows, 10 current rows — that expectation is **≈ 1.2: five times
the "major" threshold, from sampling noise alone, on unchanged data.**

We measured the consequence: before dedrift's validity guard, PSI flagged
**100% of 500 stable-agent checks**. The guard now computes the null
expectation and refuses to emit PSI where noise alone can cross the
thresholds (roughly n ≳ 360 per window). PSI remains available at the
production scale it was designed for; it just no longer pretends to work
below it.

## Exhibit 2: a standard drift workflow at canary scale

We ran [Evidently](https://github.com/evidentlyai/evidently) (v0.7.21,
`DataDriftPreset`, **all defaults**) on the *same 500 seeded stable-agent
histories* used for dedrift's own published null bound, feeding it the same
behavioral signature tables (reference: 180 rows; current: 60). This is the
shape of Evidently's own documented route for LLM-output drift — per-output
descriptors, then the drift methods over the descriptor table. Its defaults
test every column independently at p < 0.05, with no multiplicity control.

![Any-alarm rates on identical stable-agent histories](assets/fig4_evidently.png)

| Alarm | Stable agents | Simulated model swap | Real model swap |
|---|---|---|---|
| Evidently: any drifted column | 249/500 (49.8%) | 200/200 | fires (2/11 columns) |
| Evidently: dataset-level verdict (≥50% of columns) | 1/500 (0.2%) | 136/200 (68%) | **does not fire** (share 0.18) |
| dedrift: gated alerts (BH-FDR + materiality) | 7/500 (1.4%) | 200/200 | fires (both baselines, 36 alerts) |

Below its validity scale the workflow fails **in both directions at once**:

- The per-column channel false-alarms on half of stable checks pooled — and
  on **every single one** at per-family granularity. The excess over the
  textbook arithmetic (1 − 0.95¹¹ ≈ 43%) traces to two low-cardinality
  integer columns whose auto-selected test falsely rejects **30.5%** of the
  time each; a battery is only as calibrated as its worst member, and
  nothing in an uncorrected per-column report tells you which member that
  is.
- The dataset-level verdict — quiet on stable agents — requires half of all
  columns to move at once, so it **missed 32% of simulated model swaps and
  the real one** (only 2 of 11 columns moved in the [case
  study](case-study.md), while dedrift raised 36 gated alerts on both
  baselines).

The finding is not that Evidently's tests are wrong — they are the standard
ones, and Evidently's data-drift reports at production scale are a fine
tool. The finding is that **nothing in the workflow refuses the regime
where its decisions are noise.** Refusal below the validity scale is a
design decision, and it is the one this whole page argues for.

All numbers are at defaults, measured on our tables, and every claim is
scoped to the configuration surface we exercised (v0.7.21). The benchmark
scripts and raw results ship in the repository and reproduce every number
from fixed seeds.

## What dedrift does instead

One primary test per behavioral channel enters a single
[Benjamini–Hochberg pool](statistics.md); alerts additionally require a
materiality gate on the same scale the test reports; PSI is
validity-guarded; comparisons whose windows aren't composition-comparable
are suppressed and reported as such; and the whole pipeline's null alert
rate is [measured in CI](statistics.md) — 7/500 stable checks, Wilson 95%
upper bound 0.029 — at a stated scale, alongside its measured limitations.
