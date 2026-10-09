# Dedrift production-readiness review

Date: 2026-10-10
Scope: `review04`, package, CLI, storage, fixed and anytime inference, reports,
documentation, benchmark evidence, and product positioning.

## Decision

**No-go for a general production agent-drift monitor or pager.**

**Conditional go for a time-boxed pivot to a CI/nightly behavioral regression
and release gate.** The useful core is frozen canaries, explicit baselines,
typed domain signals, reproducible evidence, and fail-closed detector health.
That is narrower than the current category claim, but it is real.

Do not build a generic Python agent framework in this repository. It would
compete with mature orchestration libraries, dilute the only differentiated
asset, and add no evidence that users need another framework. Build a small,
stable monitoring SDK only after the release-gate workflow earns adoption.

## Evidence, without the flattering interpretation

| Scenario | Observed result | Honest interpretation |
|---|---:|---|
| README synthetic model swap, 18 canaries × 7 repetitions, 8 cycles | Drift detected; 110 alert rows; 742-line/65 KB report; about 7.1 MB stored | A gross scripted change is detectable, but incident output is too noisy for operations |
| Stable default suite, one fixed check | 8/500 histories had any false alert (1.6%) | Potentially acceptable as one investigated release gate; not evidence for continuous monitoring |
| Stable default suite, 50 fixed checks | 358/500 histories ever false-alerted (71.6%) | The default fixed mode is unsuitable as a recurring pager |
| Stable small suite, one fixed check | 16/500 histories had any false alert (3.2%) | This one-check number is incomplete without repeated-use context |
| Stable small suite, 50 fixed checks | 420/500 histories ever false-alerted (84.0%) | Repeated fixed checks will usually create operator noise |
| +2 percentage-point refusal shift at default N=7 | Approximate power 0.11 before FDR | Small business-relevant changes are usually missed at this sample size |
| Tool-order reversal | Detected on 8/20 first checks with the channel; 0/30 without it | The signal adds value, but first-check sensitivity was only about 40% in this scenario |
| Corrected anytime path | Rate channels only; no replacement end-to-end null/power table | The math path is promising research, not an operationally validated monitor |
| External numeric-agent deployment ([issue #12](https://github.com/dedrift/dedrift/issues/12)) | 13/15 alerts were latency noise; the real regression was missed; package removed after six days | The only identifiable field evidence is negative and must outweigh synthetic demos |
| Public adoption as reviewed | 0 stars, 0 forks, 0 watchers; roughly 58 recent monthly PyPI downloads | There is no demonstrated product-market fit |

The committed benchmark remains useful engineering evidence, but it does not
establish performance under provider-correlated cycles, real incident mixes,
or an end-to-end semantic evaluator. Benchmark artifacts must identify the
exact commit, release, environment, and detector configuration before they can
support release claims.

## Final branch verification

| Verification | Result |
|---|---:|
| Unit/integration suite excluding marked simulations | 421 passed |
| Marked null-calibration and power simulations | 26 passed |
| Ruff / mypy / whitespace checks | Passed |
| Wheel built from sdist, fresh-venv install, import, `pip check`, CLI help | Passed |
| Installed-wheel demo | 1,680 records; both baselines detected the scripted change; 110 raw alert rows |
| Doctor after the demo check | 0 failures; full fixed-path channel/family coverage; warnings retained |
| Strict documentation build and Twine artifact validation | Passed |
| Distribution content and secret/private-path scan | Passed |

The 110 alert rows are a test-orchestration result, not a desirable incident
experience. Passing calibration tests validates only their declared synthetic
models and acceptance bands; it does not resolve the dependence and field-power
blockers below.

## Material defects fixed in this branch

These were real correctness or false-success defects, not refactoring:

- rejected SQLite integer overflows before JSONL publication, including
  expected-cycle counts;
- rejected duplicate canary coordinates during streaming before they can
  leave a permanently unfinalizable log tail;
- rejected duplicate crash-tail records on the first reopen and verified
  JSONL/SQLite record order, digest, and payload instead of allowing the two
  stores to describe different histories;
- revalidated a concurrent embedder pin before persisting a fixed check;
- revalidated the same pin inside the anytime writer transaction before epoch
  declaration;
- made the embedder pin authoritative in SQLite and treated a missing or
  conflicting pin file as corruption;
- counted suppressed hypotheses in the prospectively declared BH family;
- included detector/test identity in persistence streaks;
- made missing declared scalar channels produce partial coverage instead of a
  green result;
- added a native-unit materiality floor and an explicit latency-channel opt-out;
- rejected imported non-callable agent attributes;
- made canary runs with adapter/agent errors exit non-zero while retaining the
  diagnostic records;
- reserved anytime cycles durably before the first agent call, checkpointed
  each completed coordinate, and rejected post-hoc imports as prospective
  evidence;
- added a strictly non-mutating `dedrift doctor` that reads a private
  consistency-checked SQLite/WAL snapshot and detects stale baseline, config,
  embedder, suite, coverage, and epoch evidence;
- escaped untrusted report values as Markdown and control text;
- validated project identity on repeated `init` instead of returning false
  success for an unrelated SQLite file.

Passing tests prove these regressions are closed in the tested configurations.
They do **not** upgrade the whole product to production-ready.

## Remaining blockers

### Statistical model

1. Repetitions within one hosted-provider cycle are commonly correlated. The
   default fixed path and current rate e-process do not establish their stated
   operating characteristics under that dependence. The project needs inference
   whose independent unit is a cycle, plus measured null, power, and delay.
2. The active anytime epoch now enforces durable pre-observation registration,
   but golden baselines still need a prospective campaign workflow such as
   “the next N completed cycles.” Choosing attractive historical cycles is
   exploratory, not confirmatory, in fixed mode as well as anytime mode.
3. The corrected anytime implementation needs realistic end-to-end calibration;
   rate-only mathematical validity is not the same as useful detection power.

### Operations and lifecycle

1. An abandoned open cycle blocks the contiguous anytime stream. Add visible
   blocked-cycle diagnostics and a prospectively declared, auditable
   abort/expiry policy; an operator must not inspect an outcome and then decide
   whether it counts.
2. Canary execution needs deadlines, bounded concurrency, retry policy,
   checkpoint/resume, and deterministic coordinate idempotency.
3. Add a consistent whole-project snapshot/archive/restore command with a
   hashed manifest. JSONL, SQLite/WAL, config, baseline, pin, and caches are one
   recovery unit.
4. Add schema versions and downgrade refusal, retention and redaction controls,
   raw-payload size limits, privacy guidance, and migration fixtures.
5. Sentence-transformer pins need immutable artifact revisions/hashes and
   output-dimension/runtime provenance; a mutable Hub name is not a forever pin.
6. Full-history checks and whole-NPZ embedding-cache rewrites will not scale.
   Add bounded indexed windows/checkpoints and per-record embedding storage,
   then publish load and soak results.

### Product and incident UX

1. `dedrift doctor` and fixed reports now expose detector health and
   required-versus-observed channels. Persist the same structured health model
   for every check mode and API consumer; “no alerts” must never mean both
   healthy and blind.
2. Aggregate correlated test rows into an incident summary. Keep raw tests in
   an appendix and provide a machine-readable persisted result (JSON first;
   SARIF/JUnit/webhook later).
3. Persist a check once and render it by check ID. `check` and `report` should
   not recompute the same inference and create duplicate incidents.
4. Attribution must refuse confidence when prompt/tool/RAG/version identities
   are unset. Automatically hash supplied artifacts where possible.

### Release and security

Before another public release: require a pull request and green CI on the exact
commit, synchronize package/site/docs versions, correct the live withdrawn
anytime documentation, publish a security policy, protect `main`, pin release
inputs, generate an SBOM/provenance, and rerun the benchmark on the release
artifact. Plaintext prompt/output storage also rules out many enterprise uses
until redaction, retention, access, and encryption integration are designed.

## Jev / RLCD decision

[Jev/System One](https://docs.typesafe.ai/concepts/system-one) is a typed
judgment model: it answers Choice, Score, and Noul questions with decisions or
probabilities instead of generating prose. RLCD is TypeSafe's method for
training calibrated decisions. Their own documentation correctly frames
calibration as an aggregate property, not a guarantee that an individual
judgment is true.

Jev is **complementary**, not a replacement for Dedrift:

- Jev can turn semantic properties—policy compliance, correct tool choice,
  answer quality—into typed numeric signals that Dedrift compares over time.
- Jev does not provide frozen-suite history, baselines, sequential
  multiplicity, onset analysis, release decisions, or monitoring lifecycle.
- Dedrift's regex/length/hash signals are too weak for many real regressions;
  optional typed evaluator signals are a plausible improvement.

Integration requirements:

1. Define an evaluator-agnostic typed signal interface first: boolean,
   probability, bounded score, and categorical probability vector.
2. Record evaluator provider, immutable model version, instructions, criteria,
   and option order separately from the agent-under-test identity.
3. Pin a versioned Jev model; moving aliases can themselves create drift.
4. Validate calibration, thresholds, option permutations, adversarial content,
   and accuracy on each customer's domain. Do not use Jev for arithmetic/date
   truth that deterministic code can compute.
5. Keep Jev optional. A hosted third-party evaluator adds availability,
   privacy, version, injection, and vendor-dependence risks.

The current absence of a TypeSafe API key means no live Jev claim has been
tested in this review. Documentation analysis is not a performance benchmark.

## Six-to-eight-week kill-or-continue gate

Continue only if all of the following are achieved:

- 3–5 external teams use the release gate weekly on real changes;
- at least two become paid pilots or sign a credible letter of intent;
- each partner replays at least one historical incident and agrees, before the
  test, on a target false-block rate and incident-detection target;
- the tool catches useful incidents without blind-green health states and the
  teams retain it for at least 30 days;
- the revised cycle-level inference has published, reproducible null, power,
  and delay results on the exact release artifact.

If this gate is missed, stop platform investment. Keep the work as an honest
research/open-source regression library or archive it. A dashboard, agent
framework, and more marketing will not create statistical validity or demand.

## Commercial hypothesis, if the gate passes

Start with a paid service pilot (roughly USD 2,000–5,000) for canary-suite
design, historical incident replay, threshold calibration, and CI integration.
Do not charge for a vague “AI reliability platform” before repeated value is
observed. A later hosted product can monetize scheduled orchestration, private
runners, team history/approvals, audit trails, SSO, retention controls, and an
SLA. The statistical core alone is not yet a defensible paid tier.
