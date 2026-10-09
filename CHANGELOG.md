# Changelog

## 0.5.1

Safety-correction release for anytime inference. The 0.4.0 and 0.5.0
documentation treated the `twosample` beta-binomial prior-predictive Bayes
factor as an e-value for every fixed unknown null rate after observing the
reference. That claim is false. For example, with 0 successes in 5 frozen
reference trials and one current Bernoulli trial at the stable null `p = 0.5`,
the claimed factor has expectation 2.0417, not at most 1. Realistic
fixed-reference simulations also exceeded the advertised lifetime budget.

- Withdrew `anytime.rate_model = "twosample"`. It is now rejected during
  configuration validation rather than accepted as a production option.
- Removed the pooled per-signature products. Multiplying dependent
  per-family e-values did not preserve the e-value property without the
  independence that the production streams do not have.
- Removed repeated e-BH from the production decision. The only supported
  model is now `frozen_cp`, the frozen Clopper–Pearson worst-case construction.
- Froze the process pool once per epoch and split both budgets across its `K`
  members: `gamma_total / K` per reference interval and
  `alpha_prime / K`, where `alpha_prime = alpha - gamma_total`, per persistent
  Ville crossing. A Bonferroni union bound handles dependence between
  processes. Once crossed, a process stays crossed for the rest of the epoch,
  including when one check folds a backlog of cycles.
- Kept the scope narrow: outcome-independently selected golden baseline, rate
  channels, finalized cycles, and the stated fixed-rate Bernoulli assumptions.
  Outcome-independently unavailable observations contribute a neutral factor;
  composition mismatches fail closed rather than changing the pool.
- Retracted the old `twosample` null-rate, wobble-boundary, and power
  headlines. They must not be used to size or approve a deployment. The
  corrected construction is substantially more conservative at canary scale,
  and 0.5.1 makes no replacement null-rate or power claim.
- Made epoch creation prospective: the first anytime check declares the pool
  after every canary cycle already observed (including open/crash-tail cycles)
  and folds no history. Only cycles first recorded after that durable watermark
  may contribute evidence; later backlog folding remains exactly-once.
- Made e-process time follow durable cycle-open order. A finalized cycle now
  waits behind any earlier open cycle, so completion order cannot permute the
  wealth path or manufacture an absorbing crossing. Suite identity and family
  composition are evaluated only for the golden reference and eligible
  cycles; partial suite fingerprints fail closed.
- Made degraded-cycle warnings persistent for the whole epoch because later
  wealth and alerts still contain that evidence. Excluded or deferred cycles
  cannot produce a supported green status.
- Retracted the fixed-path `cycle_effect="auto"` calibration/power ladder after
  correctness changes invalidated its measurements. No replacement `auto`
  performance table is claimed in 0.5.1.
- Corrected fixed-path semantic displacement to use reference-only centroids,
  made cluster-aware corrections conservative relative to the exact path, and
  tied alert-persistence streaks to distinct cycles and a detector fingerprint.
- Replaced duplicated full snapshot-ID manifests in every check row with the
  compact finalized-revision watermark; exact snapshots reconstruct from
  immutable cycle order. The remaining O(history)-per-check / O(cycles²)
  cumulative compute cost is documented as a pre-1.0 production blocker.
- Hardened store recovery and concurrency: lifecycle/check/anytime state is
  treated as authoritative, schema migration and project initialization are
  serialized, finalized cycles are immutable, snapshot races retry, and a
  missing, empty, unrelated, or incompatible SQLite index is never silently
  rebuilt from the interaction log.
- Closed additional review blockers before release: SQLite-sized integer and
  expected-count validation happens before JSONL publication; duplicate
  canary coordinates are rejected for streaming as well as finalization;
  indexed JSONL prefixes are integrity-checked on open; embedder pins are
  authoritative lifecycle state and are revalidated at check commit.
- Fixed current-data-dependent multiplicity shrinkage by retaining the
  prospectively declared primary family size when families or channels are
  suppressed, and included the exact statistical test in persistence keys.
- Added channel-level evidence coverage, a native-unit scalar materiality
  floor, and a latency opt-out. A declared custom channel that disappears can
  no longer return a green fixed result.
- Added durable pre-observation cycle declaration for anytime inference.
  Runner-managed cycles are declared before the first agent call and
  checkpointed record by record; post-hoc batches after an active epoch fail
  closed rather than masquerading as future evidence.
- Added `dedrift doctor`, report health output, non-callable/degraded-run CLI
  failures, Markdown-injection escaping, and honest release-gate positioning.

Upgrade note: an explicit `rate_model = "twosample"` now fails closed. Remove
that setting or replace it with `rate_model = "frozen_cp"`, then review the
coverage/multiplicity split and the limitations in `docs/anytime.md` before
enabling anytime alerts.

## 0.5.0

Field-report release. [#9](https://github.com/dedrift/dedrift/issues/9) ran 0.4.0 against a
deterministic numeric scoring model — not a text agent — and every one of its six findings
reproduced. The common cause: the observation model assumed the agent emits prose, and the
tool stayed quiet about the places where that assumption made it blind.

- **Declared numeric channels** (`project.custom_scalars`). The eight built-in scalars measure
  properties of generated text; for a scorer or ranker, seven are constant by construction and
  the eighth is latency. A declared key is read from `output.structured` and enters the scalar
  battery on the same terms as a built-in — location, dispersion and tail primaries, the same
  materiality gates, the same BH pool, plus the PSI and Page–Hinkley diagnostics. Non-numeric
  and missing values become NaN, never a guess (booleans explicitly included: `isinstance(True,
  int)` holds in Python, and a flag entering as `1.0` would be a rate channel under a
  continuous channel's name). Capped at 16 channels, since each adds three primaries per family
  per baseline and an uncapped list would raise the family-wise alert rate with nobody deciding
  to. Built-in signature names are reserved: a collision would overwrite a shipped channel and
  silently retire it.
- **`project.deterministic`**. The `canary_repetitions >= 2` floor exists because one run cannot
  support a distributional comparison. That does not apply to an exactly reproducible agent,
  where the second repetition is a byte-identical copy. Declaring determinism lowers the floor
  to 1. It is a claim about the agent, and the docs say what it costs if the claim is wrong.
- **Per-channel materiality** (`[materiality.per_channel.<signature>]`). Gates were global, so
  raising `ks_distance` because latency is noisy on your hardware desensitised every scalar in
  the battery. The three scalar gates are now overridable per signature, range-validated
  exactly as the global values are.
- **`Store()` fails fast on an uninitialised directory.** It used to open successfully and die
  at the first write — after a full canary cycle of real work had been done and could not be
  saved. Now `FileNotFoundError` naming `dedrift init`.
- **`baseline set` warns when the golden baseline eats the rolling window.** Golden cycles are
  excluded from the rolling reference, so freezing everything starves the sudden channel. The
  only symptom was a persistent `NO REFERENCE`, which reads like "not enough history yet"
  rather than "you consumed the history you had".
- **Reports state family coverage.** `Families populated: N of 6`, and which are missing. An
  empty family is now reported rather than inferred: a project whose agent cannot refuse has no
  honest `refusal_boundary` canaries, and that is a different thing from having forgotten them.

Calibration note: every published false-alarm figure remains measured on the **default**
battery (m ≈ 336). Declared channels enlarge m and carry their own correlation structure; the
docs and the paper's limitations section now say so explicitly.

> **Correction in 0.5.1:** The anytime section below is retained as a
> historical record of what 0.4.0 claimed, not as current guidance. Its
> `twosample` e-value claim, dependent pooled products, repeated e-BH
> decision, null/wobble measurements, power numbers, and default/upgrade
> advice are retracted and superseded by the 0.5.1 entry above.

## 0.4.0

Independent-audit release: every defect the external audit confirmed is fixed, and both
inference paths were upgraded where the audit showed the mathematics was the bottleneck.
All behavioral numbers below are measured on the audit's harness and the shipped
calibration suite, not asserted.

Fixed, with regression tests (audit-measured defect in parentheses):

- Fixed Page-Hinkley NaN poisoning: one missing (family, cycle) reindexed the stream
  with NaN and the running mean never recovered — the family's streams could never alarm
  again (audit: 0/30 projects alarmed after a mid-history gap). Streams now run over the
  family's observed cycles.
- Fixed Page-Hinkley on near-constant discrete streams rendering statistics like 2.6e15:
  a scale floor at the window's floating-point resolution keeps constant streams at ~0.
- Fixed attribution nominating post-onset config events: the nearest-event search used
  absolute time, so an unrelated change AFTER the drift could be printed as "before
  onset". Events are now restricted to those preceding the estimated onset; when the
  (noisy) onset estimate predates every event, the nearest event before detection is
  shown labelled as weak evidence, and a drift with no preceding event is reported as
  silent. (`event_relation` on `Attribution`.)
- Fixed `ebh()` mapping `+inf` e-values to 0 — an overflowed e-value is the strongest
  possible evidence, now clamped to the largest finite double instead of zeroed.
- Fixed the degraded-data evasion: checks with >20% errored records suppressed ALL
  alerts, so an error storm masked any drift. The error-rate channel itself now alerts
  under degradation; other channels stay suppressed and the verdict remains DEGRADED DATA.
- Corrected the stale `checks.baseline_kind` schema comment ('dual' | 'anytime').

Validity-scale benchmark:

- Added `benchmark/`: a reproducible null-calibration study measuring false-alarm
  rates of four drift-detection configurations (folk-threshold PSI, the same index
  under the validity guard, naive uncontrolled two-sample KS, and dedrift's
  fixed-sample and anytime-valid paths) on 500 seeded stable-agent histories at
  two canary scales. `make benchmark` regenerates the results JSONs, the
  `web/benchmark/` page table, and the paper's validity-scale table.
  The study measures generic statistical practice rather than named products:
  what a given tool does at its own defaults, on a table it was not built for, is
  a fact about a configuration we chose for it, and the failure modes shown here
  are properties of the methods themselves.

New detection capability:

- Added `tool_order_inversions` (Kendall-tau inversions of the tool-call name sequence)
  to the Tier-1 scalar battery. The audit measured tool-call ORDER drift as completely
  invisible (0/30 by construction — counts and schema were never the question); the new
  channel catches full workflow reversals in 8/20 first-check projects at canary scale.
  (These notes originally carried the earlier 10-replicate run of the same probe; it was
  rerun at 20 replicates for the full adversarial matrix, and the supersession is
  recorded on the statistics page.)
- Refusal alerts now carry an inline semantics note: the signature is pattern-matched
  phrasing, and a model that paraphrases its refusals can move the measured rate DOWN
  while true refusals rise (audit-confirmed wrong-direction alerts).
- Reports state plainly when no embedder is pinned that meaning-level drift is
  unmeasurable in the project (the audit measured the no-embedder arm at the null rate).

Cycle-effect-robust fixed path (`detection.cycle_effect = "auto"`, opt-in; the
default `"off"` keeps the exact v0.3.1 record-level battery):

- Per-channel intraclass-correlation estimates from the frozen golden window engage a
  cluster-aware composite when a latent per-cycle offset is present: within-cycle-
  standardized KS (shape; offsets cancel exactly) disjoined with design-effect Welch
  (location), design-effect-inflated rate z, and Student-t cycle-level summaries for
  dispersion and P95. On the audit's sigma ladder the per-check false-alert rate under
  within-version wobble drops from {36.5%, 71.0%, 89.0%, 98.5%} at sigma = {0.05, 0.10,
  0.15, 0.25} to {34.0%, 64.0%, 81.3%, 94.0%} with `auto` — and to {6.0%, 31.0%,
  47.0%, 66.0%} with `auto` + `alert_persistence = 2` — while the exchangeable null
  stays calibrated (auto at sigma = 0: 3.8% at CI scale, 4.5% [1.4, 7.6] on the
  harness) and gross-swap detection remains 10/10. The correction reduces but does not
  restore calibration at canary scale; the residual is information-theoretic.
  See docs/statistics.md#cycle-effects.

Anytime-valid mode power upgrade (`anytime.rate_model = "twosample"`, default;
`"frozen_cp"` restores the v0.3.1 construction):

- Rate processes now use the two-sample SAFE e-value (beta-binomial posterior-predictive
  ratio against the frozen reference) instead of the Clopper-Pearson worst-case tilt
  mixture, whose interval contained the alternative at canary scale: the audit measured
  0/30 detections at +5/+10/+20pp over 60 cycles, with wealth DECAYING on real drift.
  The two-sample process, with pooled per-signature processes (the product of the six
  per-family e-values — itself an e-value), detects a family-wide +20pp refusal shift in
  100/100 runs (median 17 cycles) and +10pp in 89/100 runs (median 50) within 400 cycles
  at documentation scale, while ever-alert
  rates on stable agents stay within the alpha = 0.05 lifetime budget at 2000-cycle
  horizons, including iid cycle wobble up to sigma = 0.25. The persistent-confounding
  regime (sigma = 0.25, phi = 0.9) exceeds budget at 7.2% (36/500, Wilson upper 9.8%) —
  published as the measured boundary of the guarantee. No coverage budget is spent, so
  the e-BH level is the full per-epoch alpha.
- The alternative was also implemented reference-anchored and REJECTED: it never stalls
  but is not an e-value when the reference posterior misses the true rate (measured
  E[E] up to 5.8 under the null). Documented in evalues/twosample.py.

### Upgrade notes

- Default behavior changes: `anytime.rate_model = "twosample"` is the new default;
  set `"frozen_cp"` to reproduce v0.3.1 anytime outputs. `detection.cycle_effect`
  stays `"off"` by default (the v0.3.1 record-level battery); `"auto"` and
  `alert_persistence = 2` are opt-in, recommended for wobble-prone hosted models
  with the measured trade-off in docs/statistics.md#cycle-effects.
- The scalar battery grows by one signature (`tool_order_inversions`), so the BH
  primary pool grows accordingly; measured pipeline null rate remains within the
  published band (Phase-E numbers in docs/statistics.md).
- `anytime.gamma_total` and `anytime.tilts` are meaningful only with
  `rate_model = "frozen_cp"`.

The package remains pre-alpha. Measured bands are documented next to every guarantee;
assumption-plus-measurement areas (stopped-e-BH causal condition, cluster-robust
approximations) say so in place.

## 0.3.1

Production-hardening release for the reviewer findings.

- Added strict, timezone-aware interaction validation and fail-closed adapter coercion.
- Added durable open/finalized canary-cycle lifecycle. Checks and baselines consume only
  finalized, immutable cycles; check rows persist their record-ID/log-offset snapshot.
- Reconciled complete crash-tail JSONL records when a store opens and cached canonical
  payloads in the rebuildable SQLite index for non-blocking inference snapshots.
- Made append retries content-idempotent and rejected conflicting IDs or extensions to
  finalized/anytime-processed cycles.
- Made fixed and anytime coverage/inconclusive states explicit and gave the CLI stable
  exit codes: 0 healthy, 2 drift, 3 inconclusive, 1 operational/configuration error.
- Made configuration parsing strict, raised permutation resolution when necessary,
  counted undefined declared hypotheses conservatively in BH, and made uncalibrated
  MMD diagnostic-only.
- Added wheel smoke tests and release tag/version verification.

### Upgrade notes

- Existing histories without cycle lifecycle metadata are migrated as **open**, because
  completion cannot be inferred from durable records alone. Finalize each known-complete
  cycle with `dedrift cycle finalize CYCLE_ID --expected-records N` before checking.
- `dedrift log` now leaves imported canary cycles open unless `--finalize-cycles` is set.
- Strict schema validation rejects unknown fields, negative counters, naive timestamps,
  incomplete canary coordinates, and mismatched serialized config fingerprints.
- `materiality.scalar_cohen_d` remains accepted for config compatibility but is not an
  alert gate in 0.3.1; Welch is corroboration-only.

The package remains pre-alpha. Fixed-mode BH assumptions, observed-effect filtering,
cycle-level provider noise, and anytime/fixed practical-hypothesis differences are
documented limitations, not production guarantees.
