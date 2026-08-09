# Changelog

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
  rates of six drift-detection configurations (folk-threshold PSI, the same index
  under the validity guard, Evidently 0.7.21 DataDriftPreset at defaults over the
  pooled signature table, naive uncontrolled two-sample KS, and dedrift's
  fixed-sample and anytime-valid paths) on 500 seeded stable-agent histories at
  two canary scales. `make benchmark` regenerates the results JSONs, the
  `web/benchmark/` page table, and the paper's validity-scale table.
  A per-canary-family Evidently arm is measured and kept in the results JSON but
  deliberately not tabled: splitting one report into six and taking any-of is a
  user's choice rather than a documented default, so its rate would partly be our
  own multiplicity rather than the tool's calibration.
  `benchmark/METHODS_CONSIDERED.md` records the tools attempted and excluded with
  checkable reasons; the page and paper carry a standing right-of-reply invitation
  to measured tools' maintainers (optional notification drafts are held locally in
  `benchmark/OUTREACH.md`, git-ignored).

New detection capability:

- Added `tool_order_inversions` (Kendall-tau inversions of the tool-call name sequence)
  to the Tier-1 scalar battery. The audit measured tool-call ORDER drift as completely
  invisible (0/30 by construction — counts and schema were never the question); the new
  channel catches full workflow reversals in 5/10 first-check projects at canary scale.
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
  within-version wobble drops from {36.8%, 70.6%, 88.3%, 97.6%} at sigma = {0.05, 0.10,
  0.15, 0.25} to {34.4%, 63.7%, 80.6%, 92.9%} with `auto` — and to {7.6%, 31.7%,
  47.1%, 65.4%} with `auto` + `alert_persistence = 2` — while the exchangeable null
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
