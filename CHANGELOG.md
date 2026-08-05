# Changelog

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
