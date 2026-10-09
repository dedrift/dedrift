"""The anytime-valid check: stateful counterpart to :mod:`dedrift.check`.

Same logs, same canary design, same channels — a different inference layer.
The fixed-sample path answers "is this cycle different?" and controls error
*per check*; this one answers "has anything changed since the epoch began?"
and controls error over the whole trajectory. Both run on identical inputs
so the comparison is reproducible from the CLI rather than from a study
script.

The budget, and the part that is easy to get wrong
--------------------------------------------------
``alpha = alpha_prime + gamma_total``, where ``alpha_prime`` is divided
across the frozen process pool and ``gamma_total`` buys nuisance-parameter
coverage for the *whole battery*. A process whose interval misses the truth
is not an e-process, so coverage failures also union-bound across processes:

    P(ever a false alert) <= alpha_prime + sum_i gamma_i

Hence ``gamma_i = gamma_total / K``, computed from the live process count
rather than typed in. At 24 processes, using ``gamma_i = gamma_total``
directly would state 0.05 while delivering 0.28.

Why only the golden baseline
----------------------------
The bound above needs each process's nuisance interval to be a *single
fixed event*, settled once and then held. A frozen golden baseline gives
exactly that: the reference counts never change, so one Clopper-Pearson
interval at level ``gamma_i`` covers or does not, once, for the whole
epoch.

A *rolling* reference is recomputed every cycle, so its interval is a new
event each cycle and the union bound over a horizon of ``T`` cycles is
``T * gamma_i``, not ``gamma_i``. At the shipped default and ``T = 2000``
that is not a guarantee at all. Honouring it needs a time-uniform interval
-- a confidence sequence -- which this package does not yet have. Until it
does, rolling processes are **excluded from the anytime pool** and
:func:`dedrift.evalues.rates.rate_evalue` refuses to build a bet on a
non-frozen reference rather than labelling one as if it were covered.
Rolling comparisons remain fully supported on the fixed-sample path, which
is where they were always adjudicated.

What is and is not established
------------------------------
* Per process, Ville's inequality bounds the probability of *ever* crossing.
* A Bonferroni split across the fixed pool converts those bounds to a
  trajectory-wide family-wise bound without any cross-process independence
  assumption.
* A crossing is absorbing for the epoch. Checks that fold a backlog therefore
  cannot erase an alert merely because wealth fell again before the last
  unseen cycle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dedrift.canary import DEDRIFT_METADATA_KEY, SUITE_FINGERPRINT_KEY
from dedrift.check import DEGRADED_ERROR_FRACTION, get_golden_baseline
from dedrift.config import ProjectConfig
from dedrift.embeddings import get_pinned_embedder
from dedrift.evalues import (
    EProcessState,
    EProcessUpdate,
    EValueOutcome,
    PriorState,
    epoch_fingerprint,
    geometric_allocation,
    rate_evalue,
    symmetric_grid,
    update_process,
    ville_threshold,
)
from dedrift.evalues.process import state_to_row
from dedrift.evalues.rates import per_process_gamma
from dedrift.schema import InteractionRecord
from dedrift.signatures import signatures_frame
from dedrift.signatures.structural import RATE_SIGNATURES
from dedrift.store import FinalizedSnapshotChangedError, Store

#: Bumped when signature extraction changes meaning; part of the epoch
#: fingerprint, because evidence gathered with a different instrument is
#: not evidence about the same null.
EXTRACTOR_VERSION = "1"

#: Bumped when the anytime decision rule changes. Version 2 withdrew the
#: invalid prior-predictive model, removed dependent pooled products, and
#: replaced repeated e-BH with persistent Ville--Bonferroni crossings.
ANYTIME_INFERENCE_VERSION = "3"

#: The only baseline the anytime path admits. A rolling reference lacks a
#: time-uniform interval (module docstring); until a confidence sequence
#: exists, betting against it would state a guarantee we cannot deliver.
ANYTIME_BASELINE = "golden"

#: Verdict when the epoch has no frozen baseline to bet against. Distinct
#: from ``OK`` on purpose: an empty pool means nothing was monitored, and
#: an operator reading ``OK`` would conclude the opposite.
NO_BASELINE_VERDICT = "NO GOLDEN BASELINE"


@dataclass(frozen=True)
class ProcessReport:
    """One e-process as the report renders it."""

    key: tuple[str, str, str, str]
    log_wealth: float
    evidence: float
    epoch: int
    cycles: int
    bets_placed: int
    rise_cycle: int | None
    crossed_at: int | None
    rejected: bool
    detail: str

    @property
    def label(self) -> str:
        """Report-friendly identity, e.g. ``[golden] edge_case/refusal (rate)``."""
        b, f, s, c = self.key
        return f"[{b}] {f}/{s} ({c})"

    @property
    def evalue_capped(self) -> float:
        """``E`` on the natural scale, capped so reports never print ``inf``.

        Wealth overflows float64 well before it stops being interesting, so
        the decision rule runs in logs; this exists only for display.
        """
        return float(np.exp(min(self.log_wealth, 700.0)))


@dataclass(frozen=True)
class AnytimeCheckResult:
    """Outcome of one anytime-valid check.

    Attributes:
        ts: Check timestamp (UTC, ISO).
        current_cycle: Latest finalized cycle in this check's snapshot. It can
            be ineligible when it was first opened before the epoch declaration;
            ``coverage_status`` then reports ``NO OBSERVATION``.
        fingerprint: Epoch fingerprint in force.
        alpha: Lifetime battery-wide budget.
        alpha_prime: Family-wise Ville crossing budget.
        gamma_total: Total coverage budget.
        gamma_per_process: ``gamma_total / K``, the level each interval uses.
        n_processes: Processes in the epoch's declared pool.
        pool_declared_now: True when this check declared the pool, i.e. it is
            the epoch's first. Worth surfacing: membership is frozen here, so
            a signature with no reference data at this moment waits for the
            next epoch to join.
        verdict: ``DRIFT DETECTED`` / ``OK`` / ``DEGRADED DATA``.
        degraded: Whether any monitored cycle folded into this epoch exceeded
            the collection-error threshold. Rate evidence is still folded;
            this persistent warning is not an outcome-dependent skip rule.
        degraded_cycles: Epoch-eligible cycles responsible for the warning.
        coverage_status: ``FULL``, ``PARTIAL``, ``NONE``, ``NO OBSERVATION``,
            or ``NO REFERENCE`` for the current cycle's declared rate-process
            pool.
        suppressed_families: Families not comparable to the frozen baseline.
        processes: Per-process report rows.
        resets: Human-readable epoch-reset notices.
        processed_cycles: Post-golden cycles durably folded by this invocation.
            Empty means this was an idempotent read of already-processed state.
        n_alerts: Convenience count.
    """

    ts: str
    current_cycle: str
    fingerprint: str
    alpha: float
    alpha_prime: float
    gamma_total: float
    gamma_per_process: float
    n_processes: int
    pool_declared_now: bool
    verdict: str
    degraded: bool
    degraded_cycles: tuple[str, ...]
    coverage_status: str
    suppressed_families: tuple[str, ...]
    processes: list[ProcessReport] = field(default_factory=list)
    resets: list[str] = field(default_factory=list)
    processed_cycles: tuple[str, ...] = field(default_factory=tuple)
    snapshot_log_offset: int = 0
    snapshot_finalized_revision: int = 0
    snapshot_record_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_alerts(self) -> int:
        """Processes with a persistent Ville--Bonferroni crossing."""
        return sum(1 for p in self.processes if p.rejected)

    def alerts(self) -> list[ProcessReport]:
        """Rejected processes, deterministically ordered."""
        return sorted((p for p in self.processes if p.rejected), key=lambda p: p.key)


def load_states(store: Store) -> dict[tuple[str, str, str, str], EProcessState]:
    """Read persisted e-process state.

    Columns are named explicitly rather than relying on ``SELECT *`` and a
    row factory: the schema will grow, and positional unpacking would break
    silently at exactly the moment state stops meaning what it says.
    """
    conn = store.connect()
    out: dict[tuple[str, str, str, str], EProcessState] = {}
    query = (
        "SELECT baseline, family, signature, channel, log_wealth, cycles, "
        "bets_placed, epoch, fingerprint, peak_log_wealth, rise_cycle, "
        "crossed_at, prior_successes, prior_trials, reference_successes, "
        "reference_trials FROM eprocess_state"
    )
    for row in conn.execute(query):
        (
            baseline,
            family,
            signature,
            channel,
            log_wealth,
            cycles,
            bets_placed,
            epoch,
            fingerprint,
            peak,
            rise_cycle,
            crossed_at,
            prior_s,
            prior_t,
            ref_s,
            ref_t,
        ) = row
        key = (baseline, family, signature, channel)
        out[key] = EProcessState(
            key=key,
            log_wealth=log_wealth,
            cycles=cycles,
            bets_placed=bets_placed,
            epoch=epoch,
            fingerprint=fingerprint,
            prior=PriorState(
                # Only admissible observations train the predictable prior.
                # ``cycles`` also counts outcome-independently unavailable
                # observations; ``bets_placed`` does not.
                n_cycles=bets_placed,
                successes=prior_s,
                trials=prior_t,
                reference_successes=ref_s,
                reference_trials=ref_t,
            ),
            peak_log_wealth=peak,
            rise_cycle=rise_cycle,
            crossed_at=crossed_at,
        )
    return out


def save_states(store: Store, states: list[EProcessState], ts: str, *, commit: bool = True) -> None:
    """Upsert e-process state, optionally joining the caller's transaction."""
    conn = store.connect()
    for st in states:
        row = state_to_row(st)
        row["updated_ts"] = ts
        cols = ", ".join(row)
        marks = ", ".join(["?"] * len(row))
        conn.execute(
            f"INSERT OR REPLACE INTO eprocess_state ({cols}) VALUES ({marks})",
            tuple(row.values()),
        )
    if commit:
        conn.commit()


Key = tuple[str, str, str, str]


def load_processed_cycles(store: Store, fingerprint: str) -> set[str]:
    """Return cycle IDs already committed for one epoch fingerprint."""
    rows = store.connect().execute(
        "SELECT cycle_id FROM anytime_processed_cycles WHERE fingerprint = ?",
        (fingerprint,),
    )
    return {str(row[0]) for row in rows}


def _load_epoch_degraded_cycles(store: Store, fingerprint: str) -> tuple[str, ...]:
    """Recover every degraded cycle previously folded into one epoch.

    Degraded evidence is still part of process wealth, so its warning must be
    as persistent as that wealth. Reading all matching check rows also repairs
    the presentation state of early prerelease rows that stored only the
    degraded cycles seen by one invocation.
    """
    degraded: dict[str, None] = {}
    rows = store.connect().execute(
        "SELECT params_json FROM checks WHERE baseline_kind = 'anytime' ORDER BY id"
    )
    for row in rows:
        try:
            params = json.loads(str(row[0]))
        except (TypeError, ValueError):
            continue
        if not isinstance(params, dict) or params.get("fingerprint") != fingerprint:
            continue
        cycles = params.get("degraded_cycles", [])
        if not isinstance(cycles, list) or not all(isinstance(cycle, str) for cycle in cycles):
            continue
        for cycle in cycles:
            degraded.setdefault(cycle, None)
    return tuple(degraded)


def _declare_epoch(
    store: Store,
    base_fingerprint: str,
    states: dict[Key, EProcessState],
    ts: str,
    *,
    initial_start_after_cycle: str,
    changed_start_after_cycle: str,
    initial_start_after_open_seq: int,
    changed_start_after_open_seq: int,
) -> tuple[str, int, str, int, bool]:
    """Return a durable, globally monotone epoch index.

    Per-process rows cannot provide this invariant: a suite change may
    replace every key.  Geometric allocation therefore uses this registry,
    so a new instrument can never accidentally restart the lifetime budget
    at epoch zero.
    """
    conn = store.connect()
    latest = conn.execute(
        "SELECT fingerprint, base_fingerprint, epoch_index, start_after_cycle, "
        "start_after_open_seq "
        "FROM anytime_epochs ORDER BY epoch_index DESC LIMIT 1"
    ).fetchone()
    if latest is not None and str(latest[1]) == base_fingerprint:
        return str(latest[0]), int(latest[2]), str(latest[3]), int(latest[4]), False

    registered = int(latest[2]) if latest is not None else -1
    state_max = max((state.epoch for state in states.values()), default=-1)
    epoch_index = max(registered, state_max) + 1
    start_after_cycle = (
        initial_start_after_cycle if latest is None and not states else changed_start_after_cycle
    )
    start_after_open_seq = (
        initial_start_after_open_seq
        if latest is None and not states
        else changed_start_after_open_seq
    )
    fingerprint = f"{base_fingerprint}:e{epoch_index}"
    conn.execute(
        "INSERT INTO anytime_epochs "
        "(fingerprint, base_fingerprint, epoch_index, start_after_cycle, "
        "start_after_open_seq, declared_ts) VALUES (?, ?, ?, ?, ?, ?)",
        (
            fingerprint,
            base_fingerprint,
            epoch_index,
            start_after_cycle,
            start_after_open_seq,
            ts,
        ),
    )
    return fingerprint, epoch_index, start_after_cycle, start_after_open_seq, True


def _canonical_suite_identity(records: list[InteractionRecord], cycle_id: str) -> str:
    """Return the effective suite identity observed in ``cycle_id``.

    New records carry a full-suite fingerprint in reserved metadata. That is
    authoritative because it covers canaries that may be absent from a
    partially collected cycle. The canonical-content fallback keeps legacy
    projects safer than the old canary-count fingerprint: IDs, families,
    inputs, and any record-local expectation/rubric metadata all participate.
    """
    cycle_records = [r for r in records if r.cycle_id == cycle_id]
    declared: set[str] = set()
    declared_records = 0
    for record in cycle_records:
        metadata = record.input.metadata
        reserved = metadata.get(DEDRIFT_METADATA_KEY)
        if not isinstance(reserved, dict):
            continue
        value = reserved.get(SUITE_FINGERPRINT_KEY)
        if isinstance(value, str) and value:
            declared.add(value)
            declared_records += 1
    if len(declared) > 1:
        msg = f"cycle {cycle_id!r} contains conflicting suite fingerprints: {sorted(declared)}"
        raise ValueError(msg)
    if declared:
        if declared_records != len(cycle_records):
            msg = (
                f"cycle {cycle_id!r} mixes records with and without a declared "
                "suite fingerprint; refusing a partially identified instrument"
            )
            raise ValueError(msg)
        return next(iter(declared))

    definitions: dict[str, str] = {}
    for record in cycle_records:
        if not record.canary_id:
            continue
        payload = {
            "id": record.canary_id,
            "input": record.input.model_dump(mode="json"),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        previous = definitions.setdefault(record.canary_id, canonical)
        if previous != canonical:
            msg = f"cycle {cycle_id!r} has inconsistent definitions for canary {record.canary_id!r}"
            raise ValueError(msg)
    canonical_suite = json.dumps(
        [json.loads(definitions[key]) for key in sorted(definitions)],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"sha256:{hashlib.sha256(canonical_suite.encode()).hexdigest()}"


def _suppressed_families(
    ref_frame: pd.DataFrame, cur_frame: pd.DataFrame, families: list[str]
) -> set[str]:
    """Families whose current composition is not comparable to golden."""
    reference_families = {str(family) for family in ref_frame["family"].unique()}
    current_families = {str(family) for family in cur_frame["family"].unique()}
    # Detect both missing and unexpected families. The caller's ``families``
    # comes from the frozen reference so an excluded historical suite cannot
    # poison this comparison.
    suppressed: set[str] = reference_families ^ current_families
    for family in families:
        ref = ref_frame[ref_frame["family"] == family]
        cur = cur_frame[cur_frame["family"] == family]
        if ref.empty or cur.empty:
            suppressed.add(family)
            continue
        ref_counts = ref["canary_id"].value_counts()
        cur_counts = cur["canary_id"].value_counts()
        membership_changed = set(ref_counts.index) != set(cur_counts.index)
        unbalanced = bool(
            (len(ref_counts) > 0 and ref_counts.min() != ref_counts.max())
            or (len(cur_counts) > 0 and cur_counts.min() != cur_counts.max())
        )
        if membership_changed or unbalanced:
            suppressed.add(family)
    return suppressed


def _neutral_update(
    state: EProcessState,
    *,
    fingerprint: str,
    alpha_prime: float,
    reference: tuple[int, int],
    detail: str,
) -> EProcessUpdate:
    """Fold ``E=1`` without learning from an invalid current observation."""
    starting_prior = (
        PriorState() if state.fingerprint and state.fingerprint != fingerprint else state.prior
    )
    outcome = EValueOutcome(log_e=0.0, placed=False, detail=detail)
    update = update_process(
        state,
        outcome,
        fingerprint=fingerprint,
        alpha_prime=alpha_prime,
        successes=0,
        trials=0,
        reference=reference,
    )
    prior = replace(
        starting_prior,
        reference_successes=reference[0],
        reference_trials=reference[1],
    )
    return replace(update, state=replace(update.state, prior=prior))


def load_pool(store: Store, fingerprint: str) -> list[Key]:
    """Return the pool declared for this epoch, or empty if undeclared."""
    conn = store.connect()
    rows = conn.execute(
        "SELECT baseline, family, signature, channel FROM epoch_pool "
        "WHERE fingerprint = ? ORDER BY baseline, family, signature, channel",
        (fingerprint,),
    )
    return [(b, f, s, c) for b, f, s, c in rows]


def declare_pool(
    store: Store,
    fingerprint: str,
    keys: list[Key],
    ts: str,
    *,
    commit: bool = True,
) -> None:
    """Freeze the pool, optionally joining the caller's transaction."""
    conn = store.connect()
    conn.executemany(
        "INSERT OR REPLACE INTO epoch_pool "
        "(fingerprint, baseline, family, signature, channel, declared_ts) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(fingerprint, *k, ts) for k in keys],
    )
    if commit:
        conn.commit()


def live_keys(
    frame: pd.DataFrame,
    families: list[str],
    baselines: list[tuple[str, list[str]]],
) -> list[Key]:
    """Combinations with usable REFERENCE data — the admissible pool.

    A signature with no reference observations (an ``exact_match`` column
    that is entirely absent because the suite declares no expected answers,
    say) cannot ever produce evidence, yet including it shrinks every other
    process's coverage and crossing budgets. Excluding it is
    pure gain.

    The test uses reference data only. Current-cycle availability is *not*
    consulted: that would make pool membership depend on the very cycle
    being bet on, which is the predictability violation this package is
    built to prevent. The consequence is stated rather than hidden — a
    signature that gains data mid-epoch waits until the next epoch to join,
    and one with outcome-independently unavailable data stays in the pool
    contributing ``E_t = 1``. Composition mismatches fail closed.
    """
    keys: list[Key] = []
    for baseline_name, ref_cycles in baselines:
        if not ref_cycles:
            continue
        ref_frame = frame[frame["cycle_id"].isin(ref_cycles)]
        for family in families:
            fam = ref_frame[ref_frame["family"] == family]
            for sig in RATE_SIGNATURES:
                if len(fam[sig].dropna()) > 0:
                    keys.append((baseline_name, family, sig, "rate"))
    return sorted(keys)


def run_anytime_check(store: Store, config: ProjectConfig | None = None) -> AnytimeCheckResult:
    """Atomically fold every unseen post-golden cycle exactly once.

    Finalized record IDs are captured in a brief WAL snapshot. Signature
    extraction then runs without blocking ingestion; only state, ledger,
    pool declaration, and the check row share the final writer transaction.
    """
    cfg = config or ProjectConfig.load(store.project_dir)
    for attempt in range(3):
        records, snapshot_offset, snapshot_revision = (
            store.read_finalized_canary_snapshot_with_revision()
        )
        try:
            return _run_anytime_check_snapshot(
                store,
                cfg,
                records,
                snapshot_offset,
                snapshot_revision,
            )
        except FinalizedSnapshotChangedError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def _run_anytime_check_snapshot(
    store: Store,
    cfg: ProjectConfig,
    records: list[InteractionRecord],
    snapshot_offset: int,
    snapshot_revision: int,
) -> AnytimeCheckResult:
    """Declare or advance an epoch, folding eligible unseen cycles once.

    A brand-new epoch starts *after* the current finalized cycle.  Its first
    check declares the pool and betting rule but cannot replay already
    observed outcomes: choosing either after seeing those outcomes would
    violate predictability.  Later checks may fold a backlog because that
    epoch and pool were durably declared before every cycle in the backlog.

    Only the rate channel is implemented: a verified single channel is worth
    more than four unverified ones, and the scalar and semantic channels
    need constructions of their own rather than a reuse of this one.

    Args:
        store: Project store containing finalized canary cycles.
        cfg: Validated project configuration.
        records: Immutable finalized-record snapshot.
        snapshot_offset: Committed JSONL offset recorded with the check.
        snapshot_revision: Finalized-cycle watermark captured with the records.

    Returns:
        The check result. State is persisted before returning.

    Raises:
        ValueError: If no canary cycles exist.
    """
    ac = cfg.anytime
    if not records:
        msg = "no finalized canary cycles; finish and finalize a canary cycle first"
        raise ValueError(msg)

    frame = signatures_frame(records)
    cycles = [str(cycle) for cycle in dict.fromkeys(frame["cycle_id"])]
    current = cycles[-1]
    # Declared baseline cycles are reference data, even when the latest log
    # currently ends on one of them. They must never be folded as evidence.
    declared_golden = get_golden_baseline(store)
    golden = [c for c in declared_golden if c in cycles]

    records_by_cycle: dict[str, list[InteractionRecord]] = {}
    for record in records:
        assert record.cycle_id is not None
        records_by_cycle.setdefault(record.cycle_id, []).append(record)
    golden_suites = {_canonical_suite_identity(records_by_cycle[cycle], cycle) for cycle in golden}
    if len(golden_suites) > 1:
        msg = "golden baseline spans multiple canary-suite fingerprints; re-baseline"
        raise ValueError(msg)
    golden_suite = next(iter(golden_suites), None)
    # The frozen reference defines the epoch's instrument. The latest
    # finalized cycle may have been opened before declaration and therefore
    # be permanently excluded; letting that cycle define (or veto) the epoch
    # would make finalization order change the monitored hypothesis.
    epoch_suite = golden_suite or "no-golden-baseline"

    inference_config = json.dumps(
        {
            "alpha": ac.alpha,
            "gamma_total": ac.gamma_total,
            "tilts": ac.tilts,
            "epoch_allocation": ac.epoch_allocation,
            "rate_model": ac.rate_model,
            "inference_version": ANYTIME_INFERENCE_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    pinned_embedder = get_pinned_embedder(store)
    if cfg.embedder and cfg.embedder != pinned_embedder:
        raise ValueError(
            f"embeddings.model expects {cfg.embedder!r}, but the authoritative pin is "
            f"{pinned_embedder!r}; use `dedrift embedder pin` or restore project state"
        )
    base_fingerprint = epoch_fingerprint(
        suite_version=epoch_suite,
        embedder=pinned_embedder or "",
        golden_cycles=tuple(golden),
        extractor_version=EXTRACTOR_VERSION,
        inference_config=inference_config,
    )

    cur_frame = frame[frame["cycle_id"] == current]
    current_degraded = bool(cur_frame["had_error"].mean() > DEGRADED_ERROR_FRACTION)

    # The pool is declared ONCE per epoch and then frozen. Its size sets the
    # per-process coverage budget, which sets the nuisance interval, which is
    # part of the bet — and for a frozen baseline the guarantee needs that
    # interval to be a single fixed event settled at epoch start. A pool
    # recomputed per cycle would quietly turn it into a sequence of
    # different events.
    ts = datetime.now(timezone.utc).isoformat()
    # Golden only: a rolling reference has no time-uniform interval, so its
    # coverage event is fresh every cycle and the budget arithmetic above
    # does not hold for it. See the module docstring.
    baselines = [(ANYTIME_BASELINE, golden)]
    ref_by_baseline = dict(baselines)
    ref_frame_all = frame[frame["cycle_id"].isin(golden)]
    # Pool membership belongs to the frozen instrument. Excluded historical
    # cycles must not inject families into composition checks or multiplicity.
    families = sorted(str(family) for family in ref_frame_all["family"].unique())
    conn = store.connect()
    if conn.in_transaction:
        raise RuntimeError("run_anytime_check requires ownership of the SQLite transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        # A complete JSONL tail can survive a crash before its SQLite commit.
        # Reconcile and register those observations before fixing the epoch
        # boundary, including cycles that are still open and absent from the
        # finalized snapshot used for the statistical computation.
        store._reconcile_jsonl_tail(conn)
        store._sync_cycle_registry(conn, legacy_finalize=False)
        store.assert_finalized_revision(snapshot_revision)
        if get_golden_baseline(store) != declared_golden:
            raise FinalizedSnapshotChangedError(
                "golden baseline changed during the check snapshot; retry"
            )
        if get_pinned_embedder(store) != pinned_embedder:
            raise FinalizedSnapshotChangedError(
                "pinned embedder changed during the check snapshot; retry"
            )
        states = load_states(store)
        missing_open_sequences = int(
            conn.execute("SELECT COUNT(*) FROM canary_cycles WHERE open_seq IS NULL").fetchone()[0]
        )
        if missing_open_sequences:
            raise RuntimeError("canary cycle observation order is incomplete; reopen the store")
        declaration_open_seq = int(
            conn.execute("SELECT COALESCE(MAX(open_seq), 0) FROM canary_cycles").fetchone()[0]
        )
        (
            fingerprint,
            epoch_now,
            epoch_start_after,
            epoch_start_after_open_seq,
            epoch_declared_now,
        ) = _declare_epoch(
            store,
            base_fingerprint,
            states,
            ts,
            initial_start_after_cycle=current,
            changed_start_after_cycle=current,
            initial_start_after_open_seq=declaration_open_seq,
            changed_start_after_open_seq=declaration_open_seq,
        )
        # E-process time is first-observation order, never finalization order.
        # A later cycle may finalize while an earlier one is still open, but
        # folding it would let an outcome-visible completion schedule permute
        # the wealth path (and manufacture an absorbing crossing). Process
        # only the contiguous finalized prefix after the declaration boundary;
        # defer everything behind the first open cycle.
        post_golden: list[str] = []
        lifecycle = (
            conn.execute(
                "SELECT cycle_id, status, open_seq, prospective FROM canary_cycles "
                "WHERE open_seq > ? ORDER BY open_seq",
                (epoch_start_after_open_seq,),
            ).fetchall()
            if golden
            else []
        )
        snapshot_cycles = set(cycles)
        for cycle_id, status, _open_seq, prospective in lifecycle:
            if not bool(prospective):
                raise ValueError(
                    f"cycle {str(cycle_id)!r} was imported after the anytime epoch began "
                    "without a durable pre-observation begin_cycle marker. Refusing to "
                    "treat post-hoc outcomes as prospective evidence. Re-baseline after "
                    "the imported history, then collect future cycles with CanaryRunner "
                    "or call Store.begin_cycle before the first agent call."
                )
            if str(status) != "finalized":
                break
            cycle = str(cycle_id)
            if cycle not in snapshot_cycles:
                raise FinalizedSnapshotChangedError(
                    f"finalized cycle {cycle!r} is absent from the check snapshot; retry"
                )
            post_golden.append(cycle)
        eligible_suite_by_cycle = {
            cycle: _canonical_suite_identity(records_by_cycle[cycle], cycle)
            for cycle in post_golden
        }
        mismatched_suites = [
            cycle
            for cycle in post_golden
            if golden_suite is not None and eligible_suite_by_cycle[cycle] != golden_suite
        ]
        if mismatched_suites:
            msg = (
                "unprocessed post-golden backlog spans a canary-suite boundary; "
                "these eligible cycles do not match the golden baseline: "
                f"{', '.join(mismatched_suites)}. Refusing to combine instruments. "
                "Check before changing the suite, or re-baseline on prospectively "
                "selected cycles from the new suite."
            )
            raise ValueError(msg)
        current_is_eligible = current in post_golden
        pool_keys = load_pool(store, fingerprint)
        pool_declared_now = epoch_declared_now
        if not pool_keys:
            pool_keys = live_keys(frame, families, baselines)
            if epoch_declared_now:
                declare_pool(store, fingerprint, pool_keys, ts, commit=False)

        grid = symmetric_grid(ac.tilts)
        processed = load_processed_cycles(store, fingerprint)
        unseen = [cycle for cycle in post_golden if cycle not in processed]
        newly_degraded = tuple(
            cycle
            for cycle in unseen
            if bool(frame[frame["cycle_id"] == cycle]["had_error"].mean() > DEGRADED_ERROR_FRACTION)
        )
        persisted_degraded = _load_epoch_degraded_cycles(store, fingerprint)
        degraded_cycles = tuple(dict.fromkeys((*persisted_degraded, *newly_degraded)))
        if current_is_eligible and current_degraded and current not in degraded_cycles:
            degraded_cycles = (*degraded_cycles, current)
        degraded = bool(degraded_cycles)

        if ac.epoch_allocation == "geometric":
            effective_alpha = geometric_allocation(ac.alpha, epoch_now)
            effective_gamma_total = geometric_allocation(ac.gamma_total, epoch_now)
            effective_alpha_prime = effective_alpha - effective_gamma_total
        elif ac.epoch_allocation == "per_epoch":
            effective_alpha = ac.alpha
            effective_gamma_total = ac.gamma_total
            effective_alpha_prime = ac.alpha_prime
        else:
            msg = (
                f"unknown anytime.epoch_allocation {ac.epoch_allocation!r}; "
                "expected 'per_epoch' or 'geometric'"
            )
            raise ValueError(msg)
        gamma_i = per_process_gamma(effective_gamma_total, max(len(pool_keys), 1))
        # Each state gets an equal share of the trajectory-wide crossing
        # budget. Ville per process + a union bound over this frozen pool is
        # valid under arbitrary dependence and at arbitrary check cadence.
        decision_alpha = effective_alpha_prime / max(len(pool_keys), 1)

        working: dict[Key, EProcessState] = {}
        for key in pool_keys:
            state = states.get(key)
            if state is None:
                working[key] = EProcessState(key=key, epoch=epoch_now)
            elif state.fingerprint and state.fingerprint != fingerprint:
                working[key] = replace(state, epoch=epoch_now - 1)
            else:
                working[key] = replace(state, epoch=epoch_now)
        details = {
            key: "no unseen cycle; persisted state read without mutation" for key in pool_keys
        }
        resets: list[str] = []
        if epoch_declared_now:
            resets.append(
                f"epoch {epoch_now} declared after {epoch_start_after!r}; historical "
                "cycles were not replayed because the pool and bets must be fixed "
                "before observing monitored outcomes"
            )
        seen_reset_notices: set[str] = set()

        for cycle in unseen:
            cycle_frame = frame[frame["cycle_id"] == cycle]
            suppressed = _suppressed_families(ref_frame_all, cycle_frame, families)
            if suppressed:
                names = ", ".join(sorted(suppressed))
                raise ValueError(
                    f"cycle {cycle!r} is not composition-comparable to the frozen "
                    f"baseline for families: {names}. Refusing to choose E_t=1 after "
                    "observing the cycle because that would invalidate optional-stopping "
                    "control. Repair collection or re-baseline after this cycle."
                )

            for key in pool_keys:
                baseline_name, family, sig, _channel = key
                ref_cycles = ref_by_baseline.get(baseline_name, [])
                ref_frame = frame[frame["cycle_id"].isin(ref_cycles)]
                ref_fam = ref_frame[ref_frame["family"] == family][sig].dropna()
                cur_fam = cycle_frame[cycle_frame["family"] == family][sig].dropna()
                state = working[key]
                reference = (int(ref_fam.sum()), len(ref_fam))

                if cur_fam.empty:
                    update = _neutral_update(
                        state,
                        fingerprint=fingerprint,
                        alpha_prime=decision_alpha,
                        reference=reference,
                        detail=(
                            "no current-cycle trials: neutral evidence; availability "
                            "must be determined independently of the outcome"
                        ),
                    )
                else:
                    prior = PriorState(
                        n_cycles=state.prior.n_cycles,
                        successes=state.prior.successes,
                        trials=state.prior.trials,
                        reference_successes=reference[0],
                        reference_trials=reference[1],
                    )
                    successes, trials = int(cur_fam.sum()), len(cur_fam)
                    outcome = rate_evalue(
                        successes,
                        trials,
                        prior,
                        gamma=gamma_i,
                        grid=grid,
                        frozen_reference=(baseline_name == ANYTIME_BASELINE),
                    )
                    update = update_process(
                        state,
                        outcome,
                        fingerprint=fingerprint,
                        alpha_prime=decision_alpha,
                        successes=successes,
                        trials=trials,
                        reference=reference,
                    )

                if update.reset_notice:
                    notice = f"{key}: {update.reset_notice}"
                    if notice not in seen_reset_notices:
                        resets.append(notice)
                        seen_reset_notices.add(notice)
                working[key] = update.state
                details[key] = update.outcome.detail

        # A no-op invocation is a read, including after a report command.
        # Present a reset view if the instrument changed before any post-
        # golden observation, but do not mutate persisted state without a
        # cycle to fold.
        visible: list[EProcessState] = []
        for key in pool_keys:
            state = working[key]
            if not unseen and state.fingerprint and state.fingerprint != fingerprint:
                state = state.reset_for(fingerprint)
            elif not state.fingerprint:
                state = replace(state, fingerprint=fingerprint)
            visible.append(state)

        processes = [
            ProcessReport(
                key=key,
                log_wealth=state.log_wealth,
                evidence=float(min(state.log_wealth, 700.0)),
                epoch=state.epoch,
                cycles=state.cycles,
                bets_placed=state.bets_placed,
                rise_cycle=state.rise_cycle,
                crossed_at=state.crossed_at,
                # Crossings are absorbing within an epoch. Degradation is a
                # presentation warning, not an outcome-dependent skip rule.
                rejected=bool(state.crossed_at is not None),
                detail=details[key],
            )
            for key, state in zip(pool_keys, visible, strict=True)
        ]

        n_alerts = sum(1 for process in processes if process.rejected)
        monitored_families = {key[1] for key in pool_keys}
        current_suppressed = (
            _suppressed_families(ref_frame_all, cur_frame, families)
            if golden and current_is_eligible
            else set()
        )
        uncovered = current_suppressed | (set(families) - monitored_families)
        if not golden:
            coverage_status = "NO REFERENCE"
        elif not current_is_eligible:
            coverage_status = "NO OBSERVATION"
        elif not monitored_families or uncovered == set(families):
            coverage_status = "NONE"
        elif uncovered:
            coverage_status = "PARTIAL"
        else:
            coverage_status = "FULL"

        if not golden:
            verdict = NO_BASELINE_VERDICT
        elif n_alerts:
            verdict = "DRIFT DETECTED"
        elif degraded:
            verdict = "DEGRADED DATA"
        elif not current_is_eligible:
            verdict = "NO CURRENT OBSERVATION"
        elif coverage_status == "NONE":
            verdict = "NO VALID COMPARISON"
        elif coverage_status == "PARTIAL":
            verdict = "PARTIAL COVERAGE"
        else:
            verdict = "OK"

        result = AnytimeCheckResult(
            ts=ts,
            current_cycle=current,
            fingerprint=fingerprint,
            alpha=effective_alpha,
            alpha_prime=effective_alpha_prime,
            gamma_total=effective_gamma_total,
            gamma_per_process=gamma_i,
            n_processes=len(pool_keys),
            pool_declared_now=pool_declared_now,
            verdict=verdict,
            degraded=degraded,
            degraded_cycles=degraded_cycles,
            coverage_status=coverage_status,
            suppressed_families=tuple(sorted(uncovered)),
            processes=processes,
            resets=resets,
            processed_cycles=tuple(unseen),
            snapshot_log_offset=snapshot_offset,
            snapshot_finalized_revision=snapshot_revision,
            snapshot_record_ids=tuple(record.id for record in records),
        )

        if unseen or epoch_declared_now:
            save_states(store, list(working.values()) if unseen else visible, ts, commit=False)
            if unseen:
                conn.executemany(
                    "INSERT INTO anytime_processed_cycles "
                    "(fingerprint, cycle_id, processed_ts) VALUES (?, ?, ?)",
                    [(fingerprint, cycle, ts) for cycle in unseen],
                )
            _persist(store, result, commit=False)
        conn.commit()
        return result
    except BaseException:
        conn.rollback()
        raise


def _persist(store: Store, result: AnytimeCheckResult, *, commit: bool = True) -> None:
    """Persist one mutating check, optionally joining the caller's transaction."""
    conn = store.connect()
    conn.execute(
        "INSERT INTO checks "
        "(ts, baseline_kind, params_json, verdict, snapshot_offset, "
        "snapshot_record_ids_json) VALUES (?, ?, ?, ?, ?, ?)",
        (
            result.ts,
            "anytime",
            json.dumps(
                {
                    "current_cycle": result.current_cycle,
                    "fingerprint": result.fingerprint,
                    "alpha": result.alpha,
                    "alpha_prime": result.alpha_prime,
                    "gamma_total": result.gamma_total,
                    "gamma_per_process": result.gamma_per_process,
                    "n_processes": result.n_processes,
                    "pool_declared_now": result.pool_declared_now,
                    "coverage_status": result.coverage_status,
                    "degraded_cycles": result.degraded_cycles,
                    "suppressed_families": result.suppressed_families,
                    "processed_cycles": result.processed_cycles,
                    "ville_threshold_log": ville_threshold(
                        result.alpha_prime / max(result.n_processes, 1)
                    ),
                    "snapshot_log_offset": result.snapshot_log_offset,
                    "snapshot_finalized_revision": result.snapshot_finalized_revision,
                    "snapshot_record_count": len(result.snapshot_record_ids),
                }
            ),
            result.verdict,
            result.snapshot_log_offset,
            None,
        ),
    )
    if commit:
        conn.commit()


def wealth_table(result: AnytimeCheckResult) -> pd.DataFrame:
    """Per-process wealth as a frame, for reports and notebooks."""
    return pd.DataFrame(
        [
            {
                "process": p.label,
                "log_wealth": round(p.log_wealth, 3),
                "epoch": p.epoch,
                "cycles": p.cycles,
                "bets": p.bets_placed,
                "onset_cycle": p.rise_cycle,
                "crossed_at": p.crossed_at,
                "alert": p.rejected,
            }
            for p in sorted(result.processes, key=lambda x: -x.log_wealth)
        ]
    )
