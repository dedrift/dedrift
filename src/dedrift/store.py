"""Storage layer: append-only JSONL logs plus a SQLite index (SPEC.md §2.2).

Layout under the project root::

    .dedrift/
        config.toml        # project config (created by `dedrift init`)
        logs/records.jsonl # append-only source of truth
        index.db           # SQLite: records, config_events, checks, alerts

JSONL is the source of truth; SQLite is a derived index and can be rebuilt
from the logs at any time (``Store.rebuild_index``).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, TextIO

from dedrift.schema import InteractionRecord

PROJECT_DIR = ".dedrift"
LOGS_DIR = "logs"
RECORDS_FILE = "records.jsonl"
INDEX_FILE = "index.db"
CONFIG_FILE = "config.toml"

_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_SQLITE_BUSY_TIMEOUT_MS = 30_000

DEFAULT_CONFIG_TOML = """\
# dedrift project configuration
[project]
name = "default"
canary_repetitions = 7      # N repeated runs per canary per cycle
rolling_window_cycles = 5   # K cycles for the rolling reference

[detection]
fdr_q = 0.05                # Benjamini-Hochberg FDR level
permutations = 500          # permutation-test resamples (seeded)
seed = 1729                 # global seed recorded in every report
ph_lambda = 12.0            # Page-Hinkley alarm threshold (reference-SD units).
ph_delta = 0.3              # Idealized null crossing bound 2*exp(-2*delta*lambda)
                            # ~= 0.15% idealized; measured ~8.5%/stream (estimated centering
                            # and scale) - see the PH calibration test.
inference = "fixed"         # "fixed" or opt-in "anytime"

# Materiality (effect-size) gates, per signature channel. Alerts require BOTH
# statistical significance after FDR AND an effect exceeding these bands.
[materiality]
refusal_rate_pp = 2.0       # percentage-point shift
format_validity_pp = 1.0    # percentage-point shift
rate_default_pp = 2.0       # other rate signatures, percentage points
scalar_cohen_d = 0.5        # reserved compatibility key; Welch cannot alert in v0.3.1
ks_distance = 0.15          # KS statistic D floor (sup-norm CDF distance):
                            # KS catches shape changes with equal means,
                            # which Cohen's d would wrongly gate out
dispersion_ratio = 1.5      # robust dispersion (MAD) must grow/shrink by this
p95_relative = 0.10         # relative P95 shift floor
embedding_mmd2_floor = -1.0 # -1 = auto-calibrate; 0 = off; >0 = explicit

[embeddings]
# Pinned forever per project once set; dedrift refuses cross-embedder comparisons.
model = ""

[anytime]
alpha = 0.05
gamma_total = 0.02
tilts = [1.5, 2.0, 3.0]
epoch_allocation = "per_epoch"
"""

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    canary_id TEXT,
    cycle_id TEXT,
    repetition INTEGER,
    config_fingerprint TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    retries INTEGER NOT NULL,
    n_errors INTEGER NOT NULL,
    n_tool_calls INTEGER NOT NULL,
    record_digest TEXT,
    record_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_ts ON records (ts);
CREATE INDEX IF NOT EXISTS idx_records_canary ON records (canary_id, ts);
CREATE INDEX IF NOT EXISTS idx_records_cycle ON records (cycle_id);
CREATE INDEX IF NOT EXISTS idx_records_fingerprint ON records (config_fingerprint);

-- A check may only consume a finalized canary cycle. ``append`` leaves a
-- cycle open for streaming ingestion; complete-batch producers finalize it
-- atomically with their records. Once finalized, no new record ID can be
-- added, which makes both fixed snapshots and anytime exactly-once state
-- immutable.
CREATE TABLE IF NOT EXISTS canary_cycles (
    cycle_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('open', 'finalized')),
    expected_records INTEGER,
    finalized_records INTEGER,
    opened_ts TEXT NOT NULL,
    finalized_ts TEXT,
    finalized_offset INTEGER
);
CREATE INDEX IF NOT EXISTS idx_canary_cycles_status
    ON canary_cycles(status, finalized_offset);

CREATE TABLE IF NOT EXISTS config_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    old_fingerprint TEXT,
    new_fingerprint TEXT NOT NULL,
    first_record_id TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_config_events_ts ON config_events (ts);

CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    baseline_kind TEXT NOT NULL,      -- 'dual' (fixed path) | 'anytime'
    params_json TEXT NOT NULL,
    verdict TEXT NOT NULL,
    snapshot_offset INTEGER,
    snapshot_record_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_id INTEGER NOT NULL REFERENCES checks (id),
    signature TEXT NOT NULL,
    family TEXT NOT NULL,
    test TEXT NOT NULL,
    p_adjusted REAL,
    effect_size REAL NOT NULL,
    effect_units TEXT NOT NULL,
    details_json TEXT NOT NULL
);

-- Anytime-valid inference state. Unlike the fixed-sample path, which is
-- stateless (it just compares windows), an e-process accumulates across
-- cycles, so its wealth must survive between invocations. The fingerprint
-- column is what makes the guarantee honest: evidence gathered under a
-- different suite, embedder, golden baseline or extractor concerns a
-- different null, so a fingerprint change resets the row rather than
-- silently continuing.
CREATE TABLE IF NOT EXISTS eprocess_state (
    baseline TEXT NOT NULL,
    family TEXT NOT NULL,
    signature TEXT NOT NULL,
    channel TEXT NOT NULL,
    log_wealth REAL NOT NULL,
    cycles INTEGER NOT NULL,
    bets_placed INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    peak_log_wealth REAL NOT NULL,
    rise_cycle INTEGER,
    crossed_at INTEGER,
    prior_successes INTEGER NOT NULL,
    prior_trials INTEGER NOT NULL,
    reference_successes INTEGER NOT NULL,
    reference_trials INTEGER NOT NULL,
    updated_ts TEXT NOT NULL,
    PRIMARY KEY (baseline, family, signature, channel)
);

-- The e-process pool, declared once per epoch and then frozen.
--
-- Why this is a table rather than a per-cycle computation: the pool size
-- sets the per-process coverage budget, which sets the nuisance interval,
-- which is part of the bet. For a frozen golden baseline the guarantee
-- relies on that interval being a SINGLE fixed event settled at epoch
-- start; a pool that shrank or grew between cycles would quietly turn it
-- into a sequence of different events, and the coverage budget would no
-- longer cover what it claims. So membership is decided once, from data
-- already in hand, and persisted.
CREATE TABLE IF NOT EXISTS epoch_pool (
    fingerprint TEXT NOT NULL,
    baseline TEXT NOT NULL,
    family TEXT NOT NULL,
    signature TEXT NOT NULL,
    channel TEXT NOT NULL,
    declared_ts TEXT NOT NULL,
    PRIMARY KEY (fingerprint, baseline, family, signature, channel)
);

-- Durable idempotency ledger for anytime-valid processing. A cycle and all
-- e-process updates derived from it are committed in one SQLite transaction.
CREATE TABLE IF NOT EXISTS anytime_processed_cycles (
    fingerprint TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    processed_ts TEXT NOT NULL,
    PRIMARY KEY (fingerprint, cycle_id)
);
CREATE INDEX IF NOT EXISTS idx_anytime_processed_cycle
    ON anytime_processed_cycles(cycle_id);

-- Global epoch numbering is separate from per-process state. A changed
-- suite can replace every process key, so deriving the next geometric
-- allocation from surviving rows would accidentally restart at epoch zero.
CREATE TABLE IF NOT EXISTS anytime_epochs (
    fingerprint TEXT PRIMARY KEY,
    base_fingerprint TEXT NOT NULL,
    epoch_index INTEGER NOT NULL UNIQUE,
    start_after_cycle TEXT NOT NULL,
    declared_ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_anytime_epochs_base
    ON anytime_epochs(base_fingerprint, epoch_index);

-- Byte offset through which records.jsonl has been indexed. This makes the
-- unavoidable JSONL-before-SQLite crash window recoverable on the next
-- append without rescanning an ever-growing log on every healthy write.
CREATE TABLE IF NOT EXISTS store_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _harden_permissions(path: Path, mode: int) -> None:
    """Set owner-only permissions on a managed path on POSIX systems."""
    if os.name == "posix" and path.exists():
        path.chmod(mode)


def _open_private_append(path: Path) -> TextIO:
    """Open an owner-only append stream, creating the file when absent."""
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, _PRIVATE_FILE_MODE)
    _harden_permissions(path, _PRIVATE_FILE_MODE)
    return os.fdopen(descriptor, "a", encoding="utf-8")


def _write_new_private_file(path: Path, content: str) -> None:
    """Create and durably write a new owner-only UTF-8 file."""
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        _PRIVATE_FILE_MODE,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _atomic_private_writer(path: Path) -> Iterator[BinaryIO]:
    """Yield an owner-only temporary file and atomically replace ``path``.

    The file and its containing directory are synced before return.  This is
    used for small mutable project artifacts (baseline, embedder pin, and
    embedding cache) that must never be observed half-written after a crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
    _harden_permissions(path.parent, _PRIVATE_DIR_MODE)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        _PRIVATE_FILE_MODE,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _harden_permissions(path, _PRIVATE_FILE_MODE)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


class Store:
    """Append-only JSONL log with a SQLite index for one dedrift project.

    The store tracks the last-seen config fingerprint; whenever an appended
    record carries a different fingerprint, a row is added to
    ``config_events`` — these events drive attribution.

    Args:
        root: Directory containing (or to contain) the ``.dedrift`` project
            directory. Defaults to the current working directory.
    """

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)
        self.project_dir = self.root / PROJECT_DIR
        self.records_path = self.project_dir / LOGS_DIR / RECORDS_FILE
        self.index_path = self.project_dir / INDEX_FILE
        self.config_path = self.project_dir / CONFIG_FILE
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def init_project(cls, root: Path | str = ".") -> Store:
        """Create the ``.dedrift`` project structure (idempotent).

        Args:
            root: Directory in which to create the project.

        Returns:
            An open store for the new project.
        """
        store = cls(root)
        store.project_dir.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
        store.records_path.parent.mkdir(exist_ok=True, mode=_PRIVATE_DIR_MODE)
        _harden_permissions(store.project_dir, _PRIVATE_DIR_MODE)
        _harden_permissions(store.records_path.parent, _PRIVATE_DIR_MODE)
        if not store.config_path.exists():
            with suppress(FileExistsError):  # another initializer may win the race
                _write_new_private_file(store.config_path, DEFAULT_CONFIG_TOML)
        _harden_permissions(store.config_path, _PRIVATE_FILE_MODE)
        descriptor = os.open(
            store.records_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            _PRIVATE_FILE_MODE,
        )
        os.close(descriptor)
        _harden_permissions(store.records_path, _PRIVATE_FILE_MODE)
        store.connect()
        return store

    def exists(self) -> bool:
        """Return True if this looks like an initialized project."""
        return self.config_path.exists() and self.records_path.exists()

    def connect(self) -> sqlite3.Connection:
        """Open the SQLite index, migrate it, and reconcile a durable log tail.

        A complete JSONL line may survive a crash just before the matching
        SQLite commit. Reconciliation on *open* keeps reads, attribution, and
        counts consistent even when no later append happens to trigger repair.
        """
        if self._conn is None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
            _harden_permissions(self.index_path.parent, _PRIVATE_DIR_MODE)
            self._conn = sqlite3.connect(self.index_path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000)
            self._conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            self._conn.executescript(_SCHEMA_SQL)
            columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(records)")}
            if "record_digest" not in columns:
                self._conn.execute("ALTER TABLE records ADD COLUMN record_digest TEXT")
            if "record_json" not in columns:
                self._conn.execute("ALTER TABLE records ADD COLUMN record_json TEXT")
            check_columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(checks)")}
            if "snapshot_offset" not in check_columns:
                self._conn.execute("ALTER TABLE checks ADD COLUMN snapshot_offset INTEGER")
            if "snapshot_record_ids_json" not in check_columns:
                self._conn.execute("ALTER TABLE checks ADD COLUMN snapshot_record_ids_json TEXT")
            self._conn.commit()
            if self.records_path.exists():
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    # A pre-v0.3.1 index has no cached canonical payloads.
                    # Force one source-of-truth scan to backfill them before
                    # future checks switch to indexed window reads.
                    missing_payload = self._conn.execute(
                        "SELECT 1 FROM records WHERE record_json IS NULL LIMIT 1"
                    ).fetchone()
                    if missing_payload is not None:
                        self._conn.execute(
                            "DELETE FROM store_metadata WHERE key = 'jsonl_indexed_offset'"
                        )
                    self._reconcile_jsonl_tail(self._conn)
                    self._sync_cycle_registry(
                        self._conn,
                        # Completion cannot be reconstructed from durable
                        # records alone. Upgraded histories fail closed as
                        # OPEN until the operator finalizes them explicitly.
                        legacy_finalize=False,
                    )
                    self._conn.commit()
                except BaseException:
                    if self._conn.in_transaction:
                        self._conn.rollback()
                    raise
            self._harden_sqlite_files()
        return self._conn

    def _harden_sqlite_files(self) -> None:
        """Keep the database and transient WAL files owner-readable only."""
        for path in (
            self.index_path,
            Path(f"{self.index_path}-wal"),
            Path(f"{self.index_path}-shm"),
        ):
            _harden_permissions(path, _PRIVATE_FILE_MODE)

    def close(self) -> None:
        """Close the SQLite connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Store:
        """Open the store as a context manager."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close on context exit."""
        self.close()

    # -- writes --------------------------------------------------------------

    def append(self, record: InteractionRecord) -> None:
        """Append one streaming record without finalizing its canary cycle.

        Emits a config event if the record's config fingerprint differs from
        the previously indexed one.

        Args:
            record: The record to persist.
        """
        self.append_many([record], finalize_cycles=False)

    def append_many(
        self,
        records: list[InteractionRecord],
        *,
        finalize_cycles: bool = True,
        expected_cycle_counts: Mapping[str, int] | None = None,
    ) -> None:
        """Durably append a batch, then update the rebuildable index.

        The SQLite ``BEGIN IMMEDIATE`` lock serializes writers before they
        touch JSONL. JSONL is flushed and fsynced first because it is the
        source of truth; SQLite commits only after that durability boundary.
        A failure between those steps can leave the index behind the log,
        but never the reverse. The next append reconciles a complete,
        fsynced tail before applying an idempotent retry; ``rebuild_index``
        remains the explicit full-repair path.

        Args:
            records: Records in the order they occurred.
            finalize_cycles: Treat every canary cycle represented in this
                batch as complete. This is the complete-batch API; use
                :meth:`append` or pass False while streaming a cycle, then
                call :meth:`finalize_cycle`.
            expected_cycle_counts: Optional exact record count per cycle.
                Finalization fails closed when an observed count differs.
        """
        unique: dict[str, tuple[InteractionRecord, str]] = {}
        for record in records:
            serialized = record.to_jsonl()
            # Freeze the caller-owned, shallowly immutable Pydantic object at
            # the persistence boundary. Every later digest/index operation
            # uses this detached snapshot, so concurrent nested mutation
            # cannot make JSONL and SQLite disagree.
            snapshot = InteractionRecord.from_jsonl(serialized)
            line = serialized + "\n"
            previous = unique.get(snapshot.id)
            if previous is not None and previous[1] != line:
                raise ValueError(f"batch contains conflicting records with id {snapshot.id!r}")
            unique.setdefault(snapshot.id, (snapshot, line))
        if not unique:
            return
        expected = dict(expected_cycle_counts or {})
        if any(not cycle_id or count < 1 for cycle_id, count in expected.items()):
            raise ValueError("expected cycle counts require non-empty IDs and counts >= 1")
        batch_cycle_ids = sorted(
            {
                str(record.cycle_id)
                for record, _ in unique.values()
                if record.source.value == "canary" and record.cycle_id is not None
            }
        )
        unexpected = sorted(set(expected) - set(batch_cycle_ids))
        if unexpected:
            raise ValueError(
                "expected cycle count supplied for a cycle absent from the batch: "
                + ", ".join(unexpected)
            )
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._reconcile_jsonl_tail(conn)
            pending: list[tuple[InteractionRecord, str]] = []
            for record_id, pair in unique.items():
                record, line = pair
                row = conn.execute(
                    "SELECT record_digest FROM records WHERE id = ?", (record_id,)
                ).fetchone()
                if row is None:
                    pending.append(pair)
                    continue
                incoming_digest = self._record_digest(record)
                stored_digest = str(row[0]) if row[0] is not None else None
                if stored_digest is None:
                    stored_digest = self._digest_for_logged_id(record_id)
                    if stored_digest is None:
                        raise RuntimeError(
                            f"record {record_id!r} is indexed but absent from records.jsonl"
                        )
                    conn.execute(
                        "UPDATE records SET record_digest = ? WHERE id = ?",
                        (stored_digest, record_id),
                    )
                if stored_digest != incoming_digest:
                    raise ValueError(
                        f"record id {record_id!r} already exists with a different payload"
                    )
            processed_anytime_cycles = {
                str(row[0])
                for row in conn.execute("SELECT DISTINCT cycle_id FROM anytime_processed_cycles")
            }
            extended_cycles = sorted(
                {
                    str(record.cycle_id)
                    for record, _ in pending
                    if record.source.value == "canary"
                    and record.cycle_id is not None
                    and record.cycle_id in processed_anytime_cycles
                }
            )
            if extended_cycles:
                raise ValueError(
                    "cannot add new record IDs to canary cycle(s) already processed "
                    "by anytime inference: "
                    + ", ".join(extended_cycles)
                    + "; append the complete cycle before checking it "
                    "(exact-ID retries remain allowed)"
                )
            finalized_cycles = {
                str(row[0])
                for row in conn.execute(
                    "SELECT cycle_id FROM canary_cycles WHERE status = 'finalized'"
                )
            }
            extended_finalized = sorted(
                {
                    str(record.cycle_id)
                    for record, _ in pending
                    if record.source.value == "canary"
                    and record.cycle_id is not None
                    and record.cycle_id in finalized_cycles
                }
            )
            if extended_finalized:
                raise ValueError(
                    "cannot add new record IDs to finalized canary cycle(s): "
                    + ", ".join(extended_finalized)
                    + "; reopen under a new cycle ID (exact-ID retries remain allowed)"
                )
            pending_per_cycle: dict[str, int] = {}
            for record, _ in pending:
                if record.source.value == "canary" and record.cycle_id is not None:
                    pending_per_cycle[record.cycle_id] = (
                        pending_per_cycle.get(record.cycle_id, 0) + 1
                    )
            for cycle_id in batch_cycle_ids:
                row = conn.execute(
                    "SELECT expected_records FROM canary_cycles WHERE cycle_id = ?",
                    (cycle_id,),
                ).fetchone()
                declared = int(row[0]) if row is not None and row[0] is not None else None
                requested = expected.get(cycle_id)
                if declared is not None and requested is not None and declared != requested:
                    raise ValueError(
                        f"cycle {cycle_id!r} expected count is already {declared}, not {requested}"
                    )
                limit = declared if declared is not None else requested
                if limit is not None:
                    current_count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM records WHERE source = 'canary' AND cycle_id = ?",
                            (cycle_id,),
                        ).fetchone()[0]
                    )
                    projected = current_count + pending_per_cycle.get(cycle_id, 0)
                    if projected > limit:
                        raise ValueError(
                            f"cycle {cycle_id!r} would contain {projected} records, "
                            f"exceeding its declared {limit}"
                        )
            if pending:
                with _open_private_append(self.records_path) as stream:
                    stream.writelines(line for _, line in pending)
                    stream.flush()
                    os.fsync(stream.fileno())
            for record, _ in pending:
                self._index_record(conn, record)
            log_offset = self.records_path.stat().st_size
            self._set_indexed_offset(conn, log_offset)
            opened_ts = datetime.now(timezone.utc).isoformat()
            for cycle_id in batch_cycle_ids:
                conn.execute(
                    "INSERT INTO canary_cycles "
                    "(cycle_id, status, expected_records, opened_ts) "
                    "VALUES (?, 'open', ?, ?) "
                    "ON CONFLICT(cycle_id) DO UPDATE SET expected_records = "
                    "COALESCE(canary_cycles.expected_records, excluded.expected_records)",
                    (cycle_id, expected.get(cycle_id), opened_ts),
                )
            if finalize_cycles:
                for cycle_id in batch_cycle_ids:
                    self._finalize_cycle_locked(
                        conn,
                        cycle_id,
                        expected_records=expected.get(cycle_id),
                        finalized_offset=log_offset,
                    )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            self._harden_sqlite_files()

    def _sync_cycle_registry(
        self,
        conn: sqlite3.Connection,
        *,
        legacy_finalize: bool,
    ) -> None:
        """Register indexed canary cycles missing lifecycle metadata.

        Cycles from an index created before the lifecycle table existed are
        finalized once during migration for backward compatibility. A cycle
        recovered from a later crash tail is left open, because durability of
        its records does not prove that the producer finished the cycle.
        """
        rows = conn.execute(
            "SELECT r.cycle_id, COUNT(*), MIN(r.ts) "
            "FROM records AS r "
            "LEFT JOIN canary_cycles AS c ON c.cycle_id = r.cycle_id "
            "WHERE r.source = 'canary' AND r.cycle_id IS NOT NULL "
            "AND c.cycle_id IS NULL GROUP BY r.cycle_id ORDER BY MIN(r.rowid)"
        ).fetchall()
        now = datetime.now(timezone.utc).isoformat()
        offset = self.records_path.stat().st_size if self.records_path.exists() else 0
        for cycle_id, count, first_ts in rows:
            if legacy_finalize:
                conn.execute(
                    "INSERT INTO canary_cycles "
                    "(cycle_id, status, expected_records, finalized_records, "
                    "opened_ts, finalized_ts, finalized_offset) "
                    "VALUES (?, 'finalized', ?, ?, ?, ?, ?)",
                    (str(cycle_id), int(count), int(count), str(first_ts), now, offset),
                )
            else:
                conn.execute(
                    "INSERT INTO canary_cycles (cycle_id, status, opened_ts) VALUES (?, 'open', ?)",
                    (str(cycle_id), str(first_ts)),
                )

    def _finalize_cycle_locked(
        self,
        conn: sqlite3.Connection,
        cycle_id: str,
        *,
        expected_records: int | None,
        finalized_offset: int,
    ) -> None:
        """Finalize one cycle inside the caller's writer transaction."""
        row = conn.execute(
            "SELECT status, expected_records, finalized_records "
            "FROM canary_cycles WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown canary cycle {cycle_id!r}")
        status = str(row[0])
        declared = int(row[1]) if row[1] is not None else None
        previous_count = int(row[2]) if row[2] is not None else None
        if expected_records is not None and expected_records < 1:
            raise ValueError("expected_records must be >= 1")
        if declared is not None and expected_records is not None and declared != expected_records:
            raise ValueError(
                f"cycle {cycle_id!r} expected count is already {declared}, not {expected_records}"
            )
        expected = declared if declared is not None else expected_records
        count = int(
            conn.execute(
                "SELECT COUNT(*) FROM records WHERE source = 'canary' AND cycle_id = ?",
                (cycle_id,),
            ).fetchone()[0]
        )
        if count == 0:
            raise ValueError(f"cannot finalize empty canary cycle {cycle_id!r}")
        coordinates = conn.execute(
            "SELECT canary_id, repetition FROM records "
            "WHERE source = 'canary' AND cycle_id = ? ORDER BY canary_id, repetition",
            (cycle_id,),
        ).fetchall()
        by_canary: dict[str, list[int]] = {}
        for canary_id, repetition in coordinates:
            if canary_id is None or repetition is None:
                raise ValueError(
                    f"cannot finalize cycle {cycle_id!r}: every canary record requires "
                    "canary_id and repetition"
                )
            by_canary.setdefault(str(canary_id), []).append(int(repetition))
        duplicate_coordinates = [
            f"{canary_id}#{repetition}"
            for canary_id, repetitions in by_canary.items()
            for repetition in sorted(set(repetitions))
            if repetitions.count(repetition) > 1
        ]
        if duplicate_coordinates:
            raise ValueError(
                f"cannot finalize cycle {cycle_id!r}: duplicate "
                "(canary_id, repetition) coordinates: " + ", ".join(duplicate_coordinates[:10])
            )
        repetition_designs = {tuple(repetitions) for repetitions in by_canary.values()}
        if len(repetition_designs) != 1:
            raise ValueError(
                f"cannot finalize cycle {cycle_id!r}: canaries have different repetition sets"
            )
        repetition_design = next(iter(repetition_designs))
        if repetition_design != tuple(range(1, max(repetition_design) + 1)):
            raise ValueError(
                f"cannot finalize cycle {cycle_id!r}: repetitions must be contiguous "
                "from 1 for every canary"
            )
        if expected is not None and count != expected:
            raise ValueError(
                f"cannot finalize cycle {cycle_id!r}: observed {count} records, expected {expected}"
            )
        if status == "finalized":
            if previous_count != count:
                raise RuntimeError(
                    f"finalized cycle {cycle_id!r} changed from {previous_count} to {count} records"
                )
            return
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE canary_cycles SET status = 'finalized', "
            "expected_records = ?, finalized_records = ?, finalized_ts = ?, "
            "finalized_offset = ? WHERE cycle_id = ?",
            (expected if expected is not None else count, count, now, finalized_offset, cycle_id),
        )

    def finalize_cycle(self, cycle_id: str, *, expected_records: int | None = None) -> None:
        """Mark a streamed canary cycle complete after validating its size."""
        if not cycle_id:
            raise ValueError("cycle_id must be non-empty")
        conn = self.connect()
        if conn.in_transaction:
            raise RuntimeError("finalize_cycle requires ownership of the SQLite transaction")
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._reconcile_jsonl_tail(conn)
            self._sync_cycle_registry(conn, legacy_finalize=False)
            self._finalize_cycle_locked(
                conn,
                cycle_id,
                expected_records=expected_records,
                finalized_offset=self.records_path.stat().st_size,
            )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            self._harden_sqlite_files()

    def _reconcile_jsonl_tail(self, conn: sqlite3.Connection) -> None:
        """Index complete JSONL records left by a pre-commit failure.

        The first append after upgrading scans the existing log once because
        older stores have no byte checkpoint. Later calls read only bytes
        beyond the last transactionally committed offset.
        """
        row = conn.execute(
            "SELECT value FROM store_metadata WHERE key = 'jsonl_indexed_offset'"
        ).fetchone()
        offset = int(row[0]) if row is not None else 0
        size = self.records_path.stat().st_size
        if not 0 <= offset <= size:
            raise RuntimeError(
                "records.jsonl is shorter than its indexed checkpoint; "
                "restore the append-only log before rebuilding the index"
            )
        with self.records_path.open("rb") as stream:
            stream.seek(offset)
            while raw_line := stream.readline():
                if not raw_line.endswith(b"\n"):
                    raise RuntimeError(
                        "records.jsonl has an incomplete trailing record; "
                        "repair the trailing line before retrying"
                    )
                stripped = raw_line.strip()
                if not stripped:
                    continue
                record = InteractionRecord.from_jsonl(stripped.decode("utf-8"))
                digest = self._record_digest(record)
                existing = conn.execute(
                    "SELECT record_digest, record_json FROM records WHERE id = ?",
                    (record.id,),
                ).fetchone()
                if existing is None:
                    self._index_record(conn, record)
                elif existing[0] is None or existing[1] is None:
                    conn.execute(
                        "UPDATE records SET record_digest = ?, record_json = ? WHERE id = ?",
                        (digest, record.to_jsonl(), record.id),
                    )
                elif str(existing[0]) != digest:
                    raise RuntimeError(
                        f"records.jsonl contains conflicting payloads for id {record.id!r}"
                    )
            self._set_indexed_offset(conn, stream.tell())

    @staticmethod
    def _record_digest(record: InteractionRecord) -> str:
        payload = record.to_jsonl().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _digest_for_logged_id(self, record_id: str) -> str | None:
        found: str | None = None
        with self.records_path.open("rb") as stream:
            for raw_line in stream:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                record = InteractionRecord.from_jsonl(stripped.decode("utf-8"))
                if record.id != record_id:
                    continue
                digest = self._record_digest(record)
                if found is not None and found != digest:
                    raise RuntimeError(
                        f"records.jsonl contains conflicting payloads for id {record_id!r}"
                    )
                found = digest
        return found

    @staticmethod
    def _set_indexed_offset(conn: sqlite3.Connection, offset: int) -> None:
        conn.execute(
            "INSERT INTO store_metadata (key, value) VALUES ('jsonl_indexed_offset', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(offset),),
        )

    def _index_record(self, conn: sqlite3.Connection, record: InteractionRecord) -> None:
        fingerprint = record.config_fingerprint
        last = conn.execute(
            "SELECT new_fingerprint FROM config_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_fingerprint = last[0] if last is not None else None
        if fingerprint != last_fingerprint:
            conn.execute(
                "INSERT INTO config_events (ts, old_fingerprint, new_fingerprint,"
                " first_record_id, config_json) VALUES (?, ?, ?, ?, ?)",
                (
                    record.ts.isoformat(),
                    last_fingerprint,
                    fingerprint,
                    record.id,
                    record.config.model_dump_json(),
                ),
            )
        conn.execute(
            "INSERT OR REPLACE INTO records (id, ts, source, canary_id, cycle_id,"
            " repetition, config_fingerprint, latency_ms, tokens_in, tokens_out, steps,"
            " retries, n_errors, n_tool_calls, record_digest, record_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.ts.isoformat(),
                record.source.value,
                record.canary_id,
                record.cycle_id,
                record.repetition,
                fingerprint,
                record.latency_ms,
                record.tokens_in,
                record.tokens_out,
                record.steps,
                record.retries,
                len(record.errors),
                len(record.tool_calls),
                self._record_digest(record),
                record.to_jsonl(),
            ),
        )

    # -- reads ---------------------------------------------------------------

    def read_records(self) -> list[InteractionRecord]:
        """Read one complete, writer-serialized JSONL snapshot.

        Package writers acquire the same SQLite ``BEGIN IMMEDIATE`` lock
        before touching JSONL, so this read observes either the complete
        state before a batch or the complete state after it. File metadata
        is also checked around the read to detect unsupported external
        writers that bypass the store lock.

        Returns:
            All records in append order.
        """
        if not self.records_path.exists():
            return []
        conn = self.connect()
        owns_transaction = not conn.in_transaction
        if owns_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            records = self._read_stable_snapshot()
            if owns_transaction:
                conn.commit()
            return records
        except BaseException:
            if owns_transaction and conn.in_transaction:
                conn.rollback()
            raise

    def _read_stable_snapshot(self) -> list[InteractionRecord]:
        for _ in range(3):
            before = self.records_path.stat()
            with self.records_path.open("rb") as stream:
                payload = stream.read(before.st_size)
            after = self.records_path.stat()
            unchanged = (
                len(payload) == before.st_size
                and before.st_ino == after.st_ino
                and before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
            )
            if not unchanged:
                continue
            if payload and not payload.endswith(b"\n"):
                raise RuntimeError(
                    "records.jsonl has an incomplete trailing record; "
                    "repair the trailing line before retrying"
                )
            records: list[InteractionRecord] = []
            for raw_line in payload.splitlines():
                stripped = raw_line.strip()
                if stripped:
                    records.append(InteractionRecord.from_jsonl(stripped.decode("utf-8")))
            return records
        raise RuntimeError("records.jsonl changed repeatedly during snapshot; retry the operation")

    def read_finalized_canary_snapshot(self) -> tuple[list[InteractionRecord], int]:
        """Read finalized canary records from the derived index in one WAL snapshot.

        Unlike a full JSONL scan this is proportional to the canary history
        represented in the index, does not take a writer reservation, and
        excludes open cycles by construction. The returned offset and record
        IDs can be persisted with a check for exact report replay.
        """
        conn = self.connect()
        owns_transaction = not conn.in_transaction
        if owns_transaction:
            conn.execute("BEGIN")
        try:
            rows = conn.execute(
                "SELECT r.record_json FROM records AS r "
                "JOIN canary_cycles AS c ON c.cycle_id = r.cycle_id "
                "WHERE r.source = 'canary' AND c.status = 'finalized' "
                "ORDER BY r.rowid"
            ).fetchall()
            offset_row = conn.execute(
                "SELECT value FROM store_metadata WHERE key = 'jsonl_indexed_offset'"
            ).fetchone()
            offset = int(offset_row[0]) if offset_row is not None else 0
            if owns_transaction:
                conn.commit()
        except BaseException:
            if owns_transaction and conn.in_transaction:
                conn.rollback()
            raise
        records: list[InteractionRecord] = []
        for row in rows:
            if row[0] is None:
                raise RuntimeError(
                    "SQLite record payload cache is incomplete; reopen or rebuild the index"
                )
            records.append(InteractionRecord.from_jsonl(str(row[0])))
        return records, offset

    def read_records_by_ids(self, record_ids: Collection[str]) -> list[InteractionRecord]:
        """Read a recorded check snapshot by immutable record ID."""
        ordered = list(dict.fromkeys(str(record_id) for record_id in record_ids))
        if not ordered:
            return []
        conn = self.connect()
        found: dict[str, InteractionRecord] = {}
        for start in range(0, len(ordered), 900):
            chunk = ordered[start : start + 900]
            marks = ", ".join("?" for _ in chunk)
            for record_id, payload in conn.execute(
                f"SELECT id, record_json FROM records WHERE id IN ({marks})",
                tuple(chunk),
            ):
                if payload is None:
                    raise RuntimeError(
                        "SQLite record payload cache is incomplete; reopen or rebuild the index"
                    )
                found[str(record_id)] = InteractionRecord.from_jsonl(str(payload))
        missing = [record_id for record_id in ordered if record_id not in found]
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"check snapshot references {len(missing)} missing record(s): {preview}"
            )
        return [found[record_id] for record_id in ordered]

    def finalized_cycle_ids(self) -> list[str]:
        """Return finalized canary cycles in first-record order."""
        rows = self.connect().execute(
            "SELECT c.cycle_id FROM canary_cycles AS c "
            "JOIN records AS r ON r.cycle_id = c.cycle_id AND r.source = 'canary' "
            "WHERE c.status = 'finalized' GROUP BY c.cycle_id ORDER BY MIN(r.rowid)"
        )
        return [str(row[0]) for row in rows]

    def cycle_status(self, cycle_id: str) -> str | None:
        """Return ``open``/``finalized`` for a canary cycle, if known."""
        row = (
            self.connect()
            .execute("SELECT status FROM canary_cycles WHERE cycle_id = ?", (cycle_id,))
            .fetchone()
        )
        return str(row[0]) if row is not None else None

    def count_records(self) -> int:
        """Return the number of indexed records."""
        row = self.connect().execute("SELECT COUNT(*) FROM records").fetchone()
        return int(row[0])

    def config_events(
        self, record_ids: Collection[str] | None = None
    ) -> list[tuple[str, str | None, str]]:
        """Return config events, optionally limited to a record snapshot.

        Args:
            record_ids: When provided, retain only events whose first record
                belongs to this set. This keeps a report for an earlier check
                from acquiring config changes appended after that check.
        """
        rows = self.connect().execute(
            "SELECT ts, old_fingerprint, new_fingerprint, first_record_id "
            "FROM config_events ORDER BY id"
        )
        allowed = set(record_ids) if record_ids is not None else None
        return [
            (str(row[0]), str(row[1]) if row[1] is not None else None, str(row[2]))
            for row in rows
            if allowed is None or str(row[3]) in allowed
        ]

    # -- maintenance ---------------------------------------------------------

    def rebuild_index(self) -> int:
        """Rebuild the SQLite index from the JSONL log.

        Returns:
            The number of records indexed.
        """
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            records = self._read_stable_snapshot()
            lifecycle_rows = conn.execute(
                "SELECT cycle_id, status, expected_records, finalized_records, "
                "opened_ts, finalized_ts, finalized_offset FROM canary_cycles"
            ).fetchall()
            unique: dict[str, InteractionRecord] = {}
            digests: dict[str, str] = {}
            for record in records:
                digest = self._record_digest(record)
                previous = digests.get(record.id)
                if previous is not None and previous != digest:
                    raise RuntimeError(
                        f"records.jsonl contains conflicting payloads for id {record.id!r}"
                    )
                digests.setdefault(record.id, digest)
                unique.setdefault(record.id, record)
            conn.execute("DELETE FROM records")
            conn.execute("DELETE FROM config_events")
            conn.execute("DELETE FROM canary_cycles")
            for record in unique.values():
                self._index_record(conn, record)
            self._set_indexed_offset(conn, self.records_path.stat().st_size)
            conn.executemany(
                "INSERT INTO canary_cycles "
                "(cycle_id, status, expected_records, finalized_records, opened_ts, "
                "finalized_ts, finalized_offset) VALUES (?, ?, ?, ?, ?, ?, ?)",
                lifecycle_rows,
            )
            # External complete JSONL lines discovered during a rebuild have
            # no completion proof. Register them OPEN; never turn persistence
            # alone into an inference-ready lifecycle claim.
            self._sync_cycle_registry(conn, legacy_finalize=False)
            for cycle_id, status, _expected, finalized_records, *_rest in lifecycle_rows:
                if str(status) != "finalized":
                    continue
                count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM records WHERE source = 'canary' AND cycle_id = ?",
                        (str(cycle_id),),
                    ).fetchone()[0]
                )
                if finalized_records is None or count != int(finalized_records):
                    raise RuntimeError(
                        f"finalized cycle {cycle_id!r} has {count} logged records, "
                        f"expected immutable count {finalized_records}"
                    )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            self._harden_sqlite_files()
        return len(unique)
