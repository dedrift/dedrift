"""Storage layer: append-only records plus authoritative operational state.

Layout under the project root::

    .dedrift/
        config.toml        # project config (created by `dedrift init`)
        logs/records.jsonl # durable raw interaction log
        index.db           # authoritative lifecycle, checks, and anytime state

Raw records can be re-indexed from JSONL, but finalization, checks, alerts,
and e-process state cannot be reconstructed from record payloads alone.
Production backup and restore must therefore treat the entire ``.dedrift``
directory as one unit. ``Store.rebuild_index`` repairs record indexing while
preserving operational rows from the existing database; it is not disaster
recovery for a deleted database.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import get_ident
from types import TracebackType
from typing import BinaryIO, TextIO

from dedrift.schema import SQLITE_INT64_MAX, InteractionRecord

PROJECT_DIR = ".dedrift"
LOGS_DIR = "logs"
RECORDS_FILE = "records.jsonl"
INDEX_FILE = "index.db"
CONFIG_FILE = "config.toml"

_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_SQLITE_BUSY_TIMEOUT_MS = 30_000
_FINALIZED_REVISION_KEY = "finalized_revision"
_STORE_IDENTITY_KEY = "store_identity"
_STORE_IDENTITY_VALUE = "dedrift-index-v1"
_PINNED_EMBEDDER_KEY = "pinned_embedder"
_SQLITE_APPLICATION_ID = 0x44445246  # ASCII "DDRF"


class FinalizedSnapshotChangedError(RuntimeError):
    """Raised when finalized lifecycle state changed after a snapshot read."""


DEFAULT_CONFIG_TOML = """\
# dedrift project configuration
[project]
name = "default"
canary_repetitions = 7      # N repeated runs per canary per cycle
rolling_window_cycles = 5   # K cycles for the rolling reference
deterministic = false       # true only if the agent is exactly reproducible
monitor_latency = true       # false when in-process host jitter is not product behavior
custom_scalars = []         # numeric channels read from output.structured

[detection]
fdr_q = 0.05                # Benjamini-Hochberg FDR level
permutations = 500          # permutation-test resamples (seeded)
seed = 1729                 # global seed recorded in every report
ph_lambda = 12.0            # Page-Hinkley alarm threshold (reference-SD units).
ph_delta = 0.3              # Idealized null crossing bound 2*exp(-2*delta*lambda)
                            # ~= 0.15% idealized; measured ~8.5%/stream (estimated centering
                            # and scale) - see the PH calibration test.
inference = "fixed"         # "fixed" or opt-in "anytime"
cycle_effect = "off"        # "off" (default) | "auto" cluster-aware battery
cycle_effect_icc = 0.02     # ICC threshold for engaging the correction
alert_persistence = 1       # distinct finalized cycles required to fire

# Fixed-sample materiality gates, per signature channel. Fixed-sample alerts
# require BOTH FDR-adjusted significance and an effect exceeding these bands.
# Anytime tilts target power but do not apply this observed-effect gate.
[materiality]
refusal_rate_pp = 2.0       # percentage-point shift
format_validity_pp = 1.0    # percentage-point shift
rate_default_pp = 2.0       # other rate signatures, percentage points
scalar_cohen_d = 0.5        # reserved compatibility key; not an alert gate
ks_distance = 0.15          # KS statistic D floor (sup-norm CDF distance):
                            # KS catches shape changes with equal means,
                            # which Cohen's d would wrongly gate out
dispersion_ratio = 1.5      # robust dispersion (MAD) must grow/shrink by this
p95_relative = 0.10         # relative P95 shift floor
embedding_mmd2_floor = -1.0 # -1 = auto-calibrate; 0 = off; >0 = explicit

# Native-unit gate for noisy scalar channels. Every primary on the channel
# must also exceed this Wasserstein-1 distance before it can alert.
# [materiality.per_channel.latency_ms]
# absolute_distance = 5.0    # milliseconds

[embeddings]
# Pinned forever per project once set; dedrift refuses cross-embedder comparisons.
model = ""

[anytime]
alpha = 0.05
gamma_total = 0.02
tilts = [1.5, 2.0, 3.0]
epoch_allocation = "per_epoch"
rate_model = "frozen_cp"    # only production-safe rate construction
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
    open_seq INTEGER NOT NULL,
    prospective INTEGER NOT NULL DEFAULT 0 CHECK (prospective IN (0, 1)),
    expected_records INTEGER,
    finalized_records INTEGER,
    opened_ts TEXT NOT NULL,
    finalized_ts TEXT,
    finalized_offset INTEGER,
    finalize_seq INTEGER
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
    start_after_open_seq INTEGER NOT NULL,
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


@contextmanager
def _serialized_project_init(project_dir: Path) -> Iterator[None]:
    """Serialize project initialization across processes.

    A tiny SQLite database is used as the lock because SQLite's exclusive
    transaction is portable and crash-releasing.  Keeping the lock for the
    whole publication sequence prevents a second initializer from observing
    a config file while its contents (or the other authoritative artifacts)
    are still being created.
    """
    project_dir.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
    _harden_permissions(project_dir, _PRIVATE_DIR_MODE)
    lock_path = project_dir / ".init.lock"
    lock = sqlite3.connect(lock_path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000)
    try:
        lock.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
        lock.execute("BEGIN EXCLUSIVE")
        yield
        lock.commit()
    except BaseException:
        if lock.in_transaction:
            lock.rollback()
        raise
    finally:
        lock.close()
        _harden_permissions(lock_path, _PRIVATE_FILE_MODE)


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

    A connected ``Store`` belongs to the thread that opened it. Create one
    instance per worker thread; separate instances/processes are serialized by
    SQLite and the append protocol.
    """

    def __init__(self, root: Path | str = ".", *, require_project: bool = True) -> None:
        """Open a store rooted at ``root``.

        Args:
            root: Directory containing (or to contain) the ``.dedrift`` project.
            require_project: Fail here if the project does not exist yet.
                Opening an uninitialised directory used to succeed and then
                fail at the first *write*, by which time a full cycle of real
                work could already have been done and lost. Only
                :meth:`init_project` passes ``False``.

        Raises:
            FileNotFoundError: If ``require_project`` and no project is present.
        """
        self.root = Path(root)
        self.project_dir = self.root / PROJECT_DIR
        self.records_path = self.project_dir / LOGS_DIR / RECORDS_FILE
        self.index_path = self.project_dir / INDEX_FILE
        self.config_path = self.project_dir / CONFIG_FILE
        self._conn: sqlite3.Connection | None = None
        self._connection_thread_id: int | None = None
        self._index_creation_authorized = False
        if require_project and not (
            self.config_path.is_file()
            and self.records_path.is_file()
            and self.index_path.is_file()
            and self.index_path.stat().st_size > 0
        ):
            msg = (
                f"no complete dedrift project at {self.root}: expected config, raw log, "
                f"and authoritative index ({self.config_path}, {self.records_path}, "
                f"{self.index_path}). Do not recreate a missing index from JSONL: lifecycle "
                "and anytime state are not reconstructible; restore the whole .dedrift backup. "
                f"For a genuinely new project, run `dedrift init` or call "
                f"Store.init_project({str(self.root)!r})"
            )
            raise FileNotFoundError(msg)

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def init_project(cls, root: Path | str = ".") -> Store:
        """Create the ``.dedrift`` project structure (idempotent).

        Initialization is allowed only when all three authoritative artifacts
        already exist or all three are absent.  A partial project is damage,
        not a new project: recreating (especially) ``index.db`` would silently
        discard finalization and anytime state that JSONL cannot reconstruct.

        Args:
            root: Directory in which to create the project.

        Returns:
            An open store for the new project.
        """
        store = cls(root, require_project=False)
        with _serialized_project_init(store.project_dir):
            artifacts = (store.config_path, store.records_path, store.index_path)
            present = tuple(path.is_file() for path in artifacts)
            if any(present) and not all(present):
                found = ", ".join(
                    str(path) for path, exists in zip(artifacts, present, strict=True) if exists
                )
                missing = ", ".join(
                    str(path) for path, exists in zip(artifacts, present, strict=True) if not exists
                )
                raise FileNotFoundError(
                    "incomplete dedrift project; refusing to recreate authoritative state. "
                    f"Found: {found}. Missing: {missing}. Restore the whole .dedrift backup."
                )

            store.records_path.parent.mkdir(exist_ok=True, mode=_PRIVATE_DIR_MODE)
            _harden_permissions(store.records_path.parent, _PRIVATE_DIR_MODE)
            if not all(present):
                # Publish the complete config atomically; no cooperating reader
                # can observe a zero-byte or partially written configuration.
                with _atomic_private_writer(store.config_path) as stream:
                    stream.write(DEFAULT_CONFIG_TOML.encode("utf-8"))
                descriptor = os.open(
                    store.records_path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL,
                    _PRIVATE_FILE_MODE,
                )
                os.close(descriptor)
            _harden_permissions(store.config_path, _PRIVATE_FILE_MODE)
            _harden_permissions(store.records_path, _PRIVATE_FILE_MODE)
            store._index_creation_authorized = not all(present)
            try:
                store.connect()
            finally:
                store._index_creation_authorized = False
        return store

    def exists(self) -> bool:
        """Return True if this looks like an initialized project."""
        return (
            self.config_path.is_file()
            and self.records_path.is_file()
            and self.index_path.is_file()
            and self.index_path.stat().st_size > 0
        )

    def connect(self) -> sqlite3.Connection:
        """Open the SQLite index, migrate it, and reconcile a durable log tail.

        A complete JSONL line may survive a crash just before the matching
        SQLite commit. Reconciliation on *open* keeps reads, attribution, and
        counts consistent even when no later append happens to trigger repair.
        """
        thread_id = get_ident()
        if self._conn is not None and self._connection_thread_id != thread_id:
            raise RuntimeError(
                "a connected Store instance cannot be shared across threads; "
                "create one Store per worker thread"
            )
        if self._conn is None:
            if not self._index_creation_authorized and (
                not self.index_path.is_file() or self.index_path.stat().st_size == 0
            ):
                raise FileNotFoundError(
                    "authoritative index.db is missing or empty; refusing to reconstruct "
                    "lifecycle/check/anytime state from JSONL. Restore the whole "
                    ".dedrift backup."
                )
            self.index_path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
            _harden_permissions(self.index_path.parent, _PRIVATE_DIR_MODE)
            conn = sqlite3.connect(self.index_path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000)
            try:
                if not self._index_creation_authorized:
                    self._validate_existing_index_identity(conn)
                conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = FULL")
                # Schema discovery and every conditional ALTER must share one
                # writer reservation. Otherwise two first-openers can both
                # observe a missing legacy column and the loser fails with a
                # duplicate-column error after the winner commits.
                conn.execute("BEGIN IMMEDIATE")
                self._migrate_schema(conn)
                if self.records_path.exists():
                    # A pre-v0.3.1 index has no cached canonical payloads.
                    # Force one source-of-truth scan to backfill them before
                    # future checks switch to indexed window reads.
                    missing_payload = conn.execute(
                        "SELECT 1 FROM records "
                        "WHERE record_json IS NULL OR record_digest IS NULL LIMIT 1"
                    ).fetchone()
                    if missing_payload is not None:
                        conn.execute(
                            "DELETE FROM store_metadata WHERE key = 'jsonl_indexed_offset'"
                        )
                    else:
                        self._validate_indexed_jsonl_prefix(conn)
                    self._reconcile_jsonl_tail(conn)
                    self._sync_cycle_registry(
                        conn,
                        # Completion cannot be reconstructed from durable
                        # records alone. Upgraded histories fail closed as
                        # OPEN until the operator finalizes them explicitly.
                        legacy_finalize=False,
                    )
                conn.commit()
                self._conn = conn
                self._connection_thread_id = thread_id
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                conn.close()
                self._conn = None
                self._connection_thread_id = None
                raise
            finally:
                self._harden_sqlite_files()
        return self._conn

    @staticmethod
    def _execute_schema(conn: sqlite3.Connection) -> None:
        """Execute the static schema without ``executescript`` auto-commits."""
        statement = ""
        for line in _SCHEMA_SQL.splitlines(keepends=True):
            statement += line
            if sqlite3.complete_statement(statement):
                conn.execute(statement)
                statement = ""
        if statement.strip():
            raise RuntimeError("internal SQLite schema contains an incomplete statement")

    @staticmethod
    def _validate_existing_index_identity(conn: sqlite3.Connection) -> None:
        """Reject empty, unrelated, or identity-mismatched SQLite databases."""
        try:
            application_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if application_id not in {0, _SQLITE_APPLICATION_ID}:
                raise RuntimeError(
                    "index.db belongs to another SQLite application; refusing to modify it"
                )
            # Releases before the identity marker already had these two core
            # tables. Their presence is the only legacy migration exception.
            if not {"records", "config_events"}.issubset(tables):
                raise RuntimeError(
                    "index.db is not a recognized dedrift index; refusing to create or "
                    "migrate authoritative state in place"
                )
            if "store_metadata" in tables:
                marker = conn.execute(
                    "SELECT value FROM store_metadata WHERE key = ?",
                    (_STORE_IDENTITY_KEY,),
                ).fetchone()
                if marker is not None and str(marker[0]) != _STORE_IDENTITY_VALUE:
                    raise RuntimeError(
                        "index.db has an incompatible dedrift identity marker; "
                        "restore the matching project backup"
                    )
        except sqlite3.DatabaseError as exc:
            raise RuntimeError(
                "index.db is not a valid dedrift SQLite index; restore the whole backup"
            ) from exc

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Create and migrate the index while the caller holds a writer lock."""
        self._execute_schema(conn)
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(records)")}
        if "record_digest" not in columns:
            conn.execute("ALTER TABLE records ADD COLUMN record_digest TEXT")
        if "record_json" not in columns:
            conn.execute("ALTER TABLE records ADD COLUMN record_json TEXT")
        check_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(checks)")}
        if "snapshot_offset" not in check_columns:
            conn.execute("ALTER TABLE checks ADD COLUMN snapshot_offset INTEGER")
        if "snapshot_record_ids_json" not in check_columns:
            conn.execute("ALTER TABLE checks ADD COLUMN snapshot_record_ids_json TEXT")
        cycle_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(canary_cycles)")}
        if "open_seq" not in cycle_columns:
            conn.execute("ALTER TABLE canary_cycles ADD COLUMN open_seq INTEGER")
        if "finalize_seq" not in cycle_columns:
            conn.execute("ALTER TABLE canary_cycles ADD COLUMN finalize_seq INTEGER")
        if "prospective" not in cycle_columns:
            # Existing lifecycle rows were reconstructed at or after import;
            # there is no durable proof that they preceded observation. They
            # remain valid fixed-sample/history data, but fail closed for an
            # already-active anytime epoch.
            conn.execute(
                "ALTER TABLE canary_cycles ADD COLUMN prospective INTEGER "
                "NOT NULL DEFAULT 0 CHECK (prospective IN (0, 1))"
            )
        self._backfill_cycle_open_sequences(conn)
        self._backfill_finalized_revisions(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_canary_cycles_open_seq "
            "ON canary_cycles(open_seq) WHERE open_seq IS NOT NULL"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_canary_cycles_finalize_seq "
            "ON canary_cycles(finalize_seq) WHERE finalize_seq IS NOT NULL"
        )
        duplicate_coordinate = conn.execute(
            "SELECT cycle_id, canary_id, repetition, COUNT(*) FROM records "
            "WHERE source = 'canary' AND cycle_id IS NOT NULL "
            "GROUP BY cycle_id, canary_id, repetition HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate_coordinate is not None:
            raise RuntimeError(
                "legacy index contains duplicate canary coordinates "
                f"{duplicate_coordinate[0]!r}/{duplicate_coordinate[1]!r}#"
                f"{duplicate_coordinate[2]!r}; restore or repair the project before opening"
            )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_records_canary_coordinate "
            "ON records(cycle_id, canary_id, repetition) "
            "WHERE source = 'canary' AND cycle_id IS NOT NULL"
        )
        epoch_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(anytime_epochs)")}
        if "start_after_open_seq" not in epoch_columns:
            conn.execute(
                "ALTER TABLE anytime_epochs ADD COLUMN "
                "start_after_open_seq INTEGER NOT NULL DEFAULT 0"
            )
            self._backfill_epoch_open_boundaries(conn)
        conn.execute(
            "INSERT INTO store_metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_STORE_IDENTITY_KEY, _STORE_IDENTITY_VALUE),
        )
        # The pin file is convenient for humans, but SQLite is authoritative
        # lifecycle state. Backfill legacy pins once, then fail closed if a
        # partial restore/delete leaves the two copies inconsistent.
        embedder_path = self.project_dir / "embedder.json"
        pinned_row = conn.execute(
            "SELECT value FROM store_metadata WHERE key = ?", (_PINNED_EMBEDDER_KEY,)
        ).fetchone()
        pinned_metadata = str(pinned_row[0]) if pinned_row is not None else None
        pinned_file: str | None = None
        if embedder_path.exists():
            try:
                raw_pin = json.loads(embedder_path.read_text(encoding="utf-8"))
                value = raw_pin["embedder"]
                if not isinstance(value, str) or not value:
                    raise ValueError("embedder must be a non-empty string")
                pinned_file = value
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "embedder.json is invalid; restore the complete .dedrift snapshot"
                ) from exc
        if pinned_metadata is not None and pinned_file is None:
            raise RuntimeError(
                "the authoritative embedder pin exists but embedder.json is missing; "
                "restore the complete .dedrift snapshot"
            )
        if pinned_metadata is not None and pinned_file != pinned_metadata:
            raise RuntimeError(
                "embedder.json conflicts with the authoritative embedder pin; "
                "restore the complete .dedrift snapshot"
            )
        if pinned_metadata is None and pinned_file is not None:
            conn.execute(
                "INSERT INTO store_metadata (key, value) VALUES (?, ?)",
                (_PINNED_EMBEDDER_KEY, pinned_file),
            )
        conn.execute(f"PRAGMA application_id = {_SQLITE_APPLICATION_ID}")

    @staticmethod
    def _backfill_cycle_open_sequences(conn: sqlite3.Connection) -> None:
        """Give legacy cycle rows a durable first-observation order."""
        row = conn.execute("SELECT COALESCE(MAX(open_seq), 0) FROM canary_cycles").fetchone()
        next_seq = int(row[0]) + 1
        missing = conn.execute(
            "SELECT cycle_id FROM canary_cycles WHERE open_seq IS NULL ORDER BY rowid"
        ).fetchall()
        for (cycle_id,) in missing:
            conn.execute(
                "UPDATE canary_cycles SET open_seq = ? WHERE cycle_id = ?",
                (next_seq, str(cycle_id)),
            )
            next_seq += 1

    @staticmethod
    def _next_cycle_open_seq(conn: sqlite3.Connection) -> int:
        """Return the next durable first-observation sequence."""
        row = conn.execute("SELECT COALESCE(MAX(open_seq), 0) FROM canary_cycles").fetchone()
        return int(row[0]) + 1

    @staticmethod
    def _backfill_epoch_open_boundaries(conn: sqlite3.Connection) -> None:
        """Translate legacy cycle-name epoch boundaries to observation order."""
        rows = conn.execute("SELECT fingerprint, start_after_cycle FROM anytime_epochs").fetchall()
        for fingerprint, cycle_id in rows:
            boundary = conn.execute(
                "SELECT open_seq FROM canary_cycles WHERE cycle_id = ?",
                (str(cycle_id),),
            ).fetchone()
            if boundary is None or boundary[0] is None:
                raise RuntimeError(
                    "cannot migrate anytime epoch boundary because cycle "
                    f"{cycle_id!r} is absent from lifecycle state; restore the full backup"
                )
            conn.execute(
                "UPDATE anytime_epochs SET start_after_open_seq = ? WHERE fingerprint = ?",
                (int(boundary[0]), str(fingerprint)),
            )

    @staticmethod
    def _current_finalized_revision(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT value FROM store_metadata WHERE key = ?", (_FINALIZED_REVISION_KEY,)
        ).fetchone()
        if row is None:
            return 0
        try:
            revision = int(row[0])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("stored finalized revision is not an integer") from exc
        if revision < 0:
            raise RuntimeError("stored finalized revision must be non-negative")
        return revision

    @staticmethod
    def _set_finalized_revision(conn: sqlite3.Connection, revision: int) -> None:
        conn.execute(
            "INSERT INTO store_metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_FINALIZED_REVISION_KEY, str(revision)),
        )

    def _backfill_finalized_revisions(self, conn: sqlite3.Connection) -> None:
        """Give legacy finalized cycles their prior deterministic read order."""
        max_row = conn.execute(
            "SELECT COALESCE(MAX(finalize_seq), 0) FROM canary_cycles"
        ).fetchone()
        revision = max(self._current_finalized_revision(conn), int(max_row[0]))
        rows = conn.execute(
            "SELECT c.cycle_id FROM canary_cycles AS c "
            "LEFT JOIN records AS r ON r.cycle_id = c.cycle_id AND r.source = 'canary' "
            "WHERE c.status = 'finalized' AND c.finalize_seq IS NULL "
            "GROUP BY c.cycle_id "
            "ORDER BY CASE WHEN MIN(r.rowid) IS NULL THEN 1 ELSE 0 END, "
            "MIN(r.rowid), c.finalized_ts, c.rowid"
        ).fetchall()
        for row in rows:
            revision += 1
            conn.execute(
                "UPDATE canary_cycles SET finalize_seq = ? WHERE cycle_id = ?",
                (revision, str(row[0])),
            )
        self._set_finalized_revision(conn, revision)

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
            if self._connection_thread_id != get_ident():
                raise RuntimeError("a connected Store instance must be closed by its owner thread")
            self._conn.close()
            self._conn = None
            self._connection_thread_id = None

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

    def begin_cycle(self, cycle_id: str, *, expected_records: int | None = None) -> None:
        """Durably declare a prospective canary cycle before observing it.

        This is the lifecycle boundary used by anytime-valid inference. A
        cycle first discovered while records are imported is deliberately
        marked non-prospective and cannot later be upgraded: by then its
        outcomes may already be known. Complete-batch imports remain valid
        for fixed checks and frozen golden/reference data.

        Args:
            cycle_id: Stable non-empty identifier for the upcoming cycle.
            expected_records: Optional exact number of records to collect.
        """
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ValueError("cycle_id must be a non-empty string")
        if (
            expected_records is not None
            and (
                isinstance(expected_records, bool)
                or not isinstance(expected_records, int)
                or not 1 <= expected_records <= SQLITE_INT64_MAX
            )
        ):
            raise ValueError(
                "expected_records must be a strict integer between 1 and "
                f"{SQLITE_INT64_MAX}"
            )
        conn = self.connect()
        if conn.in_transaction:
            raise RuntimeError("begin_cycle requires ownership of the SQLite transaction")
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._reconcile_jsonl_tail(conn)
            self._sync_cycle_registry(conn, legacy_finalize=False)
            row = conn.execute(
                "SELECT status, prospective, expected_records FROM canary_cycles "
                "WHERE cycle_id = ?",
                (cycle_id,),
            ).fetchone()
            if row is not None:
                status = str(row[0])
                prospective = bool(row[1])
                declared = int(row[2]) if row[2] is not None else None
                if not prospective:
                    raise ValueError(
                        f"cycle {cycle_id!r} was already observed or imported without a "
                        "pre-observation marker; it cannot be made prospective"
                    )
                if status != "open":
                    raise ValueError(f"cycle {cycle_id!r} is already finalized")
                if (
                    declared is not None
                    and expected_records is not None
                    and declared != expected_records
                ):
                    raise ValueError(
                        f"cycle {cycle_id!r} expected count is already {declared}, "
                        f"not {expected_records}"
                    )
                if declared is None and expected_records is not None:
                    conn.execute(
                        "UPDATE canary_cycles SET expected_records = ? WHERE cycle_id = ?",
                        (expected_records, cycle_id),
                    )
            else:
                conn.execute(
                    "INSERT INTO canary_cycles "
                    "(cycle_id, status, open_seq, prospective, expected_records, opened_ts) "
                    "VALUES (?, 'open', ?, 1, ?, ?)",
                    (
                        cycle_id,
                        self._next_cycle_open_seq(conn),
                        expected_records,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            self._harden_sqlite_files()

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
        """Durably append a batch, then update the operational database.

        The SQLite ``BEGIN IMMEDIATE`` lock serializes writers before they
        touch JSONL. JSONL is flushed and fsynced first because it is the
        source of truth for raw interactions; SQLite commits only after that
        durability boundary. A failure between those steps can leave the index
        behind the log, but never the reverse. The next append reconciles a
        complete, fsynced tail before applying an idempotent retry;
        ``rebuild_index`` remains the explicit record-index repair path.

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
        if any(
            not isinstance(cycle_id, str)
            or not cycle_id
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= SQLITE_INT64_MAX
            for cycle_id, count in expected.items()
        ):
            raise ValueError(
                "expected cycle counts require non-empty string IDs and strict integers "
                f"between 1 and {SQLITE_INT64_MAX}"
            )
        # Within a complete batch, open/finalize cycles in first-appearance
        # order, never lexical ID order. Cycle IDs are opaque ("cycle-10"
        # need not sort after "cycle-9"). Cross-batch first-observation order
        # is recorded separately in ``open_seq``.
        batch_cycle_ids = list(
            dict.fromkeys(
                str(record.cycle_id)
                for record, _ in unique.values()
                if record.source.value == "canary" and record.cycle_id is not None
            )
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
            self._sync_cycle_registry(conn, legacy_finalize=False)
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
            pending_coordinates: set[tuple[str, str, int]] = set()
            for record, _ in pending:
                if record.source.value != "canary" or record.cycle_id is None:
                    continue
                assert record.canary_id is not None and record.repetition is not None
                coordinate = (record.cycle_id, record.canary_id, record.repetition)
                if coordinate in pending_coordinates:
                    raise ValueError(
                        "batch contains duplicate canary coordinates "
                        f"(cycle_id, canary_id, repetition)={coordinate!r}"
                    )
                pending_coordinates.add(coordinate)
                existing_coordinate = conn.execute(
                    "SELECT id FROM records WHERE source = 'canary' AND cycle_id = ? "
                    "AND canary_id = ? AND repetition = ? LIMIT 1",
                    coordinate,
                ).fetchone()
                if existing_coordinate is not None:
                    raise ValueError(
                        "canary coordinate already exists under a different record id: "
                        f"{coordinate!r}"
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
                if finalize_cycles:
                    coordinates = conn.execute(
                        "SELECT canary_id, repetition FROM records "
                        "WHERE source = 'canary' AND cycle_id = ?",
                        (cycle_id,),
                    ).fetchall()
                    coordinates.extend(
                        (record.canary_id, record.repetition)
                        for record, _ in pending
                        if record.source.value == "canary" and record.cycle_id == cycle_id
                    )
                    self._validate_cycle_coordinates(
                        cycle_id,
                        coordinates,
                        expected_records=limit,
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
                open_seq = self._next_cycle_open_seq(conn)
                conn.execute(
                    "INSERT INTO canary_cycles "
                    "(cycle_id, status, open_seq, prospective, expected_records, opened_ts) "
                    "VALUES (?, 'open', ?, 0, ?, ?) "
                    "ON CONFLICT(cycle_id) DO UPDATE SET expected_records = "
                    "COALESCE(canary_cycles.expected_records, excluded.expected_records)",
                    (cycle_id, open_seq, expected.get(cycle_id), opened_ts),
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
            open_seq = self._next_cycle_open_seq(conn)
            if legacy_finalize:
                finalize_seq = self._current_finalized_revision(conn) + 1
                self._set_finalized_revision(conn, finalize_seq)
                conn.execute(
                    "INSERT INTO canary_cycles "
                    "(cycle_id, status, open_seq, prospective, expected_records, "
                    "finalized_records, "
                    "opened_ts, finalized_ts, finalized_offset, finalize_seq) "
                    "VALUES (?, 'finalized', ?, 0, ?, ?, ?, ?, ?, ?)",
                    (
                        str(cycle_id),
                        open_seq,
                        int(count),
                        int(count),
                        str(first_ts),
                        now,
                        offset,
                        finalize_seq,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO canary_cycles "
                    "(cycle_id, status, open_seq, prospective, opened_ts) "
                    "VALUES (?, 'open', ?, 0, ?)",
                    (str(cycle_id), open_seq, str(first_ts)),
                )

    @staticmethod
    def _validate_cycle_coordinates(
        cycle_id: str,
        coordinates: Collection[tuple[object, object]],
        *,
        expected_records: int | None,
    ) -> int:
        """Validate a complete cycle without mutating durable storage."""
        rows = list(coordinates)
        count = len(rows)
        if count == 0:
            raise ValueError(f"cannot finalize empty canary cycle {cycle_id!r}")
        by_canary: dict[str, list[int]] = {}
        for canary_id, repetition in rows:
            if canary_id is None or not isinstance(repetition, int) or isinstance(repetition, bool):
                raise ValueError(
                    f"cannot finalize cycle {cycle_id!r}: every canary record requires "
                    "canary_id and an integer repetition"
                )
            by_canary.setdefault(str(canary_id), []).append(repetition)
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
        repetition_designs = {tuple(sorted(repetitions)) for repetitions in by_canary.values()}
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
        if expected_records is not None and count != expected_records:
            raise ValueError(
                f"cannot finalize cycle {cycle_id!r}: observed {count} records, "
                f"expected {expected_records}"
            )
        return count

    def _finalize_cycle_locked(
        self,
        conn: sqlite3.Connection,
        cycle_id: str,
        *,
        expected_records: int | None,
        finalized_offset: int,
    ) -> None:
        """Finalize one cycle inside the caller's writer transaction."""
        if (
            expected_records is not None
            and (
                isinstance(expected_records, bool)
                or not isinstance(expected_records, int)
                or not 1 <= expected_records <= SQLITE_INT64_MAX
            )
        ):
            raise ValueError(
                "expected_records must be a strict integer between 1 and "
                f"{SQLITE_INT64_MAX}"
            )
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
        if declared is not None and expected_records is not None and declared != expected_records:
            raise ValueError(
                f"cycle {cycle_id!r} expected count is already {declared}, not {expected_records}"
            )
        expected = declared if declared is not None else expected_records
        coordinates = conn.execute(
            "SELECT canary_id, repetition FROM records WHERE source = 'canary' AND cycle_id = ?",
            (cycle_id,),
        ).fetchall()
        count = self._validate_cycle_coordinates(
            cycle_id,
            coordinates,
            expected_records=expected,
        )
        if status == "finalized":
            if previous_count != count:
                raise RuntimeError(
                    f"finalized cycle {cycle_id!r} changed from {previous_count} to {count} records"
                )
            return
        now = datetime.now(timezone.utc).isoformat()
        finalize_seq = self._current_finalized_revision(conn) + 1
        self._set_finalized_revision(conn, finalize_seq)
        conn.execute(
            "UPDATE canary_cycles SET status = 'finalized', "
            "expected_records = ?, finalized_records = ?, finalized_ts = ?, "
            "finalized_offset = ?, finalize_seq = ? WHERE cycle_id = ?",
            (
                expected if expected is not None else count,
                count,
                now,
                finalized_offset,
                finalize_seq,
                cycle_id,
            ),
        )

    def finalize_cycle(self, cycle_id: str, *, expected_records: int | None = None) -> None:
        """Mark a streamed canary cycle complete after validating its size."""
        if not cycle_id:
            raise ValueError("cycle_id must be non-empty")
        if (
            expected_records is not None
            and (
                isinstance(expected_records, bool)
                or not isinstance(expected_records, int)
                or not 1 <= expected_records <= SQLITE_INT64_MAX
            )
        ):
            raise ValueError(
                "expected_records must be a strict integer between 1 and "
                f"{SQLITE_INT64_MAX}"
            )
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
        legacy_full_scan = row is None
        offset = int(row[0]) if row is not None else 0
        size = self.records_path.stat().st_size
        if not 0 <= offset <= size:
            raise RuntimeError(
                "records.jsonl is shorter than its indexed checkpoint; "
                "restore the append-only log before rebuilding the index"
            )
        with self.records_path.open("rb") as stream:
            stream.seek(offset)
            seen_in_scan: set[str] = set()
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
                if record.id in seen_in_scan:
                    raise RuntimeError(
                        f"records.jsonl tail repeats record id {record.id!r}; "
                        "restore the append-only log before retrying"
                    )
                seen_in_scan.add(record.id)
                digest = self._record_digest(record)
                existing = conn.execute(
                    "SELECT record_digest, record_json FROM records WHERE id = ?",
                    (record.id,),
                ).fetchone()
                if existing is not None and not legacy_full_scan:
                    raise RuntimeError(
                        f"records.jsonl tail repeats indexed record id {record.id!r}; "
                        "restore the append-only log before retrying"
                    )
                if existing is None:
                    if record.source.value == "canary" and record.cycle_id is not None:
                        lifecycle = conn.execute(
                            "SELECT status FROM canary_cycles WHERE cycle_id = ?",
                            (record.cycle_id,),
                        ).fetchone()
                        processed = conn.execute(
                            "SELECT 1 FROM anytime_processed_cycles WHERE cycle_id = ? LIMIT 1",
                            (record.cycle_id,),
                        ).fetchone()
                        if lifecycle is not None and str(lifecycle[0]) == "finalized":
                            raise RuntimeError(
                                "records.jsonl externally extends finalized canary cycle "
                                f"{record.cycle_id!r}; restore the immutable log before retrying"
                            )
                        if processed is not None:
                            raise RuntimeError(
                                "records.jsonl externally extends anytime-processed canary "
                                f"cycle {record.cycle_id!r}; restore the immutable log "
                                "before retrying"
                            )
                    self._index_record(conn, record)
                elif existing[0] is None or existing[1] is None:
                    stored_digest = str(existing[0]) if existing[0] is not None else None
                    if stored_digest is not None and stored_digest != digest:
                        raise RuntimeError(
                            f"records.jsonl contains conflicting payloads for id {record.id!r}"
                        )
                    if existing[1] is not None:
                        try:
                            cached = InteractionRecord.from_jsonl(str(existing[1]))
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(
                                f"indexed payload is invalid for record {record.id!r}"
                            ) from exc
                        cached_digest = self._record_digest(cached)
                        if cached.id != record.id or cached_digest != digest:
                            raise RuntimeError(
                                "records.jsonl conflicts with the surviving indexed payload "
                                f"for id {record.id!r}"
                            )
                    conn.execute(
                        "UPDATE records SET record_digest = COALESCE(record_digest, ?), "
                        "record_json = COALESCE(record_json, ?) WHERE id = ?",
                        (digest, record.to_jsonl(), record.id),
                    )
                elif str(existing[0]) != digest:
                    raise RuntimeError(
                        f"records.jsonl contains conflicting payloads for id {record.id!r}"
                    )
            self._assert_finalized_counts(conn)
            self._set_indexed_offset(conn, stream.tell())

    def _validate_indexed_jsonl_prefix(self, conn: sqlite3.Connection) -> None:
        """Verify the immutable indexed prefix against SQLite on process open.

        Tail-only reconciliation cannot detect an equal-length edit before the
        byte checkpoint.  Such an edit previously allowed raw-log reads and
        finalized-snapshot reads to describe different histories.  Validate
        every indexed line once when a Store connection is established; later
        writes remain serialized by the same SQLite writer lock.

        This is deliberately fail-closed and O(indexed log size) per process
        open.  A future chunked hash manifest can make the same guarantee
        incrementally without weakening it.
        """
        row = conn.execute(
            "SELECT value FROM store_metadata WHERE key = 'jsonl_indexed_offset'"
        ).fetchone()
        if row is None:
            return
        offset = int(row[0])
        size = self.records_path.stat().st_size
        if not 0 <= offset <= size:
            raise RuntimeError(
                "records.jsonl is shorter than its indexed checkpoint; restore the "
                "complete .dedrift snapshot"
            )
        indexed = [
            (str(record_id), str(digest) if digest is not None else None, payload)
            for record_id, digest, payload in conn.execute(
                "SELECT id, record_digest, record_json FROM records ORDER BY rowid"
            )
        ]
        seen: set[str] = set()
        position = 0
        with self.records_path.open("rb") as stream:
            while stream.tell() < offset:
                raw_line = stream.readline()
                if not raw_line or stream.tell() > offset or not raw_line.endswith(b"\n"):
                    raise RuntimeError(
                        "indexed JSONL checkpoint is not on a complete record boundary; "
                        "restore the complete .dedrift snapshot"
                    )
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    record = InteractionRecord.from_jsonl(stripped.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as exc:
                    raise RuntimeError(
                        "records.jsonl indexed prefix contains an invalid record; "
                        "restore the complete .dedrift snapshot"
                    ) from exc
                if record.id in seen:
                    raise RuntimeError(
                        f"records.jsonl indexed prefix repeats record id {record.id!r}; "
                        "restore the complete .dedrift snapshot"
                    )
                if position >= len(indexed):
                    raise RuntimeError(
                        f"records.jsonl indexed prefix contains unknown id {record.id!r}; "
                        "restore the complete .dedrift snapshot"
                    )
                expected_id, stored_digest, payload = indexed[position]
                if record.id != expected_id:
                    raise RuntimeError(
                        "records.jsonl indexed prefix order conflicts with SQLite: "
                        f"expected {expected_id!r}, found {record.id!r}; restore the "
                        "complete .dedrift snapshot"
                    )
                digest = self._record_digest(record)
                if stored_digest is None or stored_digest != digest or payload is None:
                    raise RuntimeError(
                        f"records.jsonl indexed prefix conflicts with record {record.id!r}; "
                        "restore the complete .dedrift snapshot"
                    )
                try:
                    cached = InteractionRecord.from_jsonl(str(payload))
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"indexed payload is invalid for record {record.id!r}"
                    ) from exc
                if cached.id != record.id or self._record_digest(cached) != digest:
                    raise RuntimeError(
                        f"records.jsonl conflicts with cached payload for record {record.id!r}; "
                        "restore the complete .dedrift snapshot"
                    )
                seen.add(record.id)
                position += 1
            if stream.tell() != offset:
                raise RuntimeError(
                    "indexed JSONL checkpoint is inconsistent; restore the complete "
                    ".dedrift snapshot"
                )
        if position != len(indexed):
            missing = [record_id for record_id, _digest, _payload in indexed[position:]]
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                "SQLite indexes records absent from the indexed JSONL prefix "
                f"({preview}); restore the complete .dedrift snapshot"
            )

    @staticmethod
    def _assert_finalized_counts(conn: sqlite3.Connection) -> None:
        """Fail closed if lifecycle metadata and indexed records diverge."""
        bad = conn.execute(
            "SELECT c.cycle_id, c.finalized_records, COUNT(r.id) "
            "FROM canary_cycles AS c "
            "LEFT JOIN records AS r ON r.cycle_id = c.cycle_id AND r.source = 'canary' "
            "WHERE c.status = 'finalized' "
            "GROUP BY c.cycle_id "
            "HAVING c.finalized_records IS NULL OR COUNT(r.id) != c.finalized_records "
            "LIMIT 1"
        ).fetchone()
        if bad is not None:
            raise RuntimeError(
                f"finalized cycle {bad[0]!r} has {int(bad[2])} indexed records, "
                f"but lifecycle metadata records {bad[1]!r}; restore the complete "
                "project snapshot before retrying"
            )

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
        excludes open cycles by construction. Callers that need exact replay
        should use the revision-bearing variant and persist its compact
        finalized revision.
        """
        records, offset, _revision = self.read_finalized_canary_snapshot_with_revision()
        return records, offset

    def read_finalized_canary_snapshot_with_revision(
        self,
    ) -> tuple[list[InteractionRecord], int, int]:
        """Read finalized records, log offset, and lifecycle revision atomically.

        The revision increments exactly once whenever an open cycle first
        becomes finalized. A caller that performs expensive work after this
        read can acquire ``BEGIN IMMEDIATE`` and call
        :meth:`assert_finalized_revision` before committing derived state.
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
                "ORDER BY c.finalize_seq, r.rowid"
            ).fetchall()
            offset_row = conn.execute(
                "SELECT value FROM store_metadata WHERE key = 'jsonl_indexed_offset'"
            ).fetchone()
            offset = int(offset_row[0]) if offset_row is not None else 0
            revision = self._current_finalized_revision(conn)
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
        return records, offset, revision

    def read_finalized_canary_snapshot_at_revision(
        self, finalized_revision: int
    ) -> list[InteractionRecord]:
        """Reconstruct an exact historical snapshot from durable cycle order.

        Snapshot manifests no longer need to duplicate every record ID in each
        check row. Finalized cycles are immutable and receive a monotone
        ``finalize_seq``, so ``finalize_seq <= finalized_revision`` identifies
        the same record set in O(result size).
        """
        if finalized_revision < 0:
            raise ValueError("finalized revision must be non-negative")
        conn = self.connect()
        current = self._current_finalized_revision(conn)
        if finalized_revision > current:
            raise ValueError(
                f"snapshot revision {finalized_revision} is newer than current revision {current}"
            )
        rows = conn.execute(
            "SELECT r.record_json FROM records AS r "
            "JOIN canary_cycles AS c ON c.cycle_id = r.cycle_id "
            "WHERE r.source = 'canary' AND c.status = 'finalized' "
            "AND c.finalize_seq <= ? ORDER BY c.finalize_seq, r.rowid",
            (finalized_revision,),
        ).fetchall()
        records: list[InteractionRecord] = []
        for row in rows:
            if row[0] is None:
                raise RuntimeError(
                    "SQLite record payload cache is incomplete; reopen or rebuild the index"
                )
            records.append(InteractionRecord.from_jsonl(str(row[0])))
        return records

    def assert_finalized_revision(self, expected_revision: int) -> None:
        """Assert a snapshot revision while the caller owns a transaction.

        Callers should first execute ``BEGIN IMMEDIATE`` so the successful
        assertion remains true through their derived-state commit.
        """
        if expected_revision < 0:
            raise ValueError("expected finalized revision must be non-negative")
        conn = self.connect()
        if not conn.in_transaction:
            raise RuntimeError("assert_finalized_revision requires an existing writer transaction")
        actual_revision = self._current_finalized_revision(conn)
        if actual_revision != expected_revision:
            raise FinalizedSnapshotChangedError(
                "finalized canary snapshot changed during processing: "
                f"expected revision {expected_revision}, found {actual_revision}; retry"
            )

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
        """Return finalized canary cycles in finalization order."""
        rows = self.connect().execute(
            "SELECT c.cycle_id FROM canary_cycles AS c "
            "JOIN records AS r ON r.cycle_id = c.cycle_id AND r.source = 'canary' "
            "WHERE c.status = 'finalized' GROUP BY c.cycle_id "
            "ORDER BY c.finalize_seq, MIN(r.rowid)"
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

    def _immutable_cycle_manifest(
        self, conn: sqlite3.Connection
    ) -> dict[str, tuple[tuple[str, str], ...]]:
        """Return ordered record identities and content digests that cannot change.

        Finalized cycles are immutable by contract.  The anytime ledger is
        included independently because losing or rewriting an observation
        after it has entered an e-process would invalidate exactly-once
        processing even if lifecycle metadata were damaged or legacy.
        """
        protected_cycles = {
            str(row[0])
            for row in conn.execute(
                "SELECT cycle_id FROM canary_cycles WHERE status = 'finalized' "
                "UNION SELECT cycle_id FROM anytime_processed_cycles"
            )
        }
        manifest: dict[str, list[tuple[str, str]]] = {cycle_id: [] for cycle_id in protected_cycles}
        rows = conn.execute(
            "SELECT r.cycle_id, r.id, r.record_digest, r.record_json FROM records AS r "
            "WHERE r.source = 'canary' AND ("
            "EXISTS (SELECT 1 FROM canary_cycles AS c "
            "        WHERE c.cycle_id = r.cycle_id AND c.status = 'finalized') "
            "OR EXISTS (SELECT 1 FROM anytime_processed_cycles AS p "
            "           WHERE p.cycle_id = r.cycle_id)) "
            "ORDER BY r.cycle_id, r.rowid"
        )
        for cycle_id_raw, record_id_raw, stored_digest, payload in rows:
            cycle_id = str(cycle_id_raw)
            record_id = str(record_id_raw)
            if payload is None:
                raise RuntimeError(
                    "SQLite record payload cache is incomplete; reopen before rebuilding"
                )
            record = InteractionRecord.from_jsonl(str(payload))
            if (
                record.id != record_id
                or record.cycle_id != cycle_id
                or record.source.value != "canary"
            ):
                raise RuntimeError(
                    f"indexed payload identity does not match record row {record_id!r}"
                )
            canonical_digest = self._record_digest(record)
            if stored_digest is None or str(stored_digest) != canonical_digest:
                raise RuntimeError(
                    f"indexed digest does not match cached payload for record {record_id!r}"
                )
            manifest.setdefault(cycle_id, []).append((record_id, canonical_digest))
        return {cycle_id: tuple(entries) for cycle_id, entries in manifest.items()}

    def rebuild_index(self) -> int:
        """Rebuild record/config indexes while preserving operational state.

        This requires the existing SQLite database. JSONL does not encode
        finalization, checks, alerts, or anytime state, so deleting
        ``index.db`` is not recoverable through this method. Back up the
        whole ``.dedrift`` directory in production. Records belonging to a
        finalized or anytime-processed cycle must reproduce the exact ordered
        identity/content manifest already in SQLite; rebuild refuses to turn
        raw-log edits into a new inference snapshot.

        Returns:
            The number of records indexed.
        """
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._assert_finalized_counts(conn)
            immutable_before = self._immutable_cycle_manifest(conn)
            revision_before = self._current_finalized_revision(conn)
            records = self._read_stable_snapshot()
            lifecycle_rows = conn.execute(
                "SELECT cycle_id, status, open_seq, prospective, expected_records, "
                "finalized_records, "
                "opened_ts, finalized_ts, finalized_offset, finalize_seq FROM canary_cycles"
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
                "(cycle_id, status, open_seq, prospective, expected_records, "
                "finalized_records, opened_ts, "
                "finalized_ts, finalized_offset, finalize_seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                lifecycle_rows,
            )
            # External complete JSONL lines discovered during a rebuild have
            # no completion proof. Register them OPEN; never turn persistence
            # alone into an inference-ready lifecycle claim.
            self._sync_cycle_registry(conn, legacy_finalize=False)
            self._assert_finalized_counts(conn)
            immutable_after = self._immutable_cycle_manifest(conn)
            if immutable_after != immutable_before:
                changed = sorted(
                    cycle_id
                    for cycle_id in set(immutable_before) | set(immutable_after)
                    if immutable_before.get(cycle_id) != immutable_after.get(cycle_id)
                )
                preview = ", ".join(changed[:10])
                raise RuntimeError(
                    "rebuild would alter immutable record identity, order, or content for "
                    "finalized/anytime-processed cycle(s): "
                    + preview
                    + "; restore the append-only log before retrying"
                )
            revision_after = self._current_finalized_revision(conn)
            if revision_after != revision_before:
                raise RuntimeError(
                    "rebuild changed the finalized snapshot revision from "
                    f"{revision_before} to {revision_after}; refusing to commit"
                )
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            self._harden_sqlite_files()
        return len(unique)
