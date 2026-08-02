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

import sqlite3
from pathlib import Path
from types import TracebackType

from dedrift.schema import InteractionRecord

PROJECT_DIR = ".dedrift"
LOGS_DIR = "logs"
RECORDS_FILE = "records.jsonl"
INDEX_FILE = "index.db"
CONFIG_FILE = "config.toml"

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
                            # ~= 0.15%; measured ~1.5%/stream (estimated centering
                            # and scale) - see the PH calibration test.

# Materiality (effect-size) gates, per signature channel. Alerts require BOTH
# statistical significance after FDR AND an effect exceeding these bands.
[materiality]
refusal_rate_pp = 2.0       # percentage-point shift
format_validity_pp = 1.0    # percentage-point shift
rate_default_pp = 2.0       # other rate signatures, percentage points
scalar_cohen_d = 0.5        # |Cohen's d| floor for location tests (Welch)
ks_distance = 0.15          # KS statistic D floor (sup-norm CDF distance):
                            # KS catches shape changes with equal means,
                            # which Cohen's d would wrongly gate out
variance_ratio = 1.5        # variance must grow/shrink by this factor
p95_relative = 0.10         # relative P95 shift floor

[embeddings]
# Pinned forever per project once set; dedrift refuses cross-embedder comparisons.
model = ""
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
    n_tool_calls INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_ts ON records (ts);
CREATE INDEX IF NOT EXISTS idx_records_canary ON records (canary_id, ts);
CREATE INDEX IF NOT EXISTS idx_records_cycle ON records (cycle_id);
CREATE INDEX IF NOT EXISTS idx_records_fingerprint ON records (config_fingerprint);

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
    baseline_kind TEXT NOT NULL,      -- 'rolling' | 'golden'
    params_json TEXT NOT NULL,
    verdict TEXT NOT NULL
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
"""


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
        store.records_path.parent.mkdir(parents=True, exist_ok=True)
        if not store.config_path.exists():
            store.config_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
        store.records_path.touch(exist_ok=True)
        store.connect()
        return store

    def exists(self) -> bool:
        """Return True if this looks like an initialized project."""
        return self.config_path.exists() and self.records_path.exists()

    def connect(self) -> sqlite3.Connection:
        """Open (and memoize) the SQLite connection, creating tables."""
        if self._conn is None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.index_path)
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
        return self._conn

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
        """Append one record to the JSONL log and index it.

        Emits a config event if the record's config fingerprint differs from
        the previously indexed one.

        Args:
            record: The record to persist.
        """
        conn = self.connect()
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(record.to_jsonl() + "\n")
        self._index_record(conn, record)
        conn.commit()

    def append_many(self, records: list[InteractionRecord]) -> None:
        """Append a batch of records (single fsync/commit).

        Args:
            records: Records in the order they occurred.
        """
        conn = self.connect()
        with self.records_path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(record.to_jsonl() + "\n")
        for record in records:
            self._index_record(conn, record)
        conn.commit()

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
            " retries, n_errors, n_tool_calls)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            ),
        )

    # -- reads ---------------------------------------------------------------

    def read_records(self) -> list[InteractionRecord]:
        """Read all records from the JSONL log (source of truth).

        Returns:
            All records in append order.
        """
        if not self.records_path.exists():
            return []
        records: list[InteractionRecord] = []
        with self.records_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(InteractionRecord.from_jsonl(stripped))
        return records

    def count_records(self) -> int:
        """Return the number of indexed records."""
        row = self.connect().execute("SELECT COUNT(*) FROM records").fetchone()
        return int(row[0])

    def config_events(self) -> list[tuple[str, str | None, str]]:
        """Return config events as ``(ts, old_fingerprint, new_fingerprint)``."""
        rows = self.connect().execute(
            "SELECT ts, old_fingerprint, new_fingerprint FROM config_events ORDER BY id"
        )
        return [(r[0], r[1], r[2]) for r in rows]

    # -- maintenance ---------------------------------------------------------

    def rebuild_index(self) -> int:
        """Rebuild the SQLite index from the JSONL log.

        Returns:
            The number of records indexed.
        """
        conn = self.connect()
        conn.execute("DELETE FROM records")
        conn.execute("DELETE FROM config_events")
        records = self.read_records()
        for record in records:
            self._index_record(conn, record)
        conn.commit()
        return len(records)
