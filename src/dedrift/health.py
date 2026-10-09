"""Read-only detector health assessment.

``dedrift check`` answers whether the latest finalized behavior differs from
its references.  This module answers the more basic operational question:
*is the detector currently capable of making that comparison honestly?*

The project-level assessment copies a consistency-checked SQLite/WAL snapshot
to a private temporary directory before opening it.  Even SQLite's nominal
``mode=ro`` can create ``-wal``/``-shm`` sidecars beside the source database;
doctor must never repair a log tail, migrate a schema, finalize a cycle, or
otherwise turn observation into mutation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from dedrift.config import ProjectConfig
from dedrift.schema import InteractionRecord
from dedrift.store import _FINALIZED_REVISION_KEY, _PINNED_EMBEDDER_KEY, Store

if TYPE_CHECKING:
    from dedrift.check import CheckResult

HealthStatus = Literal["PASS", "WARN", "FAIL"]
BASELINE_FILE = "baseline.json"
_STALE_OPEN_AFTER = timedelta(hours=24)
_UNSET_FINGERPRINTS = frozenset({"", "unset", "sha256:unset", "unknown", "none", "null"})
_SHA256_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_SNAPSHOT_ATTEMPTS = 3


@dataclass(frozen=True)
class HealthFinding:
    """One stable, machine-sortable detector-health observation."""

    status: HealthStatus
    code: str
    message: str


@dataclass(frozen=True)
class HealthReport:
    """Complete detector-health result."""

    findings: tuple[HealthFinding, ...]

    @property
    def failures(self) -> tuple[HealthFinding, ...]:
        """Return findings that make the detector unready."""
        return tuple(item for item in self.findings if item.status == "FAIL")

    @property
    def warnings(self) -> tuple[HealthFinding, ...]:
        """Return non-blocking limitations."""
        return tuple(item for item in self.findings if item.status == "WARN")

    @property
    def status(self) -> HealthStatus:
        """Conservative aggregate status."""
        if self.failures:
            return "FAIL"
        if self.warnings:
            return "WARN"
        return "PASS"

    @property
    def exit_code(self) -> int:
        """Shell status: nonzero only when an actionable blocker exists."""
        return 1 if self.failures else 0

    def summary(self) -> str:
        """Return a compact human-readable aggregate."""
        passed = sum(item.status == "PASS" for item in self.findings)
        warned = len(self.warnings)
        failed = len(self.failures)
        return f"SUMMARY {self.status} — {passed} passed, {warned} warning(s), {failed} failure(s)"


@dataclass(frozen=True)
class _FileVersion:
    """Identity and content observed during one stable source-file read."""

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    digest: str


class _SnapshotChangedError(RuntimeError):
    """Source database files changed while a private snapshot was copied."""


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _copy_stable_file(source: Path, destination: Path) -> _FileVersion:
    """Copy and hash a file, rejecting any metadata change during the read."""
    before = _stat_identity(source)
    digest = hashlib.sha256()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            descriptor = -1
            while chunk := incoming.read(1024 * 1024):
                digest.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = _stat_identity(source)
    if before != after:
        raise _SnapshotChangedError(f"{source.name} changed while being copied")
    return _FileVersion(*after, digest.hexdigest())


def _stable_file_version(path: Path) -> _FileVersion:
    """Hash a source file while proving its metadata remained stable."""
    before = _stat_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = _stat_identity(path)
    if before != after:
        raise _SnapshotChangedError(f"{path.name} changed during validation")
    return _FileVersion(*after, digest.hexdigest())


def _snapshot_connection(
    path: Path,
) -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory[str]]:
    """Open a private SQLite snapshot without touching source sidecars.

    SQLite WAL state is the database file plus its optional ``-wal`` file;
    ``-shm`` is a derived index that SQLite safely recreates in the private
    directory.  Each copy is hashed and re-hashed at the source.  Presence,
    metadata, and content must remain unchanged across the copy, otherwise a
    bounded retry starts from a fresh temporary directory.
    """
    wal_path = Path(f"{path}-wal")
    journal_path = Path(f"{path}-journal")
    last_error = "database changed during snapshot"
    for _attempt in range(_SNAPSHOT_ATTEMPTS):
        temporary = tempfile.TemporaryDirectory(prefix="dedrift-doctor-")
        temporary_path = Path(temporary.name)
        try:
            if journal_path.exists():
                raise _SnapshotChangedError("rollback journal is active")
            initial_wal = wal_path.is_file()
            sources = [path, *([wal_path] if initial_wal else [])]
            copied: dict[Path, _FileVersion] = {}
            for source in sources:
                destination = temporary_path / (
                    "index.db-wal" if source == wal_path else "index.db"
                )
                copied[source] = _copy_stable_file(source, destination)
            if journal_path.exists() or wal_path.is_file() != initial_wal:
                raise _SnapshotChangedError("SQLite journal set changed during snapshot")
            for source, copied_version in copied.items():
                if _stable_file_version(source) != copied_version:
                    raise _SnapshotChangedError(
                        f"{source.name} changed after it was copied"
                    )
            snapshot_path = temporary_path / "index.db"
            connection = sqlite3.connect(snapshot_path, timeout=5.0)
            connection.execute("PRAGMA query_only = ON")
            return connection, temporary
        except _SnapshotChangedError as exc:
            last_error = str(exc)
            temporary.cleanup()
            continue
        except BaseException:
            temporary.cleanup()
            raise
    raise RuntimeError(
        f"could not obtain a stable SQLite snapshot after {_SNAPSHOT_ATTEMPTS} attempts: "
        f"{last_error}"
    )
def _count_jsonl_lines(path: Path) -> tuple[int, bool]:
    """Count complete JSONL records without loading a production log in memory."""
    count = 0
    last = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count, not last or last == b"\n"


def _validate_indexed_jsonl(
    connection: sqlite3.Connection,
    path: Path,
    indexed_offset: int,
) -> None:
    """Verify raw record order, identity, digest, and cached payload positionally."""
    indexed = [
        (str(record_id), str(digest) if digest is not None else None, payload)
        for record_id, digest, payload in connection.execute(
            "SELECT id, record_digest, record_json FROM records ORDER BY rowid"
        )
    ]
    position = 0
    with path.open("rb") as stream:
        while stream.tell() < indexed_offset:
            raw_line = stream.readline()
            if (
                not raw_line
                or stream.tell() > indexed_offset
                or not raw_line.endswith(b"\n")
            ):
                raise RuntimeError("indexed checkpoint is not on a complete record boundary")
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = InteractionRecord.from_jsonl(stripped.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError("indexed prefix contains an invalid record") from exc
            if position >= len(indexed):
                raise RuntimeError(f"indexed prefix contains extra id {record.id!r}")
            expected_id, stored_digest, cached_payload = indexed[position]
            if record.id != expected_id:
                raise RuntimeError(
                    "indexed prefix order conflicts at position "
                    f"{position + 1}: raw id {record.id!r}, indexed id {expected_id!r}"
                )
            digest = Store._record_digest(record)
            if stored_digest is None or stored_digest != digest or cached_payload is None:
                raise RuntimeError(f"indexed prefix conflicts with record {record.id!r}")
            try:
                cached = InteractionRecord.from_jsonl(str(cached_payload))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"cached payload is invalid for record {record.id!r}") from exc
            if cached.id != record.id or Store._record_digest(cached) != digest:
                raise RuntimeError(f"raw log conflicts with cached record {record.id!r}")
            position += 1
        if stream.tell() != indexed_offset:
            raise RuntimeError("indexed checkpoint is inconsistent")
    if position != len(indexed):
        preview = ", ".join(row[0] for row in indexed[position : position + 5])
        raise RuntimeError(f"index contains record(s) absent from raw log: {preview}")


def _baseline_cycles(path: Path) -> list[str]:
    """Read the exact, fail-closed golden-baseline format."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"golden_cycles"}:
        raise ValueError("baseline.json must contain only a golden_cycles array")
    cycles = data["golden_cycles"]
    if (
        not isinstance(cycles, list)
        or not cycles
        or any(not isinstance(cycle, str) or not cycle for cycle in cycles)
        or len(set(cycles)) != len(cycles)
    ):
        raise ValueError("golden_cycles must be a non-empty array of unique strings")
    return sorted(cycles)


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _valid_custom_scalar(record: InteractionRecord, name: str) -> bool:
    structured = record.output.structured
    if not isinstance(structured, dict):
        return False
    value = structured.get(name)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _record_rows(
    connection: sqlite3.Connection, cycle_ids: list[str]
) -> list[InteractionRecord]:
    if not cycle_ids:
        return []
    marks = ", ".join("?" for _ in cycle_ids)
    rows = connection.execute(
        f"SELECT record_json FROM records WHERE source = 'canary' "
        f"AND cycle_id IN ({marks}) ORDER BY rowid",
        tuple(cycle_ids),
    ).fetchall()
    records: list[InteractionRecord] = []
    for (payload,) in rows:
        if payload is None:
            raise RuntimeError("the indexed record payload cache is incomplete")
        records.append(InteractionRecord.from_jsonl(str(payload)))
    return records


def _check_attribution(
    findings: list[HealthFinding], latest_records: list[InteractionRecord]
) -> None:
    if not latest_records:
        findings.append(
            HealthFinding("FAIL", "attribution", "latest finalized cycle has no records")
        )
        return
    unset: set[str] = set()
    malformed: set[str] = set()
    for record in latest_records:
        for field in ("prompt_hash", "tool_schema_hash"):
            value = str(getattr(record.config, field)).strip()
            if value.lower() in _UNSET_FINGERPRINTS:
                unset.add(field)
            elif value.startswith("sha256:") and not _SHA256_RE.fullmatch(value):
                malformed.add(field)
    identities = {record.config_fingerprint for record in latest_records}
    if unset:
        findings.append(
            HealthFinding(
                "FAIL",
                "attribution",
                "unset latest-cycle attribution fingerprint(s): " + ", ".join(sorted(unset)),
            )
        )
    elif len(identities) > 1:
        findings.append(
            HealthFinding(
                "FAIL",
                "attribution",
                f"latest cycle mixes {len(identities)} agent-stack fingerprints",
            )
        )
    else:
        findings.append(
            HealthFinding("PASS", "attribution", "latest cycle has one attributed agent stack")
        )
    if malformed:
        findings.append(
            HealthFinding(
                "WARN",
                "attribution.format",
                "non-canonical sha256 value(s): " + ", ".join(sorted(malformed)),
            )
        )


def _check_embedder(
    findings: list[HealthFinding],
    connection: sqlite3.Connection,
    store: Store,
    config: ProjectConfig,
) -> None:
    row = connection.execute(
        "SELECT value FROM store_metadata WHERE key = ?", (_PINNED_EMBEDDER_KEY,)
    ).fetchone()
    authoritative = str(row[0]) if row is not None else None
    pin_path = store.project_dir / "embedder.json"
    file_pin: str | None = None
    if pin_path.exists():
        try:
            data = json.loads(pin_path.read_text(encoding="utf-8"))
            value = data["embedder"]
            if not isinstance(value, str) or not value:
                raise ValueError("embedder must be a non-empty string")
            file_pin = value
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            findings.append(
                HealthFinding("FAIL", "embedder", f"embedder.json is invalid: {exc}")
            )
            return
    if authoritative is None and file_pin is not None:
        findings.append(
            HealthFinding("FAIL", "embedder", "pin file exists without authoritative SQLite pin")
        )
        return
    if authoritative is not None and file_pin != authoritative:
        findings.append(
            HealthFinding(
                "FAIL", "embedder", "authoritative SQLite pin and embedder.json disagree"
            )
        )
        return
    if config.embedder and config.embedder != authoritative:
        findings.append(
            HealthFinding(
                "FAIL",
                "embedder",
                f"config expects {config.embedder!r}, authoritative pin is {authoritative!r}",
            )
        )
        return
    if authoritative is None:
        findings.append(
            HealthFinding("WARN", "embedder", "no embedder pinned; semantic channels are off")
        )
        return
    if authoritative != "hash" and not (
        authoritative.startswith("st:") and len(authoritative) > len("st:")
    ):
        findings.append(
            HealthFinding("FAIL", "embedder", f"unsupported pinned embedder {authoritative!r}")
        )
        return
    if (
        authoritative.startswith("st:")
        and importlib.util.find_spec("sentence_transformers") is None
    ):
        findings.append(
            HealthFinding(
                "FAIL",
                "embedder",
                "sentence-transformers pin is active but the embeddings dependency is absent",
            )
        )
        return
    findings.append(
        HealthFinding("PASS", "embedder", f"authoritative pin is consistent: {authoritative}")
    )


def _check_persisted_evidence(
    findings: list[HealthFinding],
    connection: sqlite3.Connection,
    config: ProjectConfig,
    latest_cycle: str,
    finalized_revision: int,
    finalized_cycles: list[str],
    declared_golden: list[str],
    pinned_embedder: str | None,
    latest_records: list[InteractionRecord],
    golden_records: list[InteractionRecord],
) -> None:
    kind = "anytime" if config.inference == "anytime" else "dual"
    row = connection.execute(
        "SELECT params_json, verdict FROM checks WHERE baseline_kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (kind,),
    ).fetchone()
    if row is None:
        findings.append(
            HealthFinding(
                "FAIL", "evidence", f"no persisted {config.inference} check has ever run"
            )
        )
        return
    try:
        params = json.loads(str(row[0]))
        if not isinstance(params, dict):
            raise ValueError("params_json is not an object")
    except ValueError as exc:
        findings.append(HealthFinding("FAIL", "evidence", f"latest check is unreadable: {exc}"))
        return
    freshness_issues: list[str] = []
    checked_cycle = params.get("current_cycle")
    checked_revision = params.get("snapshot_finalized_revision")
    if checked_cycle != latest_cycle:
        freshness_issues.append(
            f"covers {checked_cycle!r}, not finalized cycle {latest_cycle!r}"
        )
    elif checked_revision is None:
        freshness_issues.append("has no finalized-snapshot revision")
    elif checked_revision != finalized_revision:
        freshness_issues.append(
            f"revision {checked_revision!r} != store revision {finalized_revision}"
        )

    if kind == "dual":
        if params.get("declared_golden") != sorted(declared_golden):
            freshness_issues.append("golden baseline changed after the check")
        if "pinned_embedder" not in params or params.get("pinned_embedder") != pinned_embedder:
            freshness_issues.append("authoritative embedder pin changed after the check")
        from dedrift.check import _persistence_fingerprint

        expected_fingerprint = _persistence_fingerprint(
            config,
            declared_golden,
            latest_records,
            pinned_embedder,
        )
        if params.get("persistence_fingerprint") != expected_fingerprint:
            freshness_issues.append("detector/config/suite fingerprint changed after the check")
    else:
        try:
            from dedrift.anytime import (
                ANYTIME_INFERENCE_VERSION,
                EXTRACTOR_VERSION,
                _canonical_suite_identity,
            )
            from dedrift.evalues import epoch_fingerprint

            # Match run_anytime_check exactly: every declared baseline cycle
            # present in the finalized snapshot is reference data, including
            # the latest cycle when declaration happens immediately after it.
            anytime_golden = [
                cycle for cycle in declared_golden if cycle in set(finalized_cycles)
            ]
            golden_suites = {
                _canonical_suite_identity(golden_records, cycle)
                for cycle in anytime_golden
            }
            if len(golden_suites) > 1:
                raise ValueError("golden cycles have different suite identities")
            epoch_suite = next(iter(golden_suites), None) or "no-golden-baseline"
            anytime = config.anytime
            inference_config = json.dumps(
                {
                    "alpha": anytime.alpha,
                    "gamma_total": anytime.gamma_total,
                    "tilts": anytime.tilts,
                    "epoch_allocation": anytime.epoch_allocation,
                    "rate_model": anytime.rate_model,
                    "inference_version": ANYTIME_INFERENCE_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            expected_base = epoch_fingerprint(
                suite_version=epoch_suite,
                embedder=pinned_embedder or "",
                golden_cycles=tuple(anytime_golden),
                extractor_version=EXTRACTOR_VERSION,
                inference_config=inference_config,
            )
            persisted_epoch = params.get("fingerprint")
            epoch_row = connection.execute(
                "SELECT base_fingerprint FROM anytime_epochs WHERE fingerprint = ?",
                (persisted_epoch,),
            ).fetchone()
            latest_epoch = connection.execute(
                "SELECT fingerprint FROM anytime_epochs ORDER BY epoch_index DESC LIMIT 1"
            ).fetchone()
            if epoch_row is None:
                freshness_issues.append("persisted anytime epoch is absent")
            elif str(epoch_row[0]) != expected_base:
                freshness_issues.append("anytime epoch instrument changed after the check")
            if latest_epoch is None or str(latest_epoch[0]) != persisted_epoch:
                freshness_issues.append("persisted check is not for the active anytime epoch")
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            freshness_issues.append(f"anytime epoch identity is not provable: {exc}")

    if freshness_issues:
        findings.append(
            HealthFinding(
                "FAIL",
                "evidence.freshness",
                "; ".join(dict.fromkeys(freshness_issues)) + "; run a new check",
            )
        )
    else:
        findings.append(
            HealthFinding(
                "PASS",
                "evidence.freshness",
                f"latest check matches cycle {latest_cycle} and current instrument",
            )
        )
    if kind == "anytime":
        coverage = str(params.get("coverage_status", "UNKNOWN"))
        processes = params.get("n_processes", 0)
        if coverage != "FULL" or not isinstance(processes, int) or processes < 1:
            findings.append(
                HealthFinding(
                    "FAIL",
                    "evidence.coverage",
                    f"anytime coverage={coverage}, active processes={processes!r}",
                )
            )
        else:
            findings.append(
                HealthFinding(
                    "PASS",
                    "evidence.coverage",
                    f"anytime coverage is FULL across {processes} processes",
                )
            )
    else:
        assessments = params.get("assessments")
        if not isinstance(assessments, list) or not assessments:
            findings.append(
                HealthFinding(
                    "FAIL", "evidence.coverage", "latest check has no coverage assessment"
                )
            )
        else:
            for raw in assessments:
                if not isinstance(raw, dict):
                    findings.append(
                        HealthFinding(
                            "FAIL", "evidence.coverage", "latest coverage assessment is malformed"
                        )
                    )
                    continue
                baseline = str(raw.get("baseline", "unknown"))
                coverage = str(raw.get("coverage_status", "UNKNOWN"))
                tested = raw.get("n_channels_tested", 0)
                total = raw.get("n_channels_total", 0)
                families = raw.get("n_families_tested", 0)
                family_total = raw.get("n_families_total", 0)
                valid_tests = raw.get("n_valid_primary_tests", 0)
                message = (
                    f"{baseline}: coverage={coverage}, channels={tested}/{total}, "
                    f"families={families}/{family_total}"
                )
                complete_counts = (
                    isinstance(tested, int)
                    and not isinstance(tested, bool)
                    and isinstance(total, int)
                    and not isinstance(total, bool)
                    and isinstance(families, int)
                    and not isinstance(families, bool)
                    and isinstance(family_total, int)
                    and not isinstance(family_total, bool)
                    and isinstance(valid_tests, int)
                    and not isinstance(valid_tests, bool)
                    and total > 0
                    and tested == total
                    and family_total > 0
                    and families == family_total
                    and valid_tests > 0
                )
                status: HealthStatus = (
                    "PASS" if coverage == "FULL" and complete_counts else "FAIL"
                )
                findings.append(HealthFinding(status, f"evidence.coverage.{baseline}", message))
            findings.append(
                HealthFinding(
                    "WARN",
                    "evidence.power",
                    "latest fixed check reports coverage, not deployment-specific detection power",
                )
            )
    verdict = str(row[1])
    if any(
        marker in verdict
        for marker in ("DEGRADED", "NO VALID", "NO REFERENCE", "PARTIAL")
    ):
        findings.append(
            HealthFinding(
                "FAIL",
                "evidence.verdict",
                f"latest check is not interpretable: {verdict}",
            )
        )
    elif "DRIFT DETECTED" in verdict:
        findings.append(
            HealthFinding("WARN", "evidence.verdict", "latest check detected drift; investigate")
        )


def assess_health(
    store: Store,
    *,
    now: datetime | None = None,
) -> HealthReport:
    """Inspect one project without changing any project state.

    The returned report is total: filesystem, configuration, and database
    failures become ``FAIL`` findings rather than escaping as exceptions.
    """
    findings: list[HealthFinding] = []
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    required = (store.config_path, store.records_path, store.index_path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        findings.append(
            HealthFinding(
                "FAIL", "project", "missing required artifact(s): " + ", ".join(missing)
            )
        )
        return HealthReport(tuple(findings))
    if store.index_path.stat().st_size == 0:
        findings.append(HealthFinding("FAIL", "project", "authoritative index.db is empty"))
        return HealthReport(tuple(findings))

    try:
        config = ProjectConfig.load(store.project_dir)
    except (OSError, ValueError) as exc:
        findings.append(HealthFinding("FAIL", "config", f"configuration is invalid: {exc}"))
        config = None
    else:
        findings.append(HealthFinding("PASS", "config", "configuration is valid"))

    try:
        line_count, complete_tail = _count_jsonl_lines(store.records_path)
    except OSError as exc:
        findings.append(HealthFinding("FAIL", "log", f"cannot read records.jsonl: {exc}"))
        return HealthReport(tuple(findings))
    if not complete_tail:
        findings.append(HealthFinding("FAIL", "log", "records.jsonl has an incomplete tail"))

    try:
        connection, snapshot_directory = _snapshot_connection(store.index_path)
    except (OSError, sqlite3.Error) as exc:
        findings.append(HealthFinding("FAIL", "database", f"cannot open index.db: {exc}"))
        return HealthReport(tuple(findings))
    try:
        Store._validate_existing_index_identity(connection)
        integrity = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
        if integrity != ["ok"]:
            findings.append(
                HealthFinding(
                    "FAIL",
                    "database",
                    "SQLite integrity check failed: " + "; ".join(integrity[:3]),
                )
            )
            return HealthReport(tuple(findings))
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        required_tables = {"records", "canary_cycles", "checks", "store_metadata"}
        absent_tables = sorted(required_tables - tables)
        if absent_tables:
            findings.append(
                HealthFinding(
                    "FAIL", "database", "missing table(s): " + ", ".join(absent_tables)
                )
            )
            return HealthReport(tuple(findings))
        findings.append(HealthFinding("PASS", "database", "SQLite integrity check passed"))

        indexed = int(connection.execute("SELECT COUNT(*) FROM records").fetchone()[0])
        null_payloads = int(
            connection.execute(
                "SELECT COUNT(*) FROM records WHERE record_json IS NULL"
            ).fetchone()[0]
        )
        offset_row = connection.execute(
            "SELECT value FROM store_metadata WHERE key = 'jsonl_indexed_offset'"
        ).fetchone()
        offset = int(offset_row[0]) if offset_row is not None else -1
        log_size = store.records_path.stat().st_size
        prefix_error: str | None = None
        if 0 <= offset <= log_size:
            try:
                _validate_indexed_jsonl(connection, store.records_path, offset)
            except (OSError, RuntimeError, ValueError) as exc:
                prefix_error = str(exc)
        if null_payloads or indexed != line_count or offset != log_size or prefix_error:
            detail = (
                f"; content validation failed: {prefix_error}" if prefix_error is not None else ""
            )
            findings.append(
                HealthFinding(
                    "FAIL",
                    "log.index",
                    f"raw/index mismatch: lines={line_count}, indexed={indexed}, "
                    f"null payloads={null_payloads}, offset={offset}/{log_size}{detail}",
                )
            )
        elif complete_tail:
            findings.append(
                HealthFinding("PASS", "log.index", f"raw log and index agree on {indexed} records")
            )

        cycle_rows = connection.execute(
            "SELECT c.cycle_id, c.status, c.open_seq, c.prospective, "
            "c.expected_records, c.finalized_records, c.opened_ts, c.finalize_seq, "
            "COUNT(r.id) "
            "FROM canary_cycles AS c LEFT JOIN records AS r "
            "ON r.cycle_id = c.cycle_id AND r.source = 'canary' "
            "GROUP BY c.cycle_id ORDER BY c.open_seq"
        ).fetchall()
        finalized = [str(row[0]) for row in cycle_rows if row[1] == "finalized"]
        finalized_ordered = [
                str(row[0])
                for row in sorted(
                    (row for row in cycle_rows if row[1] == "finalized"),
                    key=lambda row: int(row[7]) if row[7] is not None else -1,
                )
        ]
        open_rows = [row for row in cycle_rows if row[1] == "open"]
        if not finalized_ordered:
            findings.append(
                HealthFinding("FAIL", "cycles.finalized", "no finalized canary cycle exists")
            )
        else:
            findings.append(
                HealthFinding(
                    "PASS", "cycles.finalized", f"{len(finalized_ordered)} finalized cycle(s)"
                )
            )
        invalid_finalized = [
            str(row[0])
            for row in cycle_rows
            if row[1] == "finalized"
            and (row[5] is None or int(row[5]) != int(row[8]))
        ]
        if invalid_finalized:
            findings.append(
                HealthFinding(
                    "FAIL",
                    "cycles.integrity",
                    "finalized record count mismatch: " + ", ".join(invalid_finalized[:5]),
                )
            )
        elif finalized_ordered:
            findings.append(
                HealthFinding(
                    "PASS", "cycles.integrity", "finalized lifecycle counts are consistent"
                )
            )
        blocking_open: list[str] = []
        active_open: list[str] = []
        for row in open_rows:
            cycle_id = str(row[0])
            expected = int(row[4]) if row[4] is not None else None
            observed = int(row[8])
            opened = _parse_utc(row[6])
            stale = opened is None or now_utc - opened > _STALE_OPEN_AFTER
            if expected is not None and observed > expected:
                blocking_open.append(f"{cycle_id} ({observed}>{expected})")
            elif expected is not None and observed == expected:
                blocking_open.append(f"{cycle_id} ({observed}/{expected}, ready to finalize)")
            elif stale:
                blocking_open.append(f"{cycle_id} ({observed}/{expected or '?'}, stale)")
            else:
                active_open.append(f"{cycle_id} ({observed}/{expected or '?'})")
        if blocking_open:
            findings.append(
                HealthFinding(
                    "FAIL",
                    "cycles.open",
                    "blocking open cycle(s): " + ", ".join(blocking_open[:5]),
                )
            )
        elif active_open:
            findings.append(
                HealthFinding(
                    "WARN",
                    "cycles.open",
                    "in-progress open cycle(s): " + ", ".join(active_open[:5]),
                )
            )
        else:
            findings.append(HealthFinding("PASS", "cycles.open", "no open canary cycles"))

        epoch_row = connection.execute(
            "SELECT start_after_open_seq FROM anytime_epochs "
            "ORDER BY epoch_index DESC LIMIT 1"
        ).fetchone()
        epoch_boundary = int(epoch_row[0]) if epoch_row is not None else None
        nonprospective = [row for row in cycle_rows if not bool(row[3])]
        post_epoch_nonprospective = [
            row
            for row in nonprospective
            if epoch_boundary is not None and int(row[2]) > epoch_boundary
        ]
        if post_epoch_nonprospective:
            affected = ", ".join(str(row[0]) for row in post_epoch_nonprospective[:5])
            status: HealthStatus = "FAIL" if config and config.inference == "anytime" else "WARN"
            findings.append(
                HealthFinding(
                    status,
                    "cycles.prospective",
                    "historical/imported cycle(s) after the active anytime epoch boundary: "
                    f"{affected}; they cannot become prospective evidence",
                )
            )
        elif nonprospective:
            findings.append(
                HealthFinding(
                    "WARN",
                    "cycles.prospective",
                    f"{len(nonprospective)} historical/imported cycle(s) lack a "
                    "pre-observation marker",
                )
            )
        elif cycle_rows:
            findings.append(
                HealthFinding(
                    "PASS",
                    "cycles.prospective",
                    "all canary cycles have durable pre-observation markers",
                )
            )

        golden: list[str] = []
        usable_golden: list[str] = []
        baseline_path = store.project_dir / BASELINE_FILE
        if not baseline_path.exists():
            findings.append(
                HealthFinding("FAIL", "baseline", "golden baseline is not configured")
            )
        else:
            try:
                golden = _baseline_cycles(baseline_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                findings.append(
                    HealthFinding("FAIL", "baseline", f"golden baseline is invalid: {exc}")
                )
            else:
                missing_golden = sorted(set(golden) - set(finalized))
                if missing_golden:
                    findings.append(
                        HealthFinding(
                            "FAIL",
                            "baseline",
                            "baseline names unknown/open cycle(s): " + ", ".join(missing_golden),
                        )
                    )
                elif finalized_ordered:
                    latest_cycle = finalized_ordered[-1]
                    usable_golden = [cycle for cycle in golden if cycle != latest_cycle]
                    if not usable_golden:
                        findings.append(
                            HealthFinding(
                                "FAIL", "baseline", "baseline has no usable pre-current cycle"
                            )
                        )
                    else:
                        findings.append(
                            HealthFinding(
                                "PASS",
                                "baseline",
                                f"{len(usable_golden)} usable golden cycle(s)",
                            )
                        )

        if config is not None:
            _check_embedder(findings, connection, store, config)
        if config is None or not finalized_ordered:
            return HealthReport(tuple(findings))
        latest_cycle = finalized_ordered[-1]
        history = finalized_ordered[:-1]
        rolling = [cycle for cycle in history if cycle not in set(golden)][
            -config.rolling_window_cycles :
        ]
        if not rolling:
            findings.append(
                HealthFinding(
                    "FAIL",
                    "baseline.rolling",
                    "no non-golden finalized history remains for sudden-drift comparison",
                )
            )
        elif len(rolling) < config.rolling_window_cycles:
            findings.append(
                HealthFinding(
                    "WARN",
                    "baseline.rolling",
                    f"rolling reference has {len(rolling)}/{config.rolling_window_cycles} cycles",
                )
            )
        else:
            findings.append(
                HealthFinding(
                    "PASS",
                    "baseline.rolling",
                    f"rolling reference has {len(rolling)} cycles",
                )
            )

        latest_records = _record_rows(connection, [latest_cycle])
        reference_cycles = list(dict.fromkeys([*usable_golden, *rolling]))
        reference_records = _record_rows(connection, reference_cycles)
        golden_records = _record_rows(connection, golden)
        _check_attribution(findings, latest_records)

        if config.custom_scalars:
            for name in config.custom_scalars:
                latest_valid = sum(_valid_custom_scalar(record, name) for record in latest_records)
                reference_valid = sum(
                    _valid_custom_scalar(record, name) for record in reference_records
                )
                if latest_valid == 0 or reference_valid == 0:
                    findings.append(
                        HealthFinding(
                            "FAIL",
                            f"channel.{name}",
                            f"declared scalar availability latest={latest_valid}/"
                            f"{len(latest_records)}, reference={reference_valid}/"
                            f"{len(reference_records)}",
                        )
                    )
                elif latest_valid < len(latest_records) or reference_valid < len(reference_records):
                    findings.append(
                        HealthFinding(
                            "WARN",
                            f"channel.{name}",
                            f"partial scalar availability latest={latest_valid}/"
                            f"{len(latest_records)}, reference={reference_valid}/"
                            f"{len(reference_records)}",
                        )
                    )
                else:
                    findings.append(
                        HealthFinding(
                            "PASS",
                            f"channel.{name}",
                            f"available in {latest_valid} latest and "
                            f"{reference_valid} reference records",
                        )
                    )
        else:
            findings.append(
                HealthFinding("PASS", "channels.custom", "no custom scalar channels declared")
            )

        revision_row = connection.execute(
            "SELECT value FROM store_metadata WHERE key = ?", (_FINALIZED_REVISION_KEY,)
        ).fetchone()
        finalized_revision = int(revision_row[0]) if revision_row is not None else 0
        pin_row = connection.execute(
            "SELECT value FROM store_metadata WHERE key = ?", (_PINNED_EMBEDDER_KEY,)
        ).fetchone()
        pinned_embedder = str(pin_row[0]) if pin_row is not None else None
        _check_persisted_evidence(
            findings,
            connection,
            config,
            latest_cycle,
            finalized_revision,
            finalized_ordered,
            golden,
            pinned_embedder,
            latest_records,
            golden_records,
        )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        findings.append(HealthFinding("FAIL", "database", f"health inspection failed: {exc}"))
    finally:
        connection.close()
        snapshot_directory.cleanup()
    return HealthReport(tuple(findings))


def assess_fixed_result_health(
    result: CheckResult, records: list[InteractionRecord]
) -> HealthReport:
    """Build a deterministic health block from one fixed-check snapshot.

    Unlike :func:`assess_health`, this intentionally uses only data frozen in
    the check result and its persisted record snapshot, so re-rendering an old
    report cannot acquire warnings from newer project state.
    """
    findings: list[HealthFinding] = []
    if result.rolling_cycles:
        findings.append(
            HealthFinding(
                "PASS", "baseline.rolling", f"{len(result.rolling_cycles)} rolling cycle(s)"
            )
        )
    else:
        findings.append(HealthFinding("FAIL", "baseline.rolling", "no rolling reference"))
    if result.golden_cycles:
        findings.append(
            HealthFinding(
                "PASS", "baseline.golden", f"{len(result.golden_cycles)} golden cycle(s)"
            )
        )
    else:
        findings.append(HealthFinding("FAIL", "baseline.golden", "no golden reference"))
    for assessment in result.assessments:
        status: HealthStatus = "PASS" if assessment.coverage_status == "FULL" else "FAIL"
        findings.append(
            HealthFinding(
                status,
                f"evidence.coverage.{assessment.baseline}",
                f"{assessment.coverage_status}; channels "
                f"{assessment.n_channels_tested}/{assessment.n_channels_total}, families "
                f"{assessment.n_families_tested}/{assessment.n_families_total}",
            )
        )
    if not result.assessments:
        findings.append(
            HealthFinding("FAIL", "evidence.coverage", "check has no coverage assessment")
        )
    findings.append(
        HealthFinding(
            "PASS" if result.pinned_embedder else "WARN",
            "embedder",
            (
                f"semantic channels use {result.pinned_embedder}"
                if result.pinned_embedder
                else "semantic channels were off"
            ),
        )
    )
    current_records = [record for record in records if record.cycle_id == result.current_cycle]
    _check_attribution(findings, current_records)
    findings.append(
        HealthFinding(
            "WARN",
            "evidence.power",
            "coverage is measured; deployment-specific detection power is not assessed",
        )
    )
    return HealthReport(tuple(findings))
