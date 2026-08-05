"""Tier-1 structural signatures (SPEC.md §4) — always on, zero ML dependencies.

Per-record extraction produces one :class:`RecordSignature`; aggregation
produces per-(canary, cycle) and per-(family, cycle) tables. Per the
dispersion rule, aggregates always include variance and P95 alongside means —
agents often go erratic before they go wrong on average.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from dedrift.canary import (
    CANARY_FINGERPRINT_KEY,
    CORRECTNESS_PREDICATE_KEY,
    DEDRIFT_METADATA_KEY,
    EXPECTATION_FINGERPRINT_KEY,
    EXPECTED_KEY,
    RUBRIC_ID_KEY,
    STRUCTURED_EXPECTED_SUBSET_PREDICATE_ID,
    SUITE_FINGERPRINT_KEY,
    expectation_fingerprint,
)
from dedrift.schema import InteractionRecord

DEFAULT_REFUSAL_PATTERNS: tuple[str, ...] = (
    r"\bi'?m sorry\b",
    r"\bi can(?:no|')t\b",
    r"\bi cannot\b",
    r"\bi am unable\b",
    r"\bi'?m unable\b",
    r"\bcan(?:no|')t help with\b",
    r"\bcannot help with\b",
    r"\bcan(?:no|')t assist\b",
    r"\bcannot assist\b",
    r"\bas an ai\b.{0,40}\b(?:can(?:no|')t|cannot|won'?t)\b",
)

#: Scalar columns aggregated with mean/var/p95.
SCALAR_SIGNATURES: tuple[str, ...] = (
    "output_chars",
    "output_words",
    "tool_call_count",
    "steps",
    "retries",
    "latency_ms",
    "tokens_out",
)

#: Boolean columns aggregated as rates (plus var/p95 are meaningless; rate only).
RATE_SIGNATURES: tuple[str, ...] = (
    "refusal",
    "format_valid",
    "args_schema_ok_all",
    "had_error",
    "exact_match",
)


@dataclass(frozen=True)
class RecordSignature:
    """Structural signature of one interaction record.

    Attributes:
        record_id: Source record ID.
        canary_id: Canary ID (None for production records).
        cycle_id: Cycle ID (None for production records).
        family: Canary family, from input metadata ("unknown" if absent).
        config_fingerprint: Config fingerprint of the source record.
        suite_fingerprint: Canonical identity of the suite which produced the
            record, when written by :class:`dedrift.canary.CanaryRunner`.
        canary_fingerprint: Identity of the input and evaluation contract.
        correctness_predicate_id: Versioned predicate applied to ``expected``.
        expectation_fingerprint: Identity of predicate plus expected values.
        rubric_id: Preserved judge rubric identity. Structural extraction does
            not execute rubric-based judging.
        output_chars: Output length in characters.
        output_words: Output length in whitespace-delimited words.
        refusal: True if the output matches a refusal pattern.
        format_valid: True if structured output is present and JSON-serializable,
            and (when ``expected`` is given) contains all required keys.
        exact_match: True if all ``expected`` key/value pairs match the
            structured output exactly; None when no ``expected`` was given.
        tool_call_count: Number of tool calls.
        tool_usage: Tool-name usage counts.
        args_schema_ok_all: True if every tool call had schema-valid args
            (vacuously True with zero tool calls).
        steps: Agent step count.
        retries: Retry count.
        had_error: True if the record carries any errors.
        latency_ms: Latency in milliseconds.
        tokens_out: Output token count.
    """

    record_id: str
    canary_id: str | None
    cycle_id: str | None
    family: str
    config_fingerprint: str
    suite_fingerprint: str | None
    canary_fingerprint: str | None
    correctness_predicate_id: str | None
    expectation_fingerprint: str | None
    rubric_id: str | None
    output_chars: int
    output_words: int
    refusal: bool
    format_valid: bool
    exact_match: bool | None
    tool_call_count: int
    tool_usage: dict[str, int]
    args_schema_ok_all: bool
    steps: int
    retries: int
    had_error: bool
    latency_ms: int
    tokens_out: int


def _is_refusal(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def _format_valid(structured: dict[str, Any] | None, expected: dict[str, Any] | None) -> bool:
    if structured is None:
        return False
    try:
        json.dumps(structured)
    except (TypeError, ValueError):
        return False
    if expected is not None:
        return all(key in structured for key in expected)
    return True


def _exact_match(structured: dict[str, Any] | None, expected: dict[str, Any] | None) -> bool | None:
    if expected is None:
        return None
    if structured is None:
        return False
    return all(structured.get(k) == v for k, v in expected.items())


def _dedrift_metadata(record: InteractionRecord) -> dict[str, Any]:
    """Return validated record-local canary provenance, or an empty mapping."""
    value = record.input.metadata.get(DEDRIFT_METADATA_KEY)
    if not isinstance(value, dict):
        return {}
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def extract_record_signature(
    record: InteractionRecord,
    expected: dict[str, Any] | None = None,
    refusal_patterns: tuple[str, ...] = DEFAULT_REFUSAL_PATTERNS,
) -> RecordSignature:
    """Extract the Tier-1 structural signature from one record.

    Args:
        record: The source record.
        expected: The canary's ``expected`` fields, if any (exact-match tier).
        refusal_patterns: Regex patterns (matched case-insensitively against
            the output text) that flag a refusal. Configurable per project.

    Returns:
        The extracted signature.
    """
    text = record.output.text
    provenance = _dedrift_metadata(record)
    record_expected = provenance.get(EXPECTED_KEY)
    if expected is None and isinstance(record_expected, dict):
        expected = record_expected
    predicate_id = _optional_string(provenance.get(CORRECTNESS_PREDICATE_KEY))
    expected_id = _optional_string(provenance.get(EXPECTATION_FINGERPRINT_KEY))
    if expected is not None:
        # Explicit ``expected_by_canary`` maps remain supported. They predate
        # record-local provenance, so derive truthful identities when the old
        # caller path supplies the expectation.
        predicate_id = STRUCTURED_EXPECTED_SUBSET_PREDICATE_ID
        expected_id = expectation_fingerprint(expected)
    usage: dict[str, int] = {}
    for call in record.tool_calls:
        usage[call.name] = usage.get(call.name, 0) + 1
    family = str(record.input.metadata.get("family", "unknown"))
    return RecordSignature(
        record_id=record.id,
        canary_id=record.canary_id,
        cycle_id=record.cycle_id,
        family=family,
        config_fingerprint=record.config_fingerprint,
        suite_fingerprint=_optional_string(provenance.get(SUITE_FINGERPRINT_KEY)),
        canary_fingerprint=_optional_string(provenance.get(CANARY_FINGERPRINT_KEY)),
        correctness_predicate_id=predicate_id,
        expectation_fingerprint=expected_id,
        rubric_id=_optional_string(provenance.get(RUBRIC_ID_KEY)),
        output_chars=len(text),
        output_words=len(text.split()),
        refusal=_is_refusal(text, refusal_patterns),
        format_valid=_format_valid(record.output.structured, expected),
        exact_match=_exact_match(record.output.structured, expected),
        tool_call_count=len(record.tool_calls),
        tool_usage=usage,
        args_schema_ok_all=all(c.args_schema_ok for c in record.tool_calls),
        steps=record.steps,
        retries=record.retries,
        had_error=len(record.errors) > 0,
        latency_ms=record.latency_ms,
        tokens_out=record.tokens_out,
    )


def signatures_frame(
    records: list[InteractionRecord],
    expected_by_canary: dict[str, dict[str, Any]] | None = None,
    refusal_patterns: tuple[str, ...] = DEFAULT_REFUSAL_PATTERNS,
) -> pd.DataFrame:
    """Extract signatures for many records into a flat DataFrame.

    Args:
        records: Source records (canary records need ``cycle_id`` set).
        expected_by_canary: Optional compatibility map canary_id -> expected
            fields. Record-local CanaryRunner metadata is used automatically;
            an explicit map entry takes precedence.
        refusal_patterns: See :func:`extract_record_signature`.

    Returns:
        One row per record; ``tool_usage`` is kept as a dict column.
    """
    expected_by_canary = expected_by_canary or {}
    rows = [
        asdict(
            extract_record_signature(
                r,
                expected=expected_by_canary.get(r.canary_id or ""),
                refusal_patterns=refusal_patterns,
            )
        )
        for r in records
    ]
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aggregate a signature frame by ``keys``.

    Scalars get mean/var/p95; booleans become rates. ``exact_match`` is a rate
    over non-null values only (canaries without ``expected`` are excluded).
    """
    if frame.empty:
        return pd.DataFrame()

    def p95(series: pd.Series[Any]) -> float:
        return float(np.percentile(series.to_numpy(dtype=float), 95))

    pieces: list[pd.DataFrame] = []
    grouped = frame.groupby(keys, dropna=False)

    agg_spec: dict[str, list[Any]] = {col: ["mean", "var", p95] for col in SCALAR_SIGNATURES}
    scalars = grouped.agg(agg_spec)
    scalars.columns = [
        f"{col}_{stat if isinstance(stat, str) else 'p95'}" for col, stat in scalars.columns
    ]
    pieces.append(scalars)

    for col in RATE_SIGNATURES:
        if col == "exact_match":
            rate = grouped[col].agg(lambda s: s.dropna().mean() if s.notna().any() else np.nan)
        else:
            rate = grouped[col].mean()
        pieces.append(rate.to_frame(name=f"{col}_rate"))

    pieces.append(grouped.size().to_frame(name="n"))
    return pd.concat(pieces, axis=1).reset_index()


def aggregate_by_canary_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signatures per (canary, cycle).

    Args:
        frame: Output of :func:`signatures_frame`.

    Returns:
        One row per (canary_id, cycle_id) with mean/var/p95 and rates.
    """
    return _aggregate(frame, ["canary_id", "cycle_id"])


def aggregate_by_family_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate signatures per (family, cycle).

    Args:
        frame: Output of :func:`signatures_frame`.

    Returns:
        One row per (family, cycle_id) with mean/var/p95 and rates.
    """
    return _aggregate(frame, ["family", "cycle_id"])
