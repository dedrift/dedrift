"""Canary suite: YAML loader and N-repetition runner (SPEC.md §3).

A canary suite is a frozen set of inputs, stratified by family. The runner
executes each canary N times per cycle against a user-supplied callable —
dedrift never implements the agent, it only calls it:

    def agent_fn(input: dict) -> dict:
        # keys understood in the returned dict (all optional except "text"):
        #   text: str                    - output text
        #   structured: dict | None     - parsed structured output
        #   tool_calls: list[dict]      - {"name", "args_schema_ok", "order"}
        #   steps, retries: int
        #   errors: list[str]
        #   latency_ms, tokens_in, tokens_out: int
        ...

A cycle (one full suite execution) is the unit of comparison for all
downstream drift detection.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dedrift.schema import (
    AgentConfig,
    InteractionInput,
    InteractionOutput,
    InteractionRecord,
    Source,
    ToolCall,
)
from dedrift.store import Store

AgentFn = Callable[[dict[str, Any]], dict[str, Any]]

FAMILIES = (
    "happy_path",
    "edge_case",
    "refusal_boundary",
    "tool_heavy",
    "adversarial",
    "long_context",
)

# Reserved interaction-metadata namespace written by :class:`CanaryRunner`.
# Keeping the evaluation contract beside every record makes historical
# correctness checks reproducible without changing the public logging schema.
DEDRIFT_METADATA_KEY = "_dedrift"
SUITE_FINGERPRINT_KEY = "suite_fingerprint"
CANARY_FINGERPRINT_KEY = "canary_fingerprint"
EXPECTED_KEY = "expected"
EXPECTATION_FINGERPRINT_KEY = "expectation_fingerprint"
CORRECTNESS_PREDICATE_KEY = "correctness_predicate_id"
RUBRIC_ID_KEY = "rubric_id"
SUITE_VERSION_KEY = "suite_version"

# Historical ``exact_match`` behavior is deliberately preserved: every
# expected key/value must match, while additional structured-output fields
# are allowed.  The versioned name makes that less expansive meaning explicit
# and gives future predicate changes a safe identity boundary.
STRUCTURED_EXPECTED_SUBSET_PREDICATE_ID = "structured_expected_subset.v1"


def _canonical_digest(payload: Any) -> str:
    """Return a stable SHA-256 identity for a JSON-compatible payload."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def expectation_fingerprint(expected: dict[str, Any]) -> str:
    """Identity of the built-in predicate together with expected values."""
    return _canonical_digest(
        {
            "predicate_id": STRUCTURED_EXPECTED_SUBSET_PREDICATE_ID,
            "expected": expected,
        }
    )


class Canary(BaseModel):
    """One frozen canary input (SPEC.md §3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    family: str
    input: dict[str, Any]
    expected: dict[str, Any] | None = None
    rubric_id: str | None = None

    @field_validator("family")
    @classmethod
    def _known_family(cls, v: str) -> str:
        if v not in FAMILIES:
            msg = f"unknown family {v!r}; expected one of {FAMILIES}"
            raise ValueError(msg)
        return v

    @field_validator("expected")
    @classmethod
    def _non_empty_expected(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value == {}:
            msg = (
                "expected must contain at least one key/value criterion; "
                "an empty structured-subset predicate would pass every structured answer"
            )
            raise ValueError(msg)
        return value

    @property
    def correctness_predicate_id(self) -> str | None:
        """Predicate actually executed by the fixed structural checker."""
        if self.expected is None:
            return None
        return STRUCTURED_EXPECTED_SUBSET_PREDICATE_ID

    @property
    def expectation_fingerprint(self) -> str | None:
        """Stable identity of the predicate and its expected values."""
        if self.expected is None:
            return None
        return expectation_fingerprint(self.model_dump(mode="json")["expected"])

    def fingerprint(self) -> str:
        """Stable identity of this canary's complete evaluation contract."""
        data = self.model_dump(mode="json")
        return _canonical_digest(
            {
                "id": data["id"],
                "family": data["family"],
                "input": data["input"],
                "expected": data["expected"],
                "rubric_id": data["rubric_id"],
                "correctness_predicate_id": self.correctness_predicate_id,
            }
        )


class CanarySuite(BaseModel):
    """A versioned, frozen collection of canaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    canaries: list[Canary] = Field(min_length=1)

    @field_validator("canaries")
    @classmethod
    def _unique_ids(cls, v: list[Canary]) -> list[Canary]:
        ids = [c.id for c in v]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            msg = f"duplicate canary ids: {dupes}"
            raise ValueError(msg)
        return v

    @classmethod
    def from_yaml(cls, path: Path | str) -> CanarySuite:
        """Load and validate a suite from a YAML file.

        Args:
            path: Path to the suite YAML.

        Returns:
            The validated suite.

        Raises:
            ValueError: If the YAML is not a mapping or fails validation.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"canary suite YAML must be a mapping, got {type(raw).__name__}"
            raise ValueError(msg)
        return cls.model_validate(raw)

    def families(self) -> dict[str, int]:
        """Return canary counts per family."""
        counts: dict[str, int] = {}
        for canary in self.canaries:
            counts[canary.family] = counts.get(canary.family, 0) + 1
        return counts

    def fingerprint(self) -> str:
        """Canonical full-suite identity, independent of YAML item order.

        Version and every behavior-affecting canary field are included.
        Sorting by the already-unique canary ID means cosmetic YAML reordering
        does not reset monitoring state, while any semantic edit does.
        """
        canaries = []
        for canary in sorted(self.canaries, key=lambda item: item.id):
            data = canary.model_dump(mode="json")
            canaries.append(
                {
                    "id": data["id"],
                    "family": data["family"],
                    "input": data["input"],
                    "expected": data["expected"],
                    "rubric_id": data["rubric_id"],
                    "correctness_predicate_id": canary.correctness_predicate_id,
                }
            )
        return _canonical_digest({"version": self.version, "canaries": canaries})


def _coerce_nonnegative_int(
    raw: Any,
    field: str,
    errors: list[str],
    *,
    default: int = 0,
    minimum: int = 0,
) -> int:
    """Coerce one adapter counter without letting malformed output abort a cycle."""
    if raw is None:
        return default
    try:
        if isinstance(raw, bool):
            raise ValueError("boolean is not an integer counter")
        value = int(raw)
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("fractional value is not an integer counter")
        if value < minimum:
            raise ValueError(f"must be >= {minimum}")
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"invalid agent output field {field!r}: {exc}")
        return default
    return value


def _coerce_tool_calls(raw: Any, errors: list[str]) -> list[ToolCall]:
    if not isinstance(raw, list):
        if raw is not None:
            errors.append("invalid agent output field 'tool_calls': expected list")
        return []
    calls: list[ToolCall] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"invalid agent output tool_calls[{i}]: expected object")
            continue
        name = item.get("name", "unknown")
        try:
            name_text = str(name)
        except Exception as exc:  # an adapter object can have a broken __str__
            errors.append(f"invalid agent output tool_calls[{i}].name: {exc}")
            name_text = "unknown"
        if not name_text:
            errors.append(f"invalid agent output tool_calls[{i}].name: must be non-empty")
            name_text = "unknown"
        calls.append(
            ToolCall(
                name=name_text,
                args_schema_ok=bool(item.get("args_schema_ok", True)),
                order=_coerce_nonnegative_int(
                    item.get("order", i + 1),
                    f"tool_calls[{i}].order",
                    errors,
                    default=i + 1,
                    minimum=1,
                ),
            )
        )
    return calls


def _coerce_errors(raw: Any, errors: list[str]) -> list[str]:
    """Return adapter error messages, recording malformed containers safely."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append("invalid agent output field 'errors': expected list")
        return []
    converted: list[str] = []
    for index, item in enumerate(raw):
        try:
            converted.append(str(item))
        except Exception as exc:
            errors.append(f"invalid agent output errors[{index}]: {exc}")
    return converted


def _json_object_snapshot(raw: Any, field: str, errors: list[str]) -> dict[str, Any] | None:
    """Detach a JSON-compatible object returned by an untrusted adapter."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        errors.append(f"invalid agent output field {field!r}: expected object or null")
        return None
    try:
        # A JSON round trip both rejects unsupported objects and prevents a
        # reused nested dictionary from mutating records from earlier calls.
        snapshot = json.loads(json.dumps(raw, ensure_ascii=True))
        if not isinstance(snapshot, dict):  # json object round-trip invariant
            raise TypeError("serialized object did not decode to an object")
        return snapshot
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"invalid agent output field {field!r}: not JSON-compatible ({exc})")
        return None


class CanaryRunner:
    """Executes a canary suite against a user-supplied agent callable.

    Args:
        suite: The frozen canary suite.
        agent_fn: User callable mapping an input dict to an output dict.
        agent_config: The agent-stack config to stamp on every record
            (drives config fingerprinting and attribution).
        repetitions: N repeated runs per canary per cycle (SPEC default 7).
    """

    def __init__(
        self,
        suite: CanarySuite,
        agent_fn: AgentFn,
        agent_config: AgentConfig,
        repetitions: int = 7,
        deterministic: bool = False,
    ) -> None:
        if repetitions < 2 and not deterministic:
            msg = (
                "repetitions must be >= 2: single runs cannot support distributional "
                "comparison. Set deterministic=True if the agent is exactly reproducible "
                "at fixed inputs, in which case the second repetition is a byte-identical "
                "copy that adds storage rather than information."
            )
            raise ValueError(msg)
        if repetitions < 1:
            msg = "repetitions must be >= 1"
            raise ValueError(msg)
        self.suite = suite
        self.agent_fn = agent_fn
        self.agent_config = agent_config
        self.repetitions = repetitions

    def run_cycle(
        self, store: Store | None = None, cycle_id: str | None = None
    ) -> list[InteractionRecord]:
        """Run one full cycle: every canary, N repetitions each.

        Args:
            store: If given, records are appended to it as one batch.
            cycle_id: Explicit cycle identifier; a timestamped UUID-suffixed
                identifier is generated when omitted.

        Returns:
            All records produced this cycle, in execution order.
        """
        if cycle_id is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            cycle_id = f"cycle-{stamp}-{uuid4().hex[:8]}"
        # Pydantic's frozen models are shallow. Snapshot the complete suite at
        # the cycle boundary so neither caller code nor an agent that mutates
        # its input can alter later repetitions or the provenance stamped on
        # records already being collected.
        suite_snapshot = CanarySuite.model_validate(self.suite.model_dump(mode="json"))
        suite_fingerprint = suite_snapshot.fingerprint()
        records: list[InteractionRecord] = []
        for canary in suite_snapshot.canaries:
            for rep in range(1, self.repetitions + 1):
                records.append(
                    self._run_one(
                        canary,
                        rep,
                        cycle_id,
                        suite_snapshot.version,
                        suite_fingerprint,
                    )
                )
        if store is not None:
            store.append_many(
                records,
                finalize_cycles=True,
                expected_cycle_counts={cycle_id: len(records)},
            )
        return records

    def _run_one(
        self,
        canary: Canary,
        repetition: int,
        cycle_id: str,
        suite_version: str,
        suite_fingerprint: str,
    ) -> InteractionRecord:
        started = time.perf_counter()
        errors: list[str] = []
        raw: dict[str, Any] = {}
        canary_data = canary.model_dump(mode="json")
        input_snapshot = canary_data["input"]
        canary_fingerprint = canary.fingerprint()
        expected = canary_data["expected"]
        expected_fingerprint = canary.expectation_fingerprint
        correctness_predicate = canary.correctness_predicate_id
        rubric_id = canary.rubric_id
        try:
            returned = self.agent_fn(deepcopy(input_snapshot))
            if not isinstance(returned, dict):
                errors.append(f"agent_fn returned {type(returned).__name__}, expected dict")
            else:
                # Detach at the adapter boundary. Some SDKs reuse and mutate
                # response containers; retaining those aliases corrupts prior
                # repetitions and their persisted provenance.
                try:
                    raw = deepcopy(returned)
                except Exception as exc:
                    errors.append(f"could not snapshot agent output: {type(exc).__name__}: {exc}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        structured = _json_object_snapshot(raw.get("structured"), "structured", errors)
        try:
            output_text = str(raw.get("text", ""))
        except Exception as exc:
            errors.append(f"invalid agent output field 'text': {exc}")
            output_text = ""
        tool_calls = _coerce_tool_calls(raw.get("tool_calls"), errors)
        steps = _coerce_nonnegative_int(raw.get("steps"), "steps", errors)
        retries = _coerce_nonnegative_int(raw.get("retries"), "retries", errors)
        latency_ms = _coerce_nonnegative_int(
            raw.get("latency_ms"), "latency_ms", errors, default=elapsed_ms
        )
        tokens_in = _coerce_nonnegative_int(raw.get("tokens_in"), "tokens_in", errors)
        tokens_out = _coerce_nonnegative_int(raw.get("tokens_out"), "tokens_out", errors)
        adapter_errors = _coerce_errors(raw.get("errors"), errors)
        correctness_metadata = {
            SUITE_VERSION_KEY: suite_version,
            SUITE_FINGERPRINT_KEY: suite_fingerprint,
            CANARY_FINGERPRINT_KEY: canary_fingerprint,
            EXPECTED_KEY: expected,
            EXPECTATION_FINGERPRINT_KEY: expected_fingerprint,
            CORRECTNESS_PREDICATE_KEY: correctness_predicate,
            RUBRIC_ID_KEY: rubric_id,
        }
        return InteractionRecord(
            source=Source.CANARY,
            canary_id=canary.id,
            cycle_id=cycle_id,
            repetition=repetition,
            input=InteractionInput(
                text=str(input_snapshot.get("text", "")),
                metadata={
                    "family": canary.family,
                    DEDRIFT_METADATA_KEY: correctness_metadata,
                },
            ),
            output=InteractionOutput(
                text=output_text,
                structured=structured,
            ),
            tool_calls=tool_calls,
            steps=steps,
            retries=retries,
            errors=errors + adapter_errors,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            config=self.agent_config,
        )
