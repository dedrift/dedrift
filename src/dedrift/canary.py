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

import time
from collections.abc import Callable
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


class Canary(BaseModel):
    """One frozen canary input (SPEC.md §3)."""

    model_config = ConfigDict(frozen=True)

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


class CanarySuite(BaseModel):
    """A versioned, frozen collection of canaries."""

    model_config = ConfigDict(frozen=True)

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


def _coerce_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            calls.append(
                ToolCall(
                    name=str(item.get("name", "unknown")),
                    args_schema_ok=bool(item.get("args_schema_ok", True)),
                    order=int(item.get("order", i + 1)),
                )
            )
    return calls


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
    ) -> None:
        if repetitions < 2:
            msg = "repetitions must be >= 2: single runs cannot support distributional comparison"
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
        records: list[InteractionRecord] = []
        for canary in self.suite.canaries:
            for rep in range(1, self.repetitions + 1):
                records.append(self._run_one(canary, rep, cycle_id))
        if store is not None:
            store.append_many(records)
        return records

    def _run_one(self, canary: Canary, repetition: int, cycle_id: str) -> InteractionRecord:
        started = time.perf_counter()
        errors: list[str] = []
        raw: dict[str, Any] = {}
        try:
            raw = self.agent_fn(dict(canary.input))
            if not isinstance(raw, dict):
                errors.append(f"agent_fn returned {type(raw).__name__}, expected dict")
                raw = {}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        structured = raw.get("structured")
        return InteractionRecord(
            source=Source.CANARY,
            canary_id=canary.id,
            cycle_id=cycle_id,
            repetition=repetition,
            input=InteractionInput(
                text=str(canary.input.get("text", "")),
                metadata={"family": canary.family},
            ),
            output=InteractionOutput(
                text=str(raw.get("text", "")),
                structured=structured if isinstance(structured, dict) else None,
            ),
            tool_calls=_coerce_tool_calls(raw.get("tool_calls")),
            steps=int(raw.get("steps", 0)),
            retries=int(raw.get("retries", 0)),
            errors=errors + [str(e) for e in raw.get("errors", [])],
            latency_ms=int(raw.get("latency_ms", elapsed_ms)),
            tokens_in=int(raw.get("tokens_in", 0)),
            tokens_out=int(raw.get("tokens_out", 0)),
            config=self.agent_config,
        )
