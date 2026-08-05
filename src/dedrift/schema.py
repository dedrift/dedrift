"""Pydantic models for dedrift's logging schema (SPEC.md §2.1).

The central type is :class:`InteractionRecord` — one per agent call. Its
``config`` block is hashed into a deterministic ``config_fingerprint``; any
fingerprint change creates a config event in the timeline, which attribution
uses to correlate drift onset with stack changes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class Source(str, Enum):
    """Where a record came from."""

    CANARY = "canary"
    PRODUCTION = "production"


class InteractionInput(BaseModel):
    """Input to a single agent call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionOutput(BaseModel):
    """Output of a single agent call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    structured: dict[str, Any] | None = None


class ToolCall(BaseModel):
    """One tool invocation within an agent call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    args_schema_ok: bool = Field(strict=True)
    order: int = Field(ge=1, strict=True)

    @field_validator("name")
    @classmethod
    def _non_blank_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool-call name must not be blank")
        return value


class AgentConfig(BaseModel):
    """The agent-stack configuration active for a record.

    Hashed deterministically into ``config_fingerprint``. Any change to any
    field (including ``extra``) produces a new fingerprint and therefore a
    config event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    tool_schema_hash: str = Field(min_length=1)
    rag_index_version: str | None = None
    agent_version: str = Field(min_length=1)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", "prompt_hash", "tool_schema_hash", "agent_version")
    @classmethod
    def _non_blank_config_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("configuration identifiers must not be blank")
        return value

    @field_validator("rag_index_version")
    @classmethod
    def _optional_non_blank_config_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("rag_index_version must not be blank when provided")
        return value

    def fingerprint(self) -> str:
        """Return a deterministic sha256 fingerprint of this config.

        Uses canonical JSON (sorted keys, no whitespace variance) so the same
        config always produces the same fingerprint across runs and platforms.

        Returns:
            Hex digest prefixed with ``sha256:``.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


class InteractionRecord(BaseModel):
    """One logged agent interaction (SPEC.md §2.1). Append-only, JSONL."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, strict=True)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Source
    canary_id: str | None = None
    cycle_id: str | None = None
    repetition: int | None = Field(default=None, ge=1, strict=True)
    input: InteractionInput
    output: InteractionOutput
    tool_calls: list[ToolCall] = Field(default_factory=list)
    steps: int = Field(default=0, ge=0, strict=True)
    retries: int = Field(default=0, ge=0, strict=True)
    errors: list[str] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0, strict=True)
    tokens_in: int = Field(default=0, ge=0, strict=True)
    tokens_out: int = Field(default=0, ge=0, strict=True)
    config: AgentConfig

    @field_validator("canary_id", "cycle_id")
    @classmethod
    def _optional_non_empty_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("identifier must be non-empty when provided")
        return value

    @field_validator("repetition")
    @classmethod
    def _positive_repetition(cls, value: int | None) -> int | None:
        if isinstance(value, bool):
            raise ValueError("repetition must be an integer, not boolean")
        if value is not None and value < 1:
            raise ValueError("repetition must be >= 1 when provided")
        return value

    @field_validator("ts")
    @classmethod
    def _timezone_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must include a timezone offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _complete_canary_coordinates(self) -> InteractionRecord:
        if self.source != Source.CANARY:
            return self
        missing = [
            name
            for name, value in (
                ("canary_id", self.canary_id),
                ("cycle_id", self.cycle_id),
                ("repetition", self.repetition),
            )
            if value is None
        ]
        if missing:
            raise ValueError("canary records require " + ", ".join(missing))
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def config_fingerprint(self) -> str:
        """Deterministic fingerprint of the ``config`` block."""
        return self.config.fingerprint()

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line (no trailing newline)."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> InteractionRecord:
        """Parse a record from a JSONL line.

        Args:
            line: One line of JSON as produced by :meth:`to_jsonl`.

        Returns:
            The parsed record.
        """
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("interaction record JSON must be an object")
        declared_fingerprint = payload.pop("config_fingerprint", None)
        record = cls.model_validate(payload)
        if declared_fingerprint is not None and declared_fingerprint != record.config_fingerprint:
            raise ValueError("serialized config_fingerprint does not match the config payload")
        return record
