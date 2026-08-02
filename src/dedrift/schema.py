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

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Source(str, Enum):
    """Where a record came from."""

    CANARY = "canary"
    PRODUCTION = "production"


class InteractionInput(BaseModel):
    """Input to a single agent call."""

    model_config = ConfigDict(frozen=True)

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionOutput(BaseModel):
    """Output of a single agent call."""

    model_config = ConfigDict(frozen=True)

    text: str
    structured: dict[str, Any] | None = None


class ToolCall(BaseModel):
    """One tool invocation within an agent call."""

    model_config = ConfigDict(frozen=True)

    name: str
    args_schema_ok: bool
    order: int


class AgentConfig(BaseModel):
    """The agent-stack configuration active for a record.

    Hashed deterministically into ``config_fingerprint``. Any change to any
    field (including ``extra``) produces a new fingerprint and therefore a
    config event.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    prompt_hash: str
    tool_schema_hash: str
    rag_index_version: str | None = None
    agent_version: str
    extra: dict[str, Any] = Field(default_factory=dict)

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

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: Source
    canary_id: str | None = None
    cycle_id: str | None = None
    repetition: int | None = None
    input: InteractionInput
    output: InteractionOutput
    tool_calls: list[ToolCall] = Field(default_factory=list)
    steps: int = 0
    retries: int = 0
    errors: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    config: AgentConfig

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
        return cls.model_validate_json(line)
