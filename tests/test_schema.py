"""Schema round-trip and fingerprint determinism tests."""

from __future__ import annotations

from uuid import uuid4

from dedrift.schema import (
    AgentConfig,
    InteractionInput,
    InteractionOutput,
    InteractionRecord,
    Source,
    ToolCall,
)


def make_config(**overrides: object) -> AgentConfig:
    base: dict[str, object] = {
        "model": "provider/model@v1",
        "prompt_hash": "sha256:" + "0" * 64,
        "tool_schema_hash": "sha256:" + "1" * 64,
        "rag_index_version": "rag-1",
        "agent_version": "1.0.0",
    }
    base.update(overrides)
    return AgentConfig.model_validate(base)


def make_record(**overrides: object) -> InteractionRecord:
    marker = uuid4().hex
    base: dict[str, object] = {
        "source": Source.CANARY,
        "canary_id": f"canary-{marker}",
        "cycle_id": "cycle-0001",
        "repetition": 1,
        "input": InteractionInput(text="hello"),
        "output": InteractionOutput(text="world", structured={"answer": "world"}),
        "tool_calls": [ToolCall(name="search", args_schema_ok=True, order=1)],
        "config": make_config(),
    }
    base.update(overrides)
    return InteractionRecord.model_validate(base)


class TestFingerprint:
    def test_deterministic(self) -> None:
        assert make_config().fingerprint() == make_config().fingerprint()

    def test_changes_with_any_field(self) -> None:
        base = make_config().fingerprint()
        assert make_config(model="provider/model@v2").fingerprint() != base
        assert make_config(rag_index_version="rag-2").fingerprint() != base
        assert make_config(extra={"temperature": 0.7}).fingerprint() != base

    def test_extra_dict_key_order_irrelevant(self) -> None:
        a = make_config(extra={"a": 1, "b": 2}).fingerprint()
        b = make_config(extra={"b": 2, "a": 1}).fingerprint()
        assert a == b

    def test_format(self) -> None:
        fp = make_config().fingerprint()
        assert fp.startswith("sha256:")
        assert len(fp) == len("sha256:") + 64


class TestRoundTrip:
    def test_jsonl_round_trip(self) -> None:
        record = make_record()
        parsed = InteractionRecord.from_jsonl(record.to_jsonl())
        assert parsed == record
        assert parsed.config_fingerprint == record.config_fingerprint

    def test_computed_fingerprint_serialized(self) -> None:
        record = make_record()
        assert '"config_fingerprint":"sha256:' in record.to_jsonl().replace(" ", "")
