"""Tier-1 structural signature extraction and aggregation tests."""

from __future__ import annotations

import numpy as np

from dedrift.schema import InteractionOutput, ToolCall
from dedrift.signatures import (
    aggregate_by_canary_cycle,
    aggregate_by_family_cycle,
    extract_record_signature,
    signatures_frame,
)
from dedrift.sim import BehaviorProfile, SimAgent, SimConfig, drifted_profile
from tests.test_schema import make_record


class TestExtraction:
    def test_lengths(self) -> None:
        r = make_record(output=InteractionOutput(text="one two three", structured=None))
        s = extract_record_signature(r)
        assert s.output_words == 3
        assert s.output_chars == 13

    def test_refusal_detection(self) -> None:
        refusals = [
            "I'm sorry, but I can't help with that.",
            "I cannot assist with this request.",
            "As an AI, I can't do that.",
        ]
        for text in refusals:
            r = make_record(output=InteractionOutput(text=text, structured=None))
            assert extract_record_signature(r).refusal, text
        ok = make_record(output=InteractionOutput(text="Here is the summary.", structured=None))
        assert not extract_record_signature(ok).refusal

    def test_format_validity(self) -> None:
        valid = make_record(output=InteractionOutput(text="x", structured={"answer": "y"}))
        assert extract_record_signature(valid).format_valid
        missing = make_record(output=InteractionOutput(text="x", structured=None))
        assert not extract_record_signature(missing).format_valid
        # expected keys enforced when provided
        assert not extract_record_signature(valid, expected={"other_key": 1}).format_valid

    def test_exact_match(self) -> None:
        r = make_record(output=InteractionOutput(text="x", structured={"answer": "42"}))
        assert extract_record_signature(r, expected={"answer": "42"}).exact_match is True
        assert extract_record_signature(r, expected={"answer": "43"}).exact_match is False
        assert extract_record_signature(r).exact_match is None

    def test_tool_stats(self) -> None:
        r = make_record(
            tool_calls=[
                ToolCall(name="search", args_schema_ok=True, order=1),
                ToolCall(name="search", args_schema_ok=False, order=2),
                ToolCall(name="calc", args_schema_ok=True, order=3),
            ]
        )
        s = extract_record_signature(r)
        assert s.tool_call_count == 3
        assert s.tool_usage == {"search": 2, "calc": 1}
        assert not s.args_schema_ok_all

    def test_tool_order_inversions(self) -> None:
        # Alphabetical order (calc < lookup < search) has zero inversions.
        ordered = make_record(
            tool_calls=[
                ToolCall(name="calc", args_schema_ok=True, order=1),
                ToolCall(name="lookup", args_schema_ok=True, order=2),
                ToolCall(name="search", args_schema_ok=True, order=3),
            ]
        )
        assert extract_record_signature(ordered).tool_order_inversions == 0
        # The exact reverse has the maximum, C(3,2) = 3.
        reversed_ = make_record(
            tool_calls=[
                ToolCall(name="search", args_schema_ok=True, order=1),
                ToolCall(name="lookup", args_schema_ok=True, order=2),
                ToolCall(name="calc", args_schema_ok=True, order=3),
            ]
        )
        assert extract_record_signature(reversed_).tool_order_inversions == 3
        # Fewer than two calls contribute zero.
        singleton = make_record(
            tool_calls=[ToolCall(name="search", args_schema_ok=True, order=1)]
        )
        assert extract_record_signature(singleton).tool_order_inversions == 0

    def test_family_from_metadata(self) -> None:
        r = make_record()
        assert extract_record_signature(r).family == "unknown"


class TestAggregation:
    def make_frame(self, change_cycle: int | None = None) -> np.ndarray:  # type: ignore[type-arg]
        config = SimConfig(
            n_canaries=12,
            repetitions=5,
            post=drifted_profile(BehaviorProfile()),
            change_cycle=change_cycle,
            seed=11,
        )
        records = SimAgent(config).run_cycles(3)
        return signatures_frame(records)  # type: ignore[return-value]

    def test_family_cycle_shape(self) -> None:
        frame = self.make_frame()
        table = aggregate_by_family_cycle(frame)
        # 12 canaries round-robin over 6 families = 6 families, 3 cycles
        assert len(table) == 6 * 3
        assert set(table["n"]) == {2 * 5}  # 2 canaries per family x 5 reps

    def test_canary_cycle_shape(self) -> None:
        frame = self.make_frame()
        table = aggregate_by_canary_cycle(frame)
        assert len(table) == 12 * 3
        assert set(table["n"]) == {5}

    def test_dispersion_columns_present(self) -> None:
        table = aggregate_by_family_cycle(self.make_frame())
        for col in (
            "output_words_mean",
            "output_words_var",
            "output_words_p95",
            "latency_ms_var",
            "latency_ms_p95",
        ):
            assert col in table.columns, col

    def test_aggregation_math_matches_numpy(self) -> None:
        frame = self.make_frame()
        table = aggregate_by_canary_cycle(frame)
        row = table.iloc[0]
        sub = frame[
            (frame["canary_id"] == row["canary_id"]) & (frame["cycle_id"] == row["cycle_id"])
        ]
        vals = sub["output_words"].to_numpy(dtype=float)
        assert np.isclose(row["output_words_mean"], vals.mean())
        assert np.isclose(row["output_words_var"], vals.var(ddof=1))
        assert np.isclose(row["output_words_p95"], np.percentile(vals, 95))
        assert np.isclose(row["refusal_rate"], sub["refusal"].mean())

    def test_drift_visible_in_family_table(self) -> None:
        frame = self.make_frame(change_cycle=2)
        table = aggregate_by_family_cycle(frame)
        pre = table[table["cycle_id"] != "cycle-0002"]["output_words_mean"].mean()
        post = table[table["cycle_id"] == "cycle-0002"]["output_words_mean"].mean()
        assert post > pre * 1.2

    def test_empty_frame(self) -> None:
        import pandas as pd

        assert aggregate_by_family_cycle(pd.DataFrame()).empty
