"""Canary suite loader and runner tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dedrift.canary import (
    CORRECTNESS_PREDICATE_KEY,
    DEDRIFT_METADATA_KEY,
    EXPECTATION_FINGERPRINT_KEY,
    EXPECTED_KEY,
    RUBRIC_ID_KEY,
    STRUCTURED_EXPECTED_SUBSET_PREDICATE_ID,
    SUITE_FINGERPRINT_KEY,
    CanaryRunner,
    CanarySuite,
)
from dedrift.schema import Source
from dedrift.signatures import signatures_frame
from dedrift.store import Store
from tests.test_schema import make_config

SUITE_YAML = """\
version: "1"
canaries:
  - id: happy-001
    family: happy_path
    input: {text: "say hi"}
  - id: refusal-001
    family: refusal_boundary
    input: {text: "do something borderline"}
  - id: tool-001
    family: tool_heavy
    input: {text: "compute stuff"}
    expected: {answer: "42"}
"""


def write_suite(tmp_path: Path, content: str = SUITE_YAML) -> Path:
    p = tmp_path / "suite.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def echo_agent(inp: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": f"echo: {inp['text']}",
        "structured": {"answer": "42"},
        "tool_calls": [{"name": "search", "args_schema_ok": True, "order": 1}],
        "steps": 2,
        "tokens_in": 10,
        "tokens_out": 5,
    }


class TestLoader:
    def test_loads_valid_suite(self, tmp_path: Path) -> None:
        suite = CanarySuite.from_yaml(write_suite(tmp_path))
        assert suite.version == "1"
        assert len(suite.canaries) == 3
        assert suite.families() == {"happy_path": 1, "refusal_boundary": 1, "tool_heavy": 1}

    def test_rejects_unknown_family(self, tmp_path: Path) -> None:
        bad = SUITE_YAML.replace("happy_path", "not_a_family")
        with pytest.raises(ValueError, match="unknown family"):
            CanarySuite.from_yaml(write_suite(tmp_path, bad))

    def test_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        bad = SUITE_YAML.replace("refusal-001", "happy-001")
        with pytest.raises(ValueError, match="duplicate canary ids"):
            CanarySuite.from_yaml(write_suite(tmp_path, bad))

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            CanarySuite.from_yaml(write_suite(tmp_path, "- just\n- a list\n"))

    def test_rejects_unknown_or_misspelled_fields(self, tmp_path: Path) -> None:
        bad = SUITE_YAML.replace('expected: {answer: "42"}', 'expectd: {answer: "42"}')
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            CanarySuite.from_yaml(write_suite(tmp_path, bad))

    def test_rejects_empty_expected_contract(self, tmp_path: Path) -> None:
        bad = SUITE_YAML.replace('expected: {answer: "42"}', "expected: {}")
        with pytest.raises(ValueError, match="expected must contain at least one"):
            CanarySuite.from_yaml(write_suite(tmp_path, bad))

    def test_suite_fingerprint_is_order_independent_and_semantic(self, tmp_path: Path) -> None:
        suite = CanarySuite.from_yaml(write_suite(tmp_path))
        reordered = CanarySuite(version=suite.version, canaries=list(reversed(suite.canaries)))
        assert reordered.fingerprint() == suite.fingerprint()
        changed = suite.model_copy(deep=True)
        changed_expected = changed.canaries[-1].expected
        assert changed_expected is not None
        changed_expected["answer"] = "43"
        assert changed.fingerprint() != suite.fingerprint()


class TestRunner:
    def make_runner(self, tmp_path: Path, agent: Any = echo_agent, reps: int = 3) -> CanaryRunner:
        suite = CanarySuite.from_yaml(write_suite(tmp_path))
        return CanaryRunner(suite, agent, make_config(), repetitions=reps)

    def test_rejects_single_repetition(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="repetitions must be >= 2"):
            self.make_runner(tmp_path, reps=1)

    def test_cycle_shape_and_bookkeeping(self, tmp_path: Path) -> None:
        records = self.make_runner(tmp_path).run_cycle(cycle_id="cycle-A")
        assert len(records) == 3 * 3
        assert all(r.cycle_id == "cycle-A" for r in records)
        assert all(r.source == Source.CANARY for r in records)
        per_canary: dict[str, list[int]] = {}
        for record in records:
            assert record.canary_id is not None
            assert record.repetition is not None
            per_canary.setdefault(record.canary_id, []).append(record.repetition)
        assert all(sorted(reps) == [1, 2, 3] for reps in per_canary.values())

    def test_family_stamped_in_metadata(self, tmp_path: Path) -> None:
        records = self.make_runner(tmp_path).run_cycle(cycle_id="cycle-A")
        fams = {r.canary_id: r.input.metadata["family"] for r in records}
        assert fams["happy-001"] == "happy_path"
        assert fams["refusal-001"] == "refusal_boundary"

    def test_correctness_contract_stamped_and_extracted_automatically(self, tmp_path: Path) -> None:
        records = self.make_runner(tmp_path).run_cycle(cycle_id="cycle-A")
        tool_records = [r for r in records if r.canary_id == "tool-001"]
        provenance = tool_records[0].input.metadata[DEDRIFT_METADATA_KEY]
        assert provenance[EXPECTED_KEY] == {"answer": "42"}
        assert provenance[CORRECTNESS_PREDICATE_KEY] == STRUCTURED_EXPECTED_SUBSET_PREDICATE_ID
        assert provenance[EXPECTATION_FINGERPRINT_KEY].startswith("sha256:")
        assert provenance[SUITE_FINGERPRINT_KEY].startswith("sha256:")
        assert provenance[RUBRIC_ID_KEY] is None

        frame = signatures_frame(records)
        exact = frame[frame["canary_id"] == "tool-001"]["exact_match"]
        assert exact.notna().all()
        assert exact.astype(bool).all()

    def test_nested_suite_mutation_changes_next_cycle_identity(self, tmp_path: Path) -> None:
        runner = self.make_runner(tmp_path)
        first = runner.run_cycle(cycle_id="cycle-A")
        first_fingerprint = first[0].input.metadata[DEDRIFT_METADATA_KEY][SUITE_FINGERPRINT_KEY]

        changed_expected = runner.suite.canaries[-1].expected
        assert changed_expected is not None
        changed_expected["answer"] = "43"
        second = runner.run_cycle(cycle_id="cycle-B")
        second_fingerprint = second[0].input.metadata[DEDRIFT_METADATA_KEY][SUITE_FINGERPRINT_KEY]
        assert second_fingerprint != first_fingerprint

    def test_generated_cycle_ids_unique(self, tmp_path: Path) -> None:
        runner = self.make_runner(tmp_path)
        a = runner.run_cycle()[0].cycle_id
        b = runner.run_cycle()[0].cycle_id
        assert a != b

    def test_agent_exception_captured_as_error(self, tmp_path: Path) -> None:
        def broken(inp: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("agent exploded")

        records = self.make_runner(tmp_path, agent=broken).run_cycle(cycle_id="c")
        assert all(r.errors and "agent exploded" in r.errors[0] for r in records)
        assert all(r.output.text == "" for r in records)

    def test_non_dict_return_captured_as_error(self, tmp_path: Path) -> None:
        records = self.make_runner(tmp_path, agent=lambda i: "nope").run_cycle(cycle_id="c")
        assert all("expected dict" in r.errors[0] for r in records)

    def test_records_persist_to_store(self, tmp_path: Path) -> None:
        with Store.init_project(tmp_path) as store:
            records = self.make_runner(tmp_path).run_cycle(store=store, cycle_id="c1")
            assert store.count_records() == len(records)
            assert all(r.cycle_id == "c1" for r in store.read_records())
