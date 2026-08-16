"""Support for agents that are not language models (issue #9).

The observation model was written for text agents, and a field report from a
deterministic numeric scoring model showed what that costs: seven of the eight
built-in scalar channels are constant by construction for such an agent, the
eighth is latency, and the quantities that actually move first -- the model's
own continuous outputs -- had nowhere to be registered. These tests pin the
behaviours added in response.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from dedrift.canary import Canary, CanaryRunner, CanarySuite
from dedrift.check import run_check, set_golden_baseline
from dedrift.cli import app
from dedrift.config import (
    MAX_CUSTOM_SCALARS,
    RESERVED_SIGNATURE_NAMES,
    Materiality,
    ProjectConfig,
)
from dedrift.schema import (
    AgentConfig,
    InteractionInput,
    InteractionOutput,
    InteractionRecord,
    Source,
)
from dedrift.signatures import signatures_frame
from dedrift.signatures.structural import RATE_SIGNATURES, SCALAR_SIGNATURES
from dedrift.store import Store

CONFIG = AgentConfig(model="scorer-1", prompt_hash="p1", tool_schema_hash="t1", agent_version="v1")
SUITE = CanarySuite(
    version="nontext-v1",
    canaries=[Canary(id="c1", family="happy_path", input={"text": "score this"})],
)


def _record(canary: str, cycle: str, structured: dict[str, object] | None) -> InteractionRecord:
    """A minimal canary record carrying a structured payload."""
    return InteractionRecord(
        source=Source.CANARY,
        canary_id=canary,
        cycle_id=cycle,
        repetition=1,
        input=InteractionInput(text="score this"),
        output=InteractionOutput(text="0.42", structured=structured),
        config=CONFIG,
    )


class TestCustomScalarChannels:
    """Declared numeric channels ride the existing scalar battery."""

    def test_declared_channels_become_frame_columns(self) -> None:
        """The whole point: a numeric channel becomes testable, not just stored."""
        records = [
            _record("c1", "cycle-1", {"top_prob": 0.91, "entropy": 0.30}),
            _record("c2", "cycle-1", {"top_prob": 0.88, "entropy": 0.35}),
        ]
        frame = signatures_frame(records, custom_scalars=("top_prob", "entropy"))
        assert list(frame["top_prob"]) == [0.91, 0.88]
        assert list(frame["entropy"]) == [0.30, 0.35]

    def test_undeclared_channels_are_ignored(self) -> None:
        """Structured output is free-form; only declared keys are tested."""
        frame = signatures_frame([_record("c1", "cycle-1", {"top_prob": 0.9})])
        assert "top_prob" not in frame.columns

    @pytest.mark.parametrize("value", [None, "0.9", True, {"nested": 1}, []])
    def test_non_numeric_values_become_nan_not_a_guess(self, value: object) -> None:
        """A missing or wrong-typed reading must not be silently coerced.

        Booleans are excluded deliberately: ``isinstance(True, int)`` is true in
        Python, and letting a flag through as 1.0 would put a rate channel into
        the scalar battery under a name that claims to be continuous.
        """
        frame = signatures_frame(
            [_record("c1", "cycle-1", {"top_prob": value})], custom_scalars=("top_prob",)
        )
        assert frame["top_prob"].isna().all()

    def test_missing_structured_payload_is_nan(self) -> None:
        """An agent that returns no structured block must not crash the frame."""
        frame = signatures_frame([_record("c1", "cycle-1", None)], custom_scalars=("top_prob",))
        assert frame["top_prob"].isna().all()

    def test_channel_count_is_capped(self) -> None:
        """Each channel adds three primaries per family per baseline to the pool.

        Uncapped growth would raise the family-wise alert rate without anyone
        deciding to, which is the failure this project exists to prevent.
        """
        with pytest.raises(ValueError, match="cap is"):
            ProjectConfig(custom_scalars=tuple(f"m{i}" for i in range(MAX_CUSTOM_SCALARS + 1)))

    def test_channel_names_must_be_identifiers(self) -> None:
        with pytest.raises(ValueError, match="not a valid identifier"):
            ProjectConfig(custom_scalars=("not a name",))

    def test_duplicate_channels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            ProjectConfig(custom_scalars=("top_prob", "top_prob"))

    @pytest.mark.parametrize("name", ["latency_ms", "refusal", "semantic_displacement", "family"])
    def test_built_in_names_are_reserved(self, name: str) -> None:
        """A collision would retire a built-in channel without saying so."""
        with pytest.raises(ValueError, match="built-in signature name"):
            ProjectConfig(custom_scalars=(name,))

    def test_reserved_names_cover_the_shipped_battery(self) -> None:
        """The guard must track the battery, not a copy of it that drifts."""
        assert set(SCALAR_SIGNATURES) <= RESERVED_SIGNATURE_NAMES
        assert set(RATE_SIGNATURES) <= RESERVED_SIGNATURE_NAMES


class TestPerChannelMateriality:
    """Gates resolve per signature, so one channel can be retuned alone."""

    def test_override_applies_only_to_its_channel(self) -> None:
        """The reported failure: retuning latency desensitised every scalar."""
        m = Materiality(per_channel={"latency_ms": {"ks_distance": 0.35}})
        assert m.scalar_threshold("ks_distance", "latency_ms") == 0.35
        assert m.scalar_threshold("ks_distance", "output_words") == m.ks_distance

    def test_all_three_scalar_gates_are_overridable(self) -> None:
        m = Materiality(
            per_channel={
                "latency_ms": {"ks_distance": 0.35, "p95_relative": 0.5, "dispersion_ratio": 3.0}
            }
        )
        assert m.scalar_threshold("p95_relative", "latency_ms") == 0.5
        assert m.scalar_threshold("dispersion_ratio", "latency_ms") == 3.0

    def test_unknown_gate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not an overridable scalar gate"):
            Materiality(per_channel={"latency_ms": {"refusal_rate_pp": 5.0}})

    def test_override_ranges_are_validated(self) -> None:
        with pytest.raises(ValueError):
            Materiality(per_channel={"latency_ms": {"ks_distance": 1.5}})


class TestDeterministicAgents:
    """A repetition floor of 2 is not honest when variance is zero."""

    def test_single_repetition_allowed_when_declared(self) -> None:
        cfg = ProjectConfig(deterministic=True, canary_repetitions=1)
        assert cfg.canary_repetitions == 1

    def test_single_repetition_still_rejected_by_default(self) -> None:
        with pytest.raises(ValueError):
            ProjectConfig(canary_repetitions=1)

    def test_runner_accepts_one_repetition_for_a_deterministic_agent(self) -> None:
        runner = CanaryRunner(
            SUITE, lambda _: {"text": "0.42"}, CONFIG, repetitions=1, deterministic=True
        )
        assert runner.repetitions == 1

    def test_runner_still_refuses_one_repetition_otherwise(self) -> None:
        with pytest.raises(ValueError, match="deterministic"):
            CanaryRunner(SUITE, lambda _: {"text": "0.42"}, CONFIG, repetitions=1)


class TestConfigFileRoundTrip:
    """The documented TOML must actually load; config parsing is fail-closed."""

    @staticmethod
    def _write(tmp_path, body: str):
        project_dir = tmp_path / ".dedrift"
        project_dir.mkdir()
        (project_dir / "config.toml").write_text(body, encoding="utf-8")
        return project_dir

    def test_documented_toml_loads(self, tmp_path) -> None:
        cfg = ProjectConfig.load(
            self._write(
                tmp_path,
                """
[project]
deterministic = true
canary_repetitions = 1
custom_scalars = ["score", "top_prob"]

[materiality.per_channel.latency_ms]
ks_distance = 0.35
p95_relative = 0.5
""",
            )
        )
        assert cfg.deterministic is True
        assert cfg.canary_repetitions == 1
        assert cfg.custom_scalars == ("score", "top_prob")
        assert cfg.materiality.scalar_threshold("ks_distance", "latency_ms") == 0.35
        assert cfg.materiality.scalar_threshold("ks_distance", "score") == 0.15

    def test_non_string_channel_names_are_rejected_by_name(self, tmp_path) -> None:
        """A typo must name the key, not surface as an AttributeError."""
        with pytest.raises(ValueError, match="custom_scalars"):
            ProjectConfig.load(
                self._write(tmp_path, "[project]\ncustom_scalars = [1, 2]\n"),
            )

    def test_non_numeric_override_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            ProjectConfig.load(
                self._write(
                    tmp_path,
                    '[materiality.per_channel.latency_ms]\nks_distance = "high"\n',
                ),
            )


class TestDeclaredChannelEndToEnd:
    """A declared channel must reach the battery, not merely the frame."""

    @staticmethod
    def _seeded_store(tmp_path, shift: float) -> Store:
        """Six cycles of a numeric agent; the last one shifts by ``shift``."""
        rng = np.random.default_rng(20260816)
        store = Store.init_project(tmp_path)
        records = []
        for cycle in range(6):
            centre = 0.50 + (shift if cycle == 5 else 0.0)
            for canary in range(6):
                for rep in range(1, 6):
                    records.append(
                        InteractionRecord(
                            source=Source.CANARY,
                            canary_id=f"c{canary}",
                            cycle_id=f"cycle-{cycle:04d}",
                            repetition=rep,
                            input=InteractionInput(text=f"row {canary}"),
                            output=InteractionOutput(
                                text="0.500",
                                structured={"score": float(rng.normal(centre, 0.02))},
                            ),
                            config=CONFIG,
                        )
                    )
        store.append_many(records)
        set_golden_baseline(store, [f"cycle-{i:04d}" for i in range(3)])
        return store

    def test_declared_channel_can_alert(self, tmp_path) -> None:
        """The point of the feature: the numeric channel is what fires."""
        store = self._seeded_store(tmp_path, shift=0.20)
        result = run_check(store, ProjectConfig(custom_scalars=("score",)))
        assert "score" in {t.signature for t in result.alerts()}
        store.close()

    def test_declared_channel_is_quiet_on_a_stable_agent(self, tmp_path) -> None:
        """And it must not fire when nothing moved."""
        store = self._seeded_store(tmp_path, shift=0.0)
        result = run_check(store, ProjectConfig(custom_scalars=("score",)))
        assert "score" not in {t.signature for t in result.alerts()}
        store.close()

    def test_channel_is_invisible_until_declared(self, tmp_path) -> None:
        """Undeclared, the same shifted history produces no test at all."""
        store = self._seeded_store(tmp_path, shift=0.20)
        result = run_check(store, ProjectConfig())
        assert "score" not in {t.signature for t in result.tests}
        store.close()


class TestStoreFailsFast:
    """Opening an uninitialised project must fail before work is done."""

    def test_uninitialised_directory_raises_immediately(self, tmp_path) -> None:
        """It used to succeed here and die at the first write, losing a cycle."""
        (tmp_path / "scratch").mkdir()
        with pytest.raises(FileNotFoundError, match="dedrift init"):
            Store(tmp_path / "scratch")

    def test_cli_init_still_works_on_a_fresh_directory(self, tmp_path) -> None:
        """The fail-fast guard must not fire on the command that creates it.

        Every CLI command opens a store and then prints its own diagnosis, so
        the guard would have replaced `dedrift init` and every "no project
        here" message with a traceback.
        """
        result = CliRunner().invoke(app, ["init", str(tmp_path / "fresh")])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "fresh" / ".dedrift").is_dir()

    def test_cli_reports_a_missing_project_without_a_traceback(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        result = CliRunner().invoke(app, ["check", "--project", str(tmp_path / "empty")])
        assert result.exit_code == 1
        assert "dedrift init" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_init_project_still_works_on_an_empty_directory(self, tmp_path) -> None:
        store = Store.init_project(tmp_path / "fresh")
        assert store.records_path.parent.is_dir()
        assert Store(tmp_path / "fresh") is not None
