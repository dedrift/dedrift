"""Synthetic agent simulator (SPEC.md §9).

Generates :class:`~dedrift.schema.InteractionRecord` streams with controllable
behavioral parameters and a scriptable config change at a chosen cycle. This
module is the backbone of the README demo and of the CI calibration/power
tests: because ground truth is known by construction, detectors can be
validated against it.

All randomness flows from a single seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

import numpy as np

from dedrift.schema import (
    AgentConfig,
    InteractionInput,
    InteractionOutput,
    InteractionRecord,
    Source,
    ToolCall,
)

_WORDS = (
    "the", "model", "returns", "a", "structured", "answer", "with", "fields",
    "and", "values", "that", "describe", "the", "requested", "entity", "in",
    "plain", "language", "covering", "context", "caveats", "and", "sources",
)  # fmt: skip

_REFUSAL_TEXT = "I'm sorry, but I can't help with that request."


@dataclass(frozen=True)
class BehaviorProfile:
    """Controllable behavioral parameters of the synthetic agent.

    Attributes:
        mean_length_words: Mean output length in words (Poisson).
        refusal_prob: Probability an output is a refusal.
        format_error_prob: Probability a structured output is malformed.
        tool_call_rate: Mean number of tool calls per interaction (Poisson).
        schema_violation_prob: Probability a tool call has bad args.
        latency_mean_ms: Mean latency (lognormal location, in ms).
        latency_sigma: Lognormal sigma for latency dispersion.
    """

    mean_length_words: float = 60.0
    refusal_prob: float = 0.05
    format_error_prob: float = 0.02
    tool_call_rate: float = 1.5
    schema_violation_prob: float = 0.01
    latency_mean_ms: float = 1200.0
    latency_sigma: float = 0.4


@dataclass(frozen=True)
class SimConfig:
    """Scenario definition for a simulated run.

    A scripted config change occurs at ``change_cycle`` (0-indexed; cycles
    ``>= change_cycle`` use ``post`` behavior and the post-change agent
    config). If ``change_cycle`` is None, behavior never changes — this is
    the null scenario used for calibration tests.

    Attributes:
        n_canaries: Number of distinct canary inputs.
        repetitions: N repeated runs per canary per cycle.
        pre: Behavior before the change.
        post: Behavior after the change (ignored in the null scenario).
        change_cycle: Cycle index at which the config change takes effect.
        seed: Master seed; identical seeds give identical record streams
            (modulo record UUIDs and wall-clock-free timestamps, which are
            deterministic here too — time is simulated).
    """

    n_canaries: int = 30
    repetitions: int = 7
    pre: BehaviorProfile = field(default_factory=BehaviorProfile)
    post: BehaviorProfile = field(default_factory=BehaviorProfile)
    change_cycle: int | None = None
    seed: int = 1729


_PRE_AGENT_CONFIG = AgentConfig(
    model="simprovider/simmodel@v1",
    prompt_hash="sha256:" + "a" * 64,
    tool_schema_hash="sha256:" + "b" * 64,
    rag_index_version="rag-2026-01",
    agent_version="0.1.0",
)


def _post_agent_config() -> AgentConfig:
    return replace_model(_PRE_AGENT_CONFIG, "simprovider/simmodel@v2")


def replace_model(config: AgentConfig, model: str) -> AgentConfig:
    """Return a copy of ``config`` with a different model string."""
    return AgentConfig(
        model=model,
        prompt_hash=config.prompt_hash,
        tool_schema_hash=config.tool_schema_hash,
        rag_index_version=config.rag_index_version,
        agent_version=config.agent_version,
        extra=dict(config.extra),
    )


class SimAgent:
    """A deterministic (seeded) synthetic agent.

    Produces one :class:`InteractionRecord` per call according to the active
    :class:`BehaviorProfile`. Time is simulated: each generated record
    advances a virtual clock, so runs are fully reproducible.

    Args:
        sim_config: Scenario definition.
        start_time: Virtual timestamp of the first record.
    """

    def __init__(
        self,
        sim_config: SimConfig,
        start_time: datetime | None = None,
    ) -> None:
        self.sim_config = sim_config
        self.rng = np.random.default_rng(sim_config.seed)
        self.clock = start_time or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self._record_counter = 0

    def _family_for(self, canary_id: str) -> str:
        """Assign canaries to families round-robin (stable per canary)."""
        from dedrift.canary import FAMILIES

        index = int(canary_id.rsplit("-", 1)[-1])
        return FAMILIES[index % len(FAMILIES)]

    def _profile_for_cycle(self, cycle: int) -> BehaviorProfile:
        change = self.sim_config.change_cycle
        if change is not None and cycle >= change:
            return self.sim_config.post
        return self.sim_config.pre

    def _agent_config_for_cycle(self, cycle: int) -> AgentConfig:
        change = self.sim_config.change_cycle
        if change is not None and cycle >= change:
            return _post_agent_config()
        return _PRE_AGENT_CONFIG

    def _generate_output(self, profile: BehaviorProfile) -> tuple[InteractionOutput, bool]:
        refused = bool(self.rng.random() < profile.refusal_prob)
        if refused:
            return InteractionOutput(text=_REFUSAL_TEXT, structured=None), True
        n_words = max(3, int(self.rng.poisson(profile.mean_length_words)))
        words = self.rng.choice(_WORDS, size=n_words)
        text = " ".join(words.tolist())
        malformed = bool(self.rng.random() < profile.format_error_prob)
        structured = None if malformed else {"answer": text[:40], "confidence": 0.9}
        return InteractionOutput(text=text, structured=structured), False

    def _generate_tool_calls(self, profile: BehaviorProfile) -> list[ToolCall]:
        n = int(self.rng.poisson(profile.tool_call_rate))
        calls: list[ToolCall] = []
        for i in range(n):
            ok = bool(self.rng.random() >= profile.schema_violation_prob)
            name = str(self.rng.choice(["search", "lookup", "calculate"]))
            calls.append(ToolCall(name=name, args_schema_ok=ok, order=i + 1))
        return calls

    def run_one(self, canary_id: str, repetition: int, cycle: int) -> InteractionRecord:
        """Simulate one agent call for one canary repetition.

        Args:
            canary_id: Identifier of the canary input.
            repetition: 1-based repetition index within the cycle.
            cycle: 0-based cycle index (determines active behavior/config).

        Returns:
            A fully populated record with simulated timestamp and latency.
        """
        profile = self._profile_for_cycle(cycle)
        output, refused = self._generate_output(profile)
        tool_calls = [] if refused else self._generate_tool_calls(profile)
        latency = float(
            self.rng.lognormal(mean=np.log(profile.latency_mean_ms), sigma=profile.latency_sigma)
        )
        self.clock += timedelta(seconds=float(self.rng.uniform(1, 5)))
        self._record_counter += 1
        n_words_out = len(output.text.split())
        return InteractionRecord(
            id=f"sim-{self.sim_config.seed}-{self._record_counter:08d}",
            ts=self.clock,
            source=Source.CANARY,
            canary_id=canary_id,
            cycle_id=f"cycle-{cycle:04d}",
            repetition=repetition,
            input=InteractionInput(
                text=f"canary input {canary_id}",
                metadata={"family": self._family_for(canary_id)},
            ),
            output=output,
            tool_calls=tool_calls,
            steps=1 + len(tool_calls),
            retries=0,
            errors=[],
            latency_ms=int(latency),
            tokens_in=64,
            tokens_out=int(n_words_out * 1.3),
            config=self._agent_config_for_cycle(cycle),
        )

    def run_cycle(self, cycle: int) -> list[InteractionRecord]:
        """Run one full canary cycle (all canaries x N repetitions).

        Args:
            cycle: 0-based cycle index.

        Returns:
            Records in execution order.
        """
        records: list[InteractionRecord] = []
        for c in range(self.sim_config.n_canaries):
            canary_id = f"canary-{c:03d}"
            for rep in range(1, self.sim_config.repetitions + 1):
                records.append(self.run_one(canary_id, rep, cycle))
        return records

    def run_cycles(self, n_cycles: int) -> list[InteractionRecord]:
        """Run several consecutive cycles.

        Args:
            n_cycles: Number of cycles to simulate.

        Returns:
            All records in execution order.
        """
        records: list[InteractionRecord] = []
        for cycle in range(n_cycles):
            records.extend(self.run_cycle(cycle))
        return records


def drifted_profile(base: BehaviorProfile) -> BehaviorProfile:
    """Return a visibly drifted variant of ``base`` for demos.

    Longer outputs, higher refusal rate, worse formatting — the classic
    silent-degradation pattern.

    Args:
        base: The pre-change behavior.

    Returns:
        A shifted profile suitable as ``SimConfig.post``.
    """
    return replace(
        base,
        mean_length_words=base.mean_length_words * 1.5,
        refusal_prob=min(1.0, base.refusal_prob + 0.10),
        format_error_prob=min(1.0, base.format_error_prob + 0.05),
        latency_mean_ms=base.latency_mean_ms * 1.2,
    )
