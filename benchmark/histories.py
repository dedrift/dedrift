"""History construction shared by every method in the benchmark.

A history is a seeded :class:`~dedrift.sim.SimAgent` run with
``change_cycle=None``: the configured stack never changes, so every alert
any method raises on it is a false alarm by construction. The first six
cycles of the 53-cycle history are byte-identical to the six-cycle history
at the same seed (the simulator's stream is sequential), which is what makes
the per-check and per-history estimands comparable across methods.

Scales:
    ``suite``: the documented default canary suite, 18 canaries x 7
        repetitions per cycle.
    ``small``: 12 canaries x 5 repetitions, the scale of dedrift's CI
        null-calibration gate, included to show scale-dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

from dedrift.schema import InteractionRecord
from dedrift.sim import SimAgent, SimConfig

#: Scale name -> (n_canaries, repetitions).
SCALES: dict[str, tuple[int, int]] = {
    "suite": (18, 7),
    "small": (12, 5),
}

#: Cycles generated for the per-check arm (3 golden + 3 monitored; the check
#: under test compares cycle 5 against cycles 0-2, matching dedrift's CI
#: null-calibration design).
PER_CHECK_CYCLES = 6

#: Cycles generated for the monitoring arm (3 golden + 50 monitored).
HISTORY_CYCLES = 53

#: Number of leading cycles frozen as the golden baseline.
GOLDEN_CYCLES = 3


@dataclass(frozen=True)
class History:
    """One seeded stable-agent history.

    Attributes:
        seed: The master seed (equal to the run index).
        scale: Scale name (key of :data:`SCALES`).
        records: All generated records, in execution order.
        cycles: Cycle IDs in execution order.
        golden: The frozen golden-baseline cycle IDs (first three).
        current: The cycle the per-check arm adjudicates.
    """

    seed: int
    scale: str
    records: list[InteractionRecord]
    cycles: list[str]
    golden: list[str]
    current: str


def make_history(seed: int, scale: str, n_cycles: int) -> History:
    """Generate one stable-agent history.

    Args:
        seed: Master seed; identical seeds give identical record streams.
        scale: Key of :data:`SCALES`.
        n_cycles: Number of cycles to run.

    Returns:
        The history with golden/current cycles identified.
    """
    n_canaries, repetitions = SCALES[scale]
    sim = SimConfig(n_canaries=n_canaries, repetitions=repetitions, change_cycle=None, seed=seed)
    records = SimAgent(sim).run_cycles(n_cycles)
    cycles = list(dict.fromkeys(r.cycle_id for r in records if r.cycle_id is not None))
    return History(
        seed=seed,
        scale=scale,
        records=records,
        cycles=cycles,
        golden=cycles[:GOLDEN_CYCLES],
        current=cycles[PER_CHECK_CYCLES - 1],
    )
