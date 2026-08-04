"""Anytime-valid inference: e-values, e-processes, e-BH.

Fixes a defect in the fixed-sample path rather than adding a feature. The
p-value pipeline controls the false-alarm rate *per check*; operators run
checks forever, so at the measured 1.6% per-check rate an unchanged agent
accrues roughly ten false alerts a month. A monitoring tool whose error
guarantee decays with use has the wrong guarantee.

E-processes replace it with a statement that holds at all stopping times:
over an unbounded horizon, the probability of ever falsely alerting on a
stable agent is at most alpha. Read :mod:`dedrift.evalues.base` first for
the contracts (especially predictability), then
:mod:`dedrift.evalues.rates` for how the unknown null rate is handled and
what it costs, and :mod:`dedrift.evalues.process` for epoch semantics —
the guarantee is alpha *per epoch*, and that is the correct reading, not a
weakened one.

The p-value path is not deleted. It is the reference implementation, and
its measured cumulative false-alarm growth is the evidence motivating this
one.
"""

from dedrift.evalues.base import EValueOutcome, PriorState
from dedrift.evalues.ebh import EBHResult, ebh
from dedrift.evalues.process import (
    EProcessState,
    EProcessUpdate,
    epoch_fingerprint,
    geometric_allocation,
    update_process,
    ville_threshold,
)
from dedrift.evalues.rates import (
    clopper_pearson,
    log_tilt_evalue,
    rate_evalue,
    symmetric_grid,
    tilt_from_materiality,
    worst_case_log_evalue,
)

__all__ = [
    "EBHResult",
    "EProcessState",
    "EProcessUpdate",
    "EValueOutcome",
    "PriorState",
    "clopper_pearson",
    "ebh",
    "epoch_fingerprint",
    "geometric_allocation",
    "log_tilt_evalue",
    "rate_evalue",
    "symmetric_grid",
    "tilt_from_materiality",
    "update_process",
    "ville_threshold",
    "worst_case_log_evalue",
]
