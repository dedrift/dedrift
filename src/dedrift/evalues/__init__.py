"""Anytime-valid inference primitives: e-values and e-processes.

Addresses a limitation of the fixed-sample path rather than merely adding a
feature. A p-value pipeline controls error *per check*; repeated monitoring
needs a trajectory-level statement or its chance of at least one false alert
grows with use.

E-processes give each process a statement that holds at all stopping times.
The production driver uses persistent Ville crossings with a Bonferroni
split over its frozen pool. The e-BH implementation remains available as a
tested statistical primitive, but is not the production sequential decision
rule. Read :mod:`dedrift.evalues.base` first for the contracts (especially predictability), then
:mod:`dedrift.evalues.rates` for how the unknown null rate is handled and
what it costs, and :mod:`dedrift.evalues.process` for epoch semantics —
the target is alpha *per epoch*.

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
