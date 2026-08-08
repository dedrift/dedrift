"""Anytime-valid inference: e-values, e-processes, e-BH.

Fixes a defect in the fixed-sample path rather than adding a feature. The
p-value pipeline controls the false-alarm rate *per check*; operators run
checks forever, so at the measured 2.0% per-check rate an unchanged agent
accrues roughly ten false alerts a month. A monitoring tool whose error
guarantee decays with use has the wrong guarantee.

E-processes give each process a statement that holds at all stopping times.
The repeated dependent e-BH battery additionally needs the causal condition
documented in ``docs/anytime.md``; its trajectory-wide target is measured,
not asserted as an unconditional theorem. Read :mod:`dedrift.evalues.base`
first for the contracts (especially predictability), then
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
from dedrift.evalues.twosample import log_beta_binomial_evalue, twosample_rate_evalue

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
    "log_beta_binomial_evalue",
    "log_tilt_evalue",
    "rate_evalue",
    "symmetric_grid",
    "tilt_from_materiality",
    "twosample_rate_evalue",
    "update_process",
    "ville_threshold",
    "worst_case_log_evalue",
]
