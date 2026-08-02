"""Drift detectors (SPEC.md §6).

Every detector here reports its evidence honestly: hypothesis tests return
statistics, p-values, and effect sizes; PSI is a heuristic index and is never
presented as a p-value; Page-Hinkley alarms are flags with change-point
estimates. Multiplicity is handled downstream by Benjamini-Hochberg FDR
(:func:`benjamini_hochberg`) — raw per-test p-values must never be surfaced
as alerts.
"""

from dedrift.detectors.fdr import benjamini_hochberg
from dedrift.detectors.heuristic import psi
from dedrift.detectors.scalar import (
    TestOutcome,
    ad_test,
    bootstrap_p95_test,
    ks_test,
    levene_test,
    two_proportion_z_test,
    welch_t_test,
)
from dedrift.detectors.sequential import PageHinkleyResult, page_hinkley

__all__ = [
    "PageHinkleyResult",
    "TestOutcome",
    "ad_test",
    "benjamini_hochberg",
    "bootstrap_p95_test",
    "ks_test",
    "levene_test",
    "page_hinkley",
    "psi",
    "two_proportion_z_test",
    "welch_t_test",
]
