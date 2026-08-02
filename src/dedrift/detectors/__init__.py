"""Drift detectors (SPEC.md §6).

Every detector here reports its evidence honestly: hypothesis tests return
statistics, p-values, and effect sizes; PSI is a heuristic index and is never
presented as a p-value; Page-Hinkley alarms are flags with change-point
estimates. Multiplicity is handled downstream by Benjamini-Hochberg FDR
(:func:`benjamini_hochberg`) — raw per-test p-values must never be surfaced
as alerts.
"""

from dedrift.detectors.fdr import benjamini_hochberg
from dedrift.detectors.heuristic import psi, psi_null_expectation
from dedrift.detectors.mmd import calibrate_mmd_floor, mmd_rbf_test
from dedrift.detectors.scalar import (
    TestOutcome,
    ad_test,
    ks_test,
    levene_test,
    p95_permutation_test,
    two_proportion_z_test,
    welch_t_test,
)
from dedrift.detectors.sequential import PageHinkleyResult, page_hinkley

__all__ = [
    "PageHinkleyResult",
    "TestOutcome",
    "ad_test",
    "benjamini_hochberg",
    "calibrate_mmd_floor",
    "ks_test",
    "levene_test",
    "mmd_rbf_test",
    "p95_permutation_test",
    "page_hinkley",
    "psi",
    "psi_null_expectation",
    "two_proportion_z_test",
    "welch_t_test",
]
