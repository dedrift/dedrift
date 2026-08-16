"""Method adapters for the null-calibration benchmark.

Each adapter consumes one seeded stable-agent history and returns what the
method would have surfaced on it. Two entry points:

* :func:`run_percheck` — methods whose natural unit is one comparison of a
  current window against a reference window: folk-threshold PSI, dedrift's
  validity-guarded PSI, and naive per-check two-sample KS testing without
  multiplicity control.
* :func:`run_dedrift_trajectory` — dedrift's own two shipped paths, run as
  they are actually used in monitoring: one fixed-sample check and one
  anytime-valid fold per cycle over a 50-cycle monitored history.

Every method runs at its DEFAULT configuration; where a default is ambiguous
the resolution is documented on the function. No method is tuned.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from benchmark.histories import GOLDEN_CYCLES, PER_CHECK_CYCLES, SCALES, History, make_history
from dedrift.anytime import run_anytime_check
from dedrift.check import run_check, set_golden_baseline
from dedrift.config import ProjectConfig
from dedrift.detectors.heuristic import PSI_MAJOR, PSI_MODERATE, psi, psi_null_expectation
from dedrift.signatures import signatures_frame
from dedrift.signatures.structural import RATE_SIGNATURES, SCALAR_SIGNATURES
from dedrift.sim import SimAgent, SimConfig
from dedrift.store import Store

#: The signature columns shared by the table-based methods (PSI,
#: naive KS): the full structural battery, scalar channels first.
COLUMNS: tuple[str, ...] = tuple(SCALAR_SIGNATURES) + tuple(RATE_SIGNATURES)


def _frames(history: History) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Split a history's signature frame into reference and current windows.

    Returns:
        ``(ref_frame, cur_frame, families, usable_columns)`` where usable
        columns are those with at least one non-NaN value in both windows
        (an all-NaN column is untestable by every method, so it is dropped
        for all of them alike).
    """
    frame = signatures_frame(history.records)
    ref_f = frame[frame["cycle_id"].isin(history.golden)]
    cur_f = frame[frame["cycle_id"] == history.current]
    families = sorted(str(f) for f in frame["family"].unique())
    usable = [c for c in COLUMNS if ref_f[c].notna().any() and cur_f[c].notna().any()]
    return ref_f, cur_f, families, usable


def _psi_arm(ref_f: pd.DataFrame, cur_f: pd.DataFrame, families: list[str]) -> dict[str, int]:
    """Folk-threshold PSI and the same metric under dedrift's validity guard.

    PSI is computed with 10 quantile bins fixed from the reference window
    (dedrift's standards-compliant implementation is used for BOTH arms, so
    the comparison isolates the guard, not the implementation). The folk arm
    flags at PSI >= 0.1 ("moderate", the industry's tripwire) and is also
    recorded at >= 0.25 ("major"). The guarded arm first computes the
    first-order null expectation E[PSI] ~ (B-1)(1/n_ref + 1/n_cur) and
    refuses to emit the metric where sampling noise alone can reach half the
    moderate threshold — exactly dedrift's shipped behavior.
    """
    comparisons = 0
    flag_moderate = 0
    flag_major = 0
    guarded_emitted = 0
    guarded_flagged = 0
    for family in families:
        ref_fam = ref_f[ref_f["family"] == family]
        cur_fam = cur_f[cur_f["family"] == family]
        for sig in SCALAR_SIGNATURES:
            ref = ref_fam[sig].to_numpy(dtype=float)
            cur = cur_fam[sig].to_numpy(dtype=float)
            ref = ref[np.isfinite(ref)]
            cur = cur[np.isfinite(cur)]
            if len(ref) == 0 or len(cur) == 0:
                continue
            comparisons += 1
            value = psi(ref, cur).value
            flag_moderate += int(value >= PSI_MODERATE)
            flag_major += int(value >= PSI_MAJOR)
            if psi_null_expectation(len(ref), len(cur)) <= PSI_MODERATE / 2:
                guarded_emitted += 1
                guarded_flagged += int(value >= PSI_MODERATE)
    return {
        "psi_comparisons": comparisons,
        "psi_flag_moderate": flag_moderate,
        "psi_flag_major": flag_major,
        "psi_guarded_emitted": guarded_emitted,
        "psi_guarded_flagged": guarded_flagged,
    }


def _naive_ks_arm(
    ref_f: pd.DataFrame, cur_f: pd.DataFrame, families: list[str], columns: list[str]
) -> dict[str, Any]:
    """Raw two-sample KS at alpha=0.05 across the battery, no multiplicity control.

    The "roll your own monitoring" baseline: every (family, signature) pair
    gets a plain ``scipy.stats.ks_2samp`` and the run alerts if any p-value
    is below 0.05. Under exact, independent per-test calibration the
    predicted any-rejection rate is 1 - 0.95^k; the measured rate is what a
    team actually gets (discrete channels, correlated signatures).
    """
    tests = 0
    rejections = 0
    per_column: dict[str, list[int]] = {c: [0, 0] for c in columns}  # [rejections, tests]
    for family in families:
        ref_fam = ref_f[ref_f["family"] == family]
        cur_fam = cur_f[cur_f["family"] == family]
        for col in columns:
            ref = ref_fam[col].dropna().to_numpy(dtype=float)
            cur = cur_fam[col].dropna().to_numpy(dtype=float)
            if len(ref) == 0 or len(cur) == 0:
                continue
            tests += 1
            per_column[col][1] += 1
            if ks_2samp(ref, cur).pvalue < 0.05:
                rejections += 1
                per_column[col][0] += 1
    return {"ks_tests": tests, "ks_rejections": rejections, "ks_per_column": per_column}


def run_percheck(seed: int, scale: str) -> dict[str, Any]:
    """Run every per-comparison method on one seeded stable history.

    Args:
        seed: Master seed (run index).
        scale: Key of :data:`benchmark.histories.SCALES`.

    Returns:
        Per-run measurements for the PSI arms and naive KS. All flags are
        false alarms by construction.
    """
    history = make_history(seed, scale, PER_CHECK_CYCLES)
    ref_f, cur_f, families, usable = _frames(history)
    out: dict[str, Any] = {"seed": seed}
    out.update(_psi_arm(ref_f, cur_f, families))
    out.update(_naive_ks_arm(ref_f, cur_f, families, usable))

    return out


def run_dedrift_trajectory(seed: int, scale: str) -> dict[str, Any]:
    """Run dedrift's two shipped paths over one 50-cycle monitored history.

    The fixed-sample path (``dedrift check``: BH-FDR + materiality gates,
    dual baselines) and the anytime-valid path (e-processes + e-BH on the
    golden baseline, ``rate_model="twosample"``) both run at the package's
    DEFAULT configuration, cycle by cycle, exactly as in deployment. The
    per-cycle flag channel (PSI under the validity guard plus Page-Hinkley
    per stream) is recorded as well: flags are deliberately uncalibrated
    diagnostics, and their rate is published rather than hidden.

    Args:
        seed: Master seed (run index).
        scale: Key of :data:`benchmark.histories.SCALES`.

    Returns:
        Per-cycle alert/flag indicators for both paths, plus the per-check
        snapshot at the cycle whose history matches the per-check arm
        (cycle index 5).
    """
    n_canaries, repetitions = SCALES[scale]
    sim = SimConfig(n_canaries=n_canaries, repetitions=repetitions, change_cycle=None, seed=seed)
    agent = SimAgent(sim)
    cfg = ProjectConfig()
    fixed_alerts: list[int] = []
    fixed_flags: list[int] = []
    ph_flags: list[int] = []
    psi_flags: list[int] = []
    anytime_alerts: list[int] = []
    per_check: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp, Store.init_project(Path(tmp)) as store:
        store.append_many(agent.run_cycles(GOLDEN_CYCLES))
        golden = [f"cycle-{i:04d}" for i in range(GOLDEN_CYCLES)]
        set_golden_baseline(store, golden)
        for i in range(GOLDEN_CYCLES, GOLDEN_CYCLES + 50):
            store.append_many(agent.run_cycle(i))
            check = run_check(store, config=cfg)
            at_result = run_anytime_check(store, config=cfg)
            fixed_alerts.append(check.n_alerts)
            fixed_flags.append(len(check.flags))
            ph_flags.append(sum(1 for f in check.flags if f.kind == "page_hinkley"))
            psi_flags.append(sum(1 for f in check.flags if f.kind == "psi"))
            anytime_alerts.append(at_result.n_alerts)
            if i == PER_CHECK_CYCLES - 1:
                per_check = {
                    "alerts": check.n_alerts,
                    "flags": len(check.flags),
                    "ph_flags": ph_flags[-1],
                    "psi_flags": psi_flags[-1],
                }
    return {
        "seed": seed,
        "fixed_alerts": fixed_alerts,
        "fixed_flags": fixed_flags,
        "ph_flags": ph_flags,
        "psi_flags": psi_flags,
        "anytime_alerts": anytime_alerts,
        "per_check": per_check,
        "anytime_ever_alerted": any(a > 0 for a in anytime_alerts),
        "fixed_ever_alerted": any(a > 0 for a in fixed_alerts),
    }
