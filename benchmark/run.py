"""Runner for the null-calibration benchmark.

Usage:
    python -m benchmark.run --all                 # full study, both scales
    python -m benchmark.run percheck --scale suite --runs 500 --workers 12
    python -m benchmark.run dedrift --scale small --runs 500 --workers 12
    python -m benchmark.run --all --quick         # smoke run (20 seeds)

Legs:
    ``percheck``: PSI (folk and validity-guarded), naive uncontrolled KS,
        and Evidently's default DataDriftPreset at pooled and per-family
        granularity — one comparison per run, on the 6-cycle history.
    ``dedrift``: dedrift's fixed-sample and anytime-valid paths, run
        cycle-by-cycle over the 50-cycle monitored history, at the shipped
        default configuration.

Every leg writes one JSON document to ``benchmark/results/`` containing the
full configuration block, per-method summaries with Wilson intervals, and
the per-run rows. Seeds come from ``benchmark/seeds.txt`` (0..499) so the
seed list ships with the results.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.histories import GOLDEN_CYCLES, HISTORY_CYCLES, PER_CHECK_CYCLES, SCALES
from benchmark.stats import rate_row

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEEDS_FILE = ROOT / "seeds.txt"


def read_seeds() -> list[int]:
    """Return the shipped seed list (one seed per line)."""
    return [int(line) for line in SEEDS_FILE.read_text().split() if line.strip()]


def _versions() -> dict[str, str]:
    """Pin every version that can move a number in this study."""
    import numpy
    import pandas
    import scipy

    import dedrift

    versions = {
        "dedrift": dedrift.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
    }
    try:
        import evidently

        versions["evidently"] = evidently.__version__
    except ImportError:
        versions["evidently"] = "not installed"
    return versions


def _config_block(leg: str, scale: str, seeds: list[int]) -> dict[str, Any]:
    """The provenance header written into every results document."""
    n_canaries, repetitions = SCALES[scale]
    return {
        "leg": leg,
        "scale": scale,
        "n_canaries": n_canaries,
        "repetitions": repetitions,
        "golden_cycles": GOLDEN_CYCLES,
        "cycles": PER_CHECK_CYCLES if leg == "percheck" else HISTORY_CYCLES,
        "monitored_cycles": 1 if leg == "percheck" else HISTORY_CYCLES - GOLDEN_CYCLES,
        "change": None,
        "n_runs": len(seeds),
        "seed_file": "benchmark/seeds.txt",
        "seed_first": seeds[0],
        "seed_last": seeds[-1],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
    }


def _map_runs(fn: Any, scale: str, seeds: list[int], workers: int) -> list[dict[str, Any]]:
    """Run ``fn(seed, scale)`` over seeds in a process pool, in seed order."""
    if workers <= 1:
        return [fn(seed, scale) for seed in seeds]
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, seeds, [scale] * len(seeds), chunksize=4))


def aggregate_percheck(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-run percheck rows to method summaries with intervals."""
    n = len(rows)
    psi_cmp = sum(r["psi_comparisons"] for r in rows)
    psi_mod = sum(r["psi_flag_moderate"] for r in rows)
    psi_maj = sum(r["psi_flag_major"] for r in rows)
    guard_emitted = sum(r["psi_guarded_emitted"] for r in rows)
    guard_flagged = sum(r["psi_guarded_flagged"] for r in rows)
    ks_tests = sum(r["ks_tests"] for r in rows)
    ks_rej = sum(r["ks_rejections"] for r in rows)
    ev_pool_cols = sum(r["ev_pooled"]["columns"] for r in rows)
    ev_pool_drifted = sum(r["ev_pooled"]["drifted"] for r in rows)
    ev_fam_cols = sum(r["ev_family"]["columns_total"] for r in rows)
    ev_fam_drifted = sum(r["ev_family"]["drifted_total"] for r in rows)

    per_column: dict[str, list[int]] = {}
    for r in rows:
        for col, (rej, tests) in r["ks_per_column"].items():
            slot = per_column.setdefault(col, [0, 0])
            slot[0] += rej
            slot[1] += tests

    ev_columns: dict[str, dict[str, Any]] = {}
    for r in rows:
        for col, cell in r["ev_pooled"]["per_column"].items():
            slot = ev_columns.setdefault(col, {"drifted": 0, "n": 0, "methods": set()})
            slot["drifted"] += int(cell["drifted"])
            slot["n"] += 1
            slot["methods"].add(cell["method"])
    ev_per_column = {
        col: {**rate_row(slot["drifted"], slot["n"]), "methods": sorted(slot["methods"])}
        for col, slot in sorted(ev_columns.items())
    }
    ks_per_run = sorted(r["ks_tests"] for r in rows)
    k_median = ks_per_run[n // 2]
    return {
        "psi_folk": {
            "description": "PSI, 10 reference-quantile bins, folk thresholds, no guard",
            "comparisons": psi_cmp,
            "flag_moderate": rate_row(psi_mod, psi_cmp),
            "flag_major": rate_row(psi_maj, psi_cmp),
            "runs_any_moderate": rate_row(sum(r["psi_flag_moderate"] > 0 for r in rows), n),
            "runs_any_major": rate_row(sum(r["psi_flag_major"] > 0 for r in rows), n),
        },
        "psi_guarded": {
            "description": "identical PSI under dedrift's domain-of-validity guard",
            "comparisons": psi_cmp,
            "emitted": rate_row(guard_emitted, psi_cmp),
            "flag_moderate": rate_row(guard_flagged, psi_cmp),
            "runs_any_flag": rate_row(sum(r["psi_guarded_flagged"] > 0 for r in rows), n),
        },
        "naive_ks": {
            "description": "two-sample KS per (family, signature), alpha=0.05, no control",
            "tests": ks_tests,
            "rejections": rate_row(ks_rej, ks_tests),
            "runs_any_rejection": rate_row(sum(r["ks_rejections"] > 0 for r in rows), n),
            "tests_per_run_median": k_median,
            "predicted_any_rejection_independent": 1.0 - 0.95**k_median,
            "per_column": {
                col: rate_row(rej, tests) for col, (rej, tests) in sorted(per_column.items())
            },
        },
        "evidently_pooled": {
            "description": "DataDriftPreset, all defaults, one report over the pooled table",
            "columns": ev_pool_cols,
            "per_column_flag": rate_row(ev_pool_drifted, ev_pool_cols),
            "runs_any_drifted_column": rate_row(
                sum(r["ev_pooled"]["drifted"] > 0 for r in rows), n
            ),
            "runs_dataset_drift": rate_row(sum(r["ev_pooled"]["dataset_drift"] for r in rows), n),
            "per_column": ev_per_column,
        },
        "evidently_family": {
            "description": "DataDriftPreset, all defaults, one report per canary family",
            "columns": ev_fam_cols,
            "per_column_flag": rate_row(ev_fam_drifted, ev_fam_cols),
            "runs_any_drifted_column": rate_row(
                sum(r["ev_family"]["any_drifted"] for r in rows), n
            ),
        },
    }


def aggregate_dedrift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce per-run trajectory rows to method summaries with intervals."""
    n = len(rows)
    n_checks = sum(len(r["fixed_alerts"]) for r in rows)
    fixed_alert_checks = sum(sum(a > 0 for a in r["fixed_alerts"]) for r in rows)
    flag_checks = sum(sum(f > 0 for f in r["fixed_flags"]) for r in rows)
    ph_checks = sum(sum(p > 0 for p in r["ph_flags"]) for r in rows)
    psi_checks = sum(sum(p > 0 for p in r["psi_flags"]) for r in rows)
    at_folds = sum(len(r["anytime_alerts"]) for r in rows)
    at_alert_folds = sum(sum(a > 0 for a in r["anytime_alerts"]) for r in rows)
    return {
        "dedrift_fixed_percheck": {
            "description": "dedrift check (BH-FDR + materiality), cycle-5 history: "
            "the same comparison the percheck leg adjudicates",
            "runs_any_alert": rate_row(sum(r["per_check"]["alerts"] > 0 for r in rows), n),
            "runs_any_flag": rate_row(sum(r["per_check"]["flags"] > 0 for r in rows), n),
            "runs_ph_flag": rate_row(sum(r["per_check"]["ph_flags"] > 0 for r in rows), n),
            "runs_psi_flag": rate_row(sum(r["per_check"]["psi_flags"] > 0 for r in rows), n),
        },
        "dedrift_fixed_cumulative": {
            "description": "dedrift check per cycle over a 50-cycle monitored history",
            "checks": n_checks,
            "per_check_alert": rate_row(fixed_alert_checks, n_checks),
            "per_check_any_flag": rate_row(flag_checks, n_checks),
            "per_check_ph_flag": rate_row(ph_checks, n_checks),
            "per_check_psi_flag": rate_row(psi_checks, n_checks),
            "runs_ever_alerted_50_cycles": rate_row(sum(r["fixed_ever_alerted"] for r in rows), n),
        },
        "dedrift_anytime": {
            "description": "anytime-valid path (e-processes + e-BH, golden, twosample "
            "rate model), lifetime alpha=0.05, folded per cycle",
            "folds": at_folds,
            "per_fold_alert": rate_row(at_alert_folds, at_folds),
            "runs_ever_alerted_50_cycles": rate_row(
                sum(r["anytime_ever_alerted"] for r in rows), n
            ),
        },
    }


def run_leg(leg: str, scale: str, seeds: list[int], workers: int) -> Path:
    """Execute one (leg, scale) cell and write its results document."""
    from benchmark import methods

    fn = methods.run_percheck if leg == "percheck" else methods.run_dedrift_trajectory
    aggregate = aggregate_percheck if leg == "percheck" else aggregate_dedrift
    t0 = time.time()
    rows = _map_runs(fn, scale, seeds, workers)
    document = {
        "config": _config_block(leg, scale, seeds),
        "methods": aggregate(rows),
        "runs": rows,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{leg}_{scale}.json"
    out.write_text(json.dumps(document, indent=1))
    print(
        f"[{leg}/{scale}] {len(rows)} runs in {document['runtime_seconds']}s -> {out}", flush=True
    )
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legs", nargs="*", choices=["percheck", "dedrift"], help="legs to run")
    parser.add_argument("--all", action="store_true", help="run every leg at both scales")
    parser.add_argument("--scale", choices=[*SCALES, "both"], default="both")
    parser.add_argument("--runs", type=int, default=None, help="first N seeds (default: all)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--quick", action="store_true", help="smoke run: 20 seeds")
    args = parser.parse_args()

    seeds = read_seeds()
    if args.quick:
        seeds = seeds[:20]
    if args.runs is not None:
        seeds = seeds[: args.runs]
    legs = args.legs or (["percheck", "dedrift"] if args.all else [])
    if not legs:
        parser.error("name at least one leg or pass --all")
    scales = list(SCALES) if args.scale == "both" else [args.scale]
    for leg in legs:
        for scale in scales:
            run_leg(leg, scale, seeds, args.workers)


if __name__ == "__main__":
    main()
