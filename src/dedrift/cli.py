"""dedrift command-line interface.

Phases 0-1 expose ``init``, ``log``, ``sim``, ``canary run``, and
``signatures``. Later phases add ``check`` and ``report`` (see ROADMAP.md).
"""

from __future__ import annotations

import importlib
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from dedrift.schema import AgentConfig, InteractionRecord
from dedrift.sim import SimAgent, SimConfig, drifted_profile
from dedrift.store import Store

if TYPE_CHECKING:  # annotation only: keeps CLI start-up import-light
    from dedrift.config import ProjectConfig

app = typer.Typer(
    name="dedrift",
    help="Statistically rigorous behavioral drift detection for AI agents.",
    no_args_is_help=True,
)
canary_app = typer.Typer(help="Canary suite operations.", no_args_is_help=True)
app.add_typer(canary_app, name="canary")
baseline_app = typer.Typer(help="Golden baseline management.", no_args_is_help=True)
app.add_typer(baseline_app, name="baseline")
embedder_app = typer.Typer(help="Pinned-embedder management.", no_args_is_help=True)
app.add_typer(embedder_app, name="embedder")
cycle_app = typer.Typer(help="Canary-cycle lifecycle operations.", no_args_is_help=True)
app.add_typer(cycle_app, name="cycle")


def _load_project_config(store: Store) -> ProjectConfig:
    """Load a project config and turn validation errors into CLI errors."""
    from dedrift.config import ProjectConfig

    try:
        return ProjectConfig.load(store.project_dir)
    except (OSError, ValueError) as exc:
        typer.echo(f"Invalid {store.config_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _project_repetitions(store: Store, requested: int | None) -> int:
    """Return authoritative project repetitions, rejecting CLI conflicts."""
    configured = _load_project_config(store).canary_repetitions
    if requested is not None and requested != configured:
        typer.echo(
            f"--repetitions={requested} conflicts with project.canary_repetitions="
            f"{configured}; edit {store.config_path} to change the project design.",
            err=True,
        )
        raise typer.Exit(code=1)
    return configured


@embedder_app.command("pin")
def embedder_pin(
    name: Annotated[
        str, typer.Argument(help="Embedder id: 'hash' or 'st:<sentence-transformers model>'.")
    ],
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Pin the project's embedder FOREVER (enables Tier-2 semantic signatures)."""
    from dedrift.embeddings import EmbedderMismatchError, pin_embedder, resolve_embedder

    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    try:
        resolve_embedder(name)  # validate before pinning
        pin_embedder(store, name)
    except (EmbedderMismatchError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Embedder pinned: {name}. Changing it later invalidates all history.")


@embedder_app.command("show")
def embedder_show(
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Show the pinned embedder."""
    from dedrift.embeddings import get_pinned_embedder

    store = Store(path, require_project=False)
    pinned = get_pinned_embedder(store)
    typer.echo(pinned or "No embedder pinned (Tier-2 semantic signatures disabled).")


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Directory in which to create the project.")] = Path(
        "."
    ),
) -> None:
    """Initialize a dedrift project (.dedrift/ directory)."""
    store = Store(path, require_project=False)
    if store.exists():
        typer.echo(f"Project already initialized at {store.project_dir}")
        raise typer.Exit(code=0)
    with Store.init_project(path):
        pass
    typer.echo(f"Initialized dedrift project at {store.project_dir}")
    typer.echo("Edit .dedrift/config.toml to configure thresholds and the pinned embedder.")


@app.command()
def log(
    file: Annotated[Path, typer.Argument(help="JSONL file of InteractionRecords to ingest.")],
    finalize_cycles: Annotated[
        bool,
        typer.Option(
            "--finalize-cycles/--keep-cycles-open",
            help=(
                "Mark canary cycles in this file complete. The safe default keeps them "
                "open for streaming; finalize explicitly before checking."
            ),
        ),
    ] = False,
    expected_records: Annotated[
        int | None,
        typer.Option(help="Expected records in each canary cycle (validated on finalization)."),
    ] = None,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Ingest agent interaction records from a JSONL file."""
    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    if expected_records is not None and expected_records < 1:
        typer.echo("--expected-records must be >= 1", err=True)
        raise typer.Exit(code=1)
    records: list[InteractionRecord] = []
    try:
        with file.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(InteractionRecord.from_jsonl(stripped))
                except ValueError as exc:
                    typer.echo(f"Line {line_number}: invalid record: {exc}", err=True)
                    raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"Could not read {file}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    cycle_ids = sorted(
        {
            str(record.cycle_id)
            for record in records
            if record.source.value == "canary" and record.cycle_id is not None
        }
    )
    expected_counts = (
        {cycle_id: expected_records for cycle_id in cycle_ids}
        if expected_records is not None
        else None
    )
    try:
        with store:
            store.append_many(
                records,
                finalize_cycles=finalize_cycles,
                expected_cycle_counts=expected_counts,
            )
            total = store.count_records()
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Could not ingest records: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    lifecycle = "finalized" if finalize_cycles else "left open"
    typer.echo(
        f"Ingested {len(records)} records ({total} total); "
        f"{len(cycle_ids)} canary cycle(s) {lifecycle}."
    )


@cycle_app.command("finalize")
def cycle_finalize(
    cycle_id: Annotated[str, typer.Argument(help="Canary cycle ID to mark complete.")],
    expected_records: Annotated[
        int | None,
        typer.Option(help="Exact record count expected for this cycle."),
    ] = None,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Finalize a streamed canary cycle so checks may consume it."""
    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    try:
        with store:
            store.finalize_cycle(cycle_id, expected_records=expected_records)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Could not finalize cycle: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Finalized canary cycle {cycle_id!r}.")


@app.command()
def sim(
    cycles: Annotated[int, typer.Option(help="Number of canary cycles to simulate.")] = 10,
    change_cycle: Annotated[
        int | None,
        typer.Option(help="Cycle at which a scripted model swap shifts behavior (default: none)."),
    ] = None,
    canaries: Annotated[int, typer.Option(help="Number of distinct canaries.")] = 30,
    repetitions: Annotated[
        int | None,
        typer.Option(
            help=(
                "Compatibility check for project.canary_repetitions; omit to use the "
                "authoritative project value."
            )
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed (reproducible).")] = 1729,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Generate synthetic agent logs (with an optional scripted config change)."""
    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    if cycles < 1 or canaries < 1 or seed < 0 or (change_cycle is not None and change_cycle < 0):
        typer.echo(
            "--cycles and --canaries must be >= 1; --seed and --change-cycle must be >= 0",
            err=True,
        )
        raise typer.Exit(code=1)
    effective_repetitions = _project_repetitions(store, repetitions)
    config = SimConfig(
        n_canaries=canaries,
        repetitions=effective_repetitions,
        change_cycle=change_cycle,
        seed=seed,
    )
    if change_cycle is not None:
        config = SimConfig(
            n_canaries=canaries,
            repetitions=effective_repetitions,
            pre=config.pre,
            post=drifted_profile(config.pre),
            change_cycle=change_cycle,
            seed=seed,
        )
    with store:
        existing_cycles = {
            record.cycle_id
            for record in store.read_records()
            if record.source.value == "canary" and record.cycle_id is not None
        }
    numeric_cycles = [
        int(match.group(1))
        for cycle_id in existing_cycles
        if (match := re.fullmatch(r"cycle-(\d+)", cycle_id)) is not None
    ]
    start_cycle = max(numeric_cycles, default=-1) + 1
    # Fast-forward the seeded simulator so repeated CLI invocations are the
    # same deterministic history as one longer invocation (RNG, timestamps,
    # record IDs, and change-cycle semantics all remain aligned).
    agent = SimAgent(config)
    generated = agent.run_cycles(start_cycle + cycles)
    records = [
        record
        for record in generated
        if record.cycle_id is not None and int(record.cycle_id.rsplit("-", 1)[1]) >= start_cycle
    ]
    try:
        with store:
            expected_counts = {
                f"cycle-{cycle:04d}": canaries * effective_repetitions
                for cycle in range(start_cycle, start_cycle + cycles)
            }
            store.append_many(
                records,
                finalize_cycles=True,
                expected_cycle_counts=expected_counts,
            )
            events = store.config_events()
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Could not persist simulation: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Simulated {cycles} cycles ({start_cycle}..{start_cycle + cycles - 1}): "
        f"{len(records)} records."
    )
    typer.echo(f"Config events in timeline: {len(events)}.")
    if change_cycle is not None:
        typer.echo(f"Scripted model swap at cycle {change_cycle} (simmodel@v1 -> v2).")


def _load_agent_fn(spec: str) -> object:
    """Import an agent callable from a ``module:attribute`` spec."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        typer.echo(f"Invalid agent spec {spec!r}; expected 'module:function'.", err=True)
        raise typer.Exit(code=1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        typer.echo(f"Cannot load agent {spec!r}: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@canary_app.command("run")
def canary_run(
    suite: Annotated[Path, typer.Option(help="Canary suite YAML file.")],
    agent: Annotated[str, typer.Option(help="Agent callable as 'module:function' (importable).")],
    model: Annotated[str, typer.Option(help="Agent stack: model identifier.")],
    prompt_hash: Annotated[str, typer.Option(help="Agent stack: prompt hash.")] = "sha256:unset",
    tool_schema_hash: Annotated[
        str, typer.Option(help="Agent stack: tool schema hash.")
    ] = "sha256:unset",
    agent_version: Annotated[str, typer.Option(help="Agent stack: agent version.")] = "0",
    rag_index_version: Annotated[
        str | None, typer.Option(help="Agent stack: RAG index version.")
    ] = None,
    repetitions: Annotated[
        int | None,
        typer.Option(
            help=(
                "Compatibility check for project.canary_repetitions; omit to use the "
                "authoritative project value."
            )
        ),
    ] = None,
    cycles: Annotated[int, typer.Option(help="Number of cycles to run.")] = 1,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Run the canary suite N times per canary against your agent."""
    from dedrift.canary import CanaryRunner, CanarySuite

    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    if cycles < 1:
        typer.echo("--cycles must be >= 1", err=True)
        raise typer.Exit(code=1)
    effective_repetitions = _project_repetitions(store, repetitions)
    try:
        loaded_suite = CanarySuite.from_yaml(suite)
        agent_fn = _load_agent_fn(agent)
        agent_config = AgentConfig(
            model=model,
            prompt_hash=prompt_hash,
            tool_schema_hash=tool_schema_hash,
            rag_index_version=rag_index_version,
            agent_version=agent_version,
        )
    except typer.Exit:
        raise
    except (OSError, ValueError) as exc:
        typer.echo(f"Could not prepare canary run: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    runner = CanaryRunner(
        loaded_suite,
        agent_fn,  # type: ignore[arg-type]
        agent_config,
        repetitions=effective_repetitions,
        deterministic=_load_project_config(store).deterministic,
    )
    try:
        with store:
            for _ in range(cycles):
                records = runner.run_cycle(store=store)
                typer.echo(f"Cycle {records[0].cycle_id}: {len(records)} records.")
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Could not persist canary cycle: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Ran {cycles} cycle(s) of suite v{loaded_suite.version} "
        f"({len(loaded_suite.canaries)} canaries x {effective_repetitions} repetitions)."
    )


@app.command()
def signatures(
    by: Annotated[str, typer.Option(help="Aggregation: 'family' or 'canary'.")] = "family",
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Show Tier-1 signature tables aggregated per cycle."""
    import pandas as pd

    from dedrift.signatures import (
        aggregate_by_canary_cycle,
        aggregate_by_family_cycle,
        signatures_frame,
    )

    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    with store:
        records, _snapshot_offset = store.read_finalized_canary_snapshot()
    if not records:
        typer.echo("No canary records with cycle IDs found.")
        raise typer.Exit(code=0)
    custom = _load_project_config(store).custom_scalars
    frame = signatures_frame(records, custom_scalars=tuple(custom))
    if by == "canary":
        table = aggregate_by_canary_cycle(frame)
    elif by == "family":
        table = aggregate_by_family_cycle(frame)
    else:
        typer.echo(f"Unknown aggregation {by!r}; use 'family' or 'canary'.", err=True)
        raise typer.Exit(code=1)
    key_cols = ["family", "cycle_id"] if by == "family" else ["canary_id", "cycle_id"]
    show = [
        *key_cols,
        "n",
        "output_words_mean",
        "output_words_var",
        "output_words_p95",
        "latency_ms_mean",
        "latency_ms_p95",
        "refusal_rate",
        "format_valid_rate",
        # Declared channels last: for a numeric agent they are the only
        # columns above that can move, so they must not be cut off.
        *[f"{name}_{stat}" for name in custom for stat in ("mean", "p95")],
    ]
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        typer.echo(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


@baseline_app.command("set")
def baseline_set(
    cycle_ids: Annotated[
        list[str] | None,
        typer.Argument(help="Cycle IDs to freeze as the golden baseline."),
    ] = None,
    last: Annotated[
        int | None, typer.Option(help="Freeze the last N completed cycles instead.")
    ] = None,
    first: Annotated[
        int | None, typer.Option(help="Freeze the first N cycles (the known-good era).")
    ] = None,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Freeze known-good cycles as the golden baseline (never auto-updated)."""
    from dedrift.check import get_golden_baseline, set_golden_baseline
    from dedrift.signatures import signatures_frame

    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    with store:
        selectors = int(bool(cycle_ids)) + int(last is not None) + int(first is not None)
        if selectors != 1:
            typer.echo("Choose exactly one: cycle IDs, --first N, or --last N.", err=True)
            raise typer.Exit(code=1)
        if (last is not None and last < 1) or (first is not None and first < 1):
            typer.echo("--first and --last must be >= 1.", err=True)
            raise typer.Exit(code=1)
        if last is not None or first is not None:
            records, _snapshot_offset = store.read_finalized_canary_snapshot()
            if not records:
                typer.echo("No canary cycles found.", err=True)
                raise typer.Exit(code=1)
            frame = signatures_frame(records)
            cycles = list(dict.fromkeys(frame["cycle_id"]))
            chosen = cycles[:first] if first is not None else cycles[-(last or 0) :]
        else:
            assert cycle_ids
            chosen = list(cycle_ids)
        try:
            set_golden_baseline(store, chosen)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
            typer.echo(f"Could not set golden baseline: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        frozen = get_golden_baseline(store)
        typer.echo(f"Golden baseline frozen: {frozen}")
        # Golden cycles are excluded from the rolling reference, so freezing
        # everything starves the sudden channel. Its only symptom is a
        # persistent "NO REFERENCE", which reads like "not enough history yet"
        # rather than "you consumed the history you had".
        try:
            records, _ = store.read_finalized_canary_snapshot()
            completed = list(dict.fromkeys(signatures_frame(records)["cycle_id"]))
        except (OSError, RuntimeError, ValueError, sqlite3.Error, KeyError):
            completed = []
        remaining = [c for c in completed if c not in set(frozen)]
        if completed and not remaining:
            typer.echo(
                f"WARNING: the golden baseline now holds all {len(completed)} completed "
                f"cycles, so the rolling window has none left to compare against and the "
                f"sudden channel will report NO REFERENCE until new cycles arrive.",
                err=True,
            )
        elif completed and len(remaining) < 2:
            typer.echo(
                f"WARNING: only {len(remaining)} completed cycle(s) remain outside the "
                f"golden baseline; the rolling window needs more before the sudden "
                f"channel becomes informative.",
                err=True,
            )


@app.command()
def check(
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
    inference: Annotated[
        str,
        typer.Option(
            "--inference",
            help=(
                "fixed = per-check FDR (default, the reference implementation); "
                "anytime = lifetime-oriented e-processes (rate channel only; "
                "trajectory-wide control has a documented causal assumption)."
            ),
        ),
    ] = "",
) -> None:
    """Run the gated drift check for the latest cycle (dual baselines)."""
    from dedrift.check import run_check

    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    cfg = _load_project_config(store)
    mode = inference or cfg.inference
    if mode not in ("fixed", "anytime"):
        typer.echo(f"--inference must be 'fixed' or 'anytime', got {mode!r}", err=True)
        raise typer.Exit(code=1)
    if mode == "anytime":
        _check_anytime(store, cfg)
        return
    try:
        with store:
            result = run_check(store, config=cfg)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Could not run check: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Current cycle: {result.current_cycle}")
    typer.echo(f"Overall: {result.overall_verdict}")
    typer.echo(f"Sudden (vs rolling {len(result.rolling_cycles)} cycles): {result.verdict_sudden}")
    typer.echo(
        f"Cumulative (vs golden {len(result.golden_cycles)} cycles): {result.verdict_cumulative}"
    )
    if result.degraded:
        typer.echo("DEGRADED DATA: current cycle error rate too high for drift analysis.")
    for c in result.composition_issues:
        typer.echo(
            f"COMPOSITION MISMATCH [{c.baseline}] {c.family}: {c.detail} "
            "(comparison suppressed — not drift)"
        )
    typer.echo(
        f"Alerts: {result.n_alerts} "
        f"(BH-adjusted equality tests q={result.fdr_q}, observed-effect gated)"
    )
    for a in result.alerts()[:10]:
        typer.echo(
            f"  [{a.baseline}] {a.family}/{a.signature} {a.outcome.test}: "
            f"effect={a.outcome.effect_size:+.3f}, p_adj={a.p_adjusted:.4g}"
        )
    if result.overall_verdict == "DRIFT DETECTED":
        raise typer.Exit(code=2)
    if result.overall_verdict != "OK":
        raise typer.Exit(code=3)


def _check_anytime(store: Store, cfg: ProjectConfig) -> None:
    """Render the anytime-valid check. Exit 2 on drift, matching `fixed`."""
    from dedrift.anytime import run_anytime_check

    try:
        with store:
            res = run_anytime_check(store, cfg)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Could not run anytime check: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Current cycle: {res.current_cycle}   epoch fingerprint {res.fingerprint}")
    typer.echo(
        f"Anytime-valid: alpha={res.alpha} = alpha'({res.alpha_prime}) "
        f"+ gamma_total({res.gamma_total}); {res.n_processes} e-processes, "
        f"gamma per process {res.gamma_per_process:.2e}"
    )
    typer.echo(f"Verdict: {res.verdict}")
    typer.echo(f"Rate-channel coverage: {res.coverage_status}")
    if res.suppressed_families:
        typer.echo("Uncovered/suppressed families: " + ", ".join(res.suppressed_families))
    for notice in res.resets:
        typer.echo(f"  RESET {notice}")
    if res.degraded:
        typer.echo("DEGRADED DATA: current cycle error rate too high for drift analysis.")
    typer.echo(f"Alerts: {res.n_alerts} (e-BH at q={res.alpha_prime})")
    for p in res.alerts()[:10]:
        typer.echo(f"  {p.label}: log-wealth={p.log_wealth:.2f}, onset~cycle {p.rise_cycle}")
    top = sorted(res.processes, key=lambda x: -x.log_wealth)[:3]
    if top and not res.alerts():
        typer.echo("Highest wealth (no alert):")
        for p in top:
            typer.echo(f"  {p.label}: log-wealth={p.log_wealth:.2f} over {p.cycles} cycles")
    if res.n_processes:
        typer.echo(
            "Target: P(ever alerting on a stable agent) <= alpha, PER EPOCH. "
            "Per-process control is proven; repeated battery-wide e-BH relies "
            "on the documented causal assumption."
        )
    else:
        typer.echo("Anytime error-control target not active: no valid processes.")
    if res.n_alerts:
        raise typer.Exit(code=2)
    if res.degraded or res.verdict != "OK" or res.n_processes == 0:
        raise typer.Exit(code=3)


@app.command()
def report(
    output: Annotated[
        Path | None, typer.Option("--out", help="Write markdown here (default: stdout).")
    ] = None,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
    inference: Annotated[
        str,
        typer.Option(
            "--inference",
            help="fixed (p-values, default) or anytime (lifetime-oriented rate e-processes).",
        ),
    ] = "",
) -> None:
    """Run a check and render the full markdown report."""
    from dedrift.check import run_check
    from dedrift.report import render_anytime_report, render_report

    store = Store(path, require_project=False)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    cfg = _load_project_config(store)
    mode = inference or cfg.inference
    if mode not in ("fixed", "anytime"):
        typer.echo(f"--inference must be 'fixed' or 'anytime', got {mode!r}", err=True)
        raise typer.Exit(code=1)
    try:
        with store:
            if mode == "anytime":
                from dedrift.anytime import run_anytime_check

                result_anytime = run_anytime_check(store, cfg)
                markdown = render_anytime_report(result_anytime)
                exit_code = (
                    2 if result_anytime.n_alerts else (0 if result_anytime.verdict == "OK" else 3)
                )
            else:
                result = run_check(store, config=cfg)
                markdown = render_report(store, result)
                exit_code = (
                    2
                    if result.overall_verdict == "DRIFT DETECTED"
                    else (0 if result.overall_verdict == "OK" else 3)
                )
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        typer.echo(f"Could not render report: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if output is None:
        typer.echo(markdown)
    else:
        output.write_text(markdown, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    if exit_code:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
