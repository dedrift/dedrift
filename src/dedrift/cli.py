"""dedrift command-line interface.

Phases 0-1 expose ``init``, ``log``, ``sim``, ``canary run``, and
``signatures``. Later phases add ``check`` and ``report`` (see ROADMAP.md).
"""

from __future__ import annotations

import importlib
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


@embedder_app.command("pin")
def embedder_pin(
    name: Annotated[
        str, typer.Argument(help="Embedder id: 'hash' or 'st:<sentence-transformers model>'.")
    ],
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Pin the project's embedder FOREVER (enables Tier-2 semantic signatures)."""
    from dedrift.embeddings import EmbedderMismatchError, pin_embedder, resolve_embedder

    store = Store(path)
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

    store = Store(path)
    pinned = get_pinned_embedder(store)
    typer.echo(pinned or "No embedder pinned (Tier-2 semantic signatures disabled).")


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Directory in which to create the project.")] = Path(
        "."
    ),
) -> None:
    """Initialize a dedrift project (.dedrift/ directory)."""
    store = Store(path)
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
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Ingest agent interaction records from a JSONL file."""
    store = Store(path)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    records: list[InteractionRecord] = []
    with file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(InteractionRecord.from_jsonl(stripped))
            except ValueError as exc:
                typer.echo(f"Line {i}: invalid record: {exc}", err=True)
                raise typer.Exit(code=1) from exc
    with store:
        store.append_many(records)
        total = store.count_records()
    typer.echo(f"Ingested {len(records)} records ({total} total).")


@app.command()
def sim(
    cycles: Annotated[int, typer.Option(help="Number of canary cycles to simulate.")] = 10,
    change_cycle: Annotated[
        int | None,
        typer.Option(help="Cycle at which a scripted model swap shifts behavior (default: none)."),
    ] = None,
    canaries: Annotated[int, typer.Option(help="Number of distinct canaries.")] = 30,
    repetitions: Annotated[int, typer.Option(help="Repetitions per canary per cycle.")] = 7,
    seed: Annotated[int, typer.Option(help="Random seed (reproducible).")] = 1729,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Generate synthetic agent logs (with an optional scripted config change)."""
    store = Store(path)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    config = SimConfig(
        n_canaries=canaries,
        repetitions=repetitions,
        change_cycle=change_cycle,
        seed=seed,
    )
    if change_cycle is not None:
        config = SimConfig(
            n_canaries=canaries,
            repetitions=repetitions,
            pre=config.pre,
            post=drifted_profile(config.pre),
            change_cycle=change_cycle,
            seed=seed,
        )
    agent = SimAgent(config)
    records = agent.run_cycles(cycles)
    with store:
        store.append_many(records)
        events = store.config_events()
    typer.echo(f"Simulated {cycles} cycles: {len(records)} records.")
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
    repetitions: Annotated[int, typer.Option(help="N repetitions per canary.")] = 7,
    cycles: Annotated[int, typer.Option(help="Number of cycles to run.")] = 1,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Run the canary suite N times per canary against your agent."""
    from dedrift.canary import CanaryRunner, CanarySuite

    store = Store(path)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    loaded_suite = CanarySuite.from_yaml(suite)
    agent_fn = _load_agent_fn(agent)
    config = AgentConfig(
        model=model,
        prompt_hash=prompt_hash,
        tool_schema_hash=tool_schema_hash,
        rag_index_version=rag_index_version,
        agent_version=agent_version,
    )
    runner = CanaryRunner(
        loaded_suite,
        agent_fn,  # type: ignore[arg-type]
        config,
        repetitions=repetitions,
    )
    with store:
        for _ in range(cycles):
            records = runner.run_cycle(store=store)
            typer.echo(f"Cycle {records[0].cycle_id}: {len(records)} records.")
    typer.echo(
        f"Ran {cycles} cycle(s) of suite v{loaded_suite.version} "
        f"({len(loaded_suite.canaries)} canaries x {repetitions} repetitions)."
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

    store = Store(path)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    with store:
        records = [r for r in store.read_records() if r.cycle_id is not None]
    if not records:
        typer.echo("No canary records with cycle IDs found.")
        raise typer.Exit(code=0)
    frame = signatures_frame(records)
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

    store = Store(path)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    with store:
        if last is not None or first is not None:
            records = [r for r in store.read_records() if r.cycle_id is not None]
            if not records:
                typer.echo("No canary cycles found.", err=True)
                raise typer.Exit(code=1)
            frame = signatures_frame(records)
            cycles = list(dict.fromkeys(frame["cycle_id"]))
            chosen = cycles[:first] if first is not None else cycles[-(last or 0) :]
        elif cycle_ids:
            chosen = list(cycle_ids)
        else:
            typer.echo("Provide cycle IDs, --first N, or --last N.", err=True)
            raise typer.Exit(code=1)
        set_golden_baseline(store, chosen)
        typer.echo(f"Golden baseline frozen: {get_golden_baseline(store)}")


@app.command()
def check(
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
    inference: Annotated[
        str,
        typer.Option(
            "--inference",
            help=(
                "fixed = per-check FDR (default, the reference implementation); "
                "anytime = e-processes with a lifetime guarantee. Both run on "
                "identical logs, so results are directly comparable."
            ),
        ),
    ] = "",
) -> None:
    """Run the gated drift check for the latest cycle (dual baselines)."""
    from dedrift.check import run_check
    from dedrift.config import ProjectConfig

    store = Store(path)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    cfg = ProjectConfig.load(store.project_dir)
    mode = inference or cfg.inference
    if mode not in ("fixed", "anytime"):
        typer.echo(f"--inference must be 'fixed' or 'anytime', got {mode!r}", err=True)
        raise typer.Exit(code=1)
    if mode == "anytime":
        _check_anytime(store, cfg)
        return
    with store:
        result = run_check(store)
    typer.echo(f"Current cycle: {result.current_cycle}")
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
    typer.echo(f"Alerts: {result.n_alerts} (q={result.fdr_q}, materiality-gated)")
    for a in result.alerts()[:10]:
        typer.echo(
            f"  [{a.baseline}] {a.family}/{a.signature} {a.outcome.test}: "
            f"effect={a.outcome.effect_size:+.3f}, p_adj={a.p_adjusted:.4g}"
        )
    if result.n_alerts > 0:
        raise typer.Exit(code=2)


def _check_anytime(store: Store, cfg: ProjectConfig) -> None:
    """Render the anytime-valid check. Exit 2 on drift, matching `fixed`."""
    from dedrift.anytime import run_anytime_check

    with store:
        res = run_anytime_check(store, cfg)
    typer.echo(f"Current cycle: {res.current_cycle}   epoch fingerprint {res.fingerprint}")
    typer.echo(
        f"Anytime-valid: alpha={res.alpha} = alpha'({res.alpha_prime}) "
        f"+ gamma_total({res.gamma_total}); {res.n_processes} e-processes, "
        f"gamma per process {res.gamma_per_process:.2e}"
    )
    typer.echo(f"Verdict: {res.verdict}")
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
    typer.echo(
        "Guarantee: P(ever alerting on a stable agent) <= alpha, PER EPOCH "
        "(a suite/embedder/baseline change resets the evidence)."
    )
    if res.n_alerts:
        raise typer.Exit(code=2)


@app.command()
def report(
    output: Annotated[
        Path | None, typer.Option("--out", help="Write markdown here (default: stdout).")
    ] = None,
    path: Annotated[Path, typer.Option("--project", help="Project directory.")] = Path("."),
) -> None:
    """Run a check and render the full markdown report."""
    from dedrift.check import run_check
    from dedrift.report import render_report

    store = Store(path)
    if not store.exists():
        typer.echo("No dedrift project here. Run `dedrift init` first.", err=True)
        raise typer.Exit(code=1)
    with store:
        result = run_check(store)
        markdown = render_report(store, result)
    if output is None:
        typer.echo(markdown)
    else:
        output.write_text(markdown, encoding="utf-8")
        typer.echo(f"Report written to {output}")


if __name__ == "__main__":
    app()
