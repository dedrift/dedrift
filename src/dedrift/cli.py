"""dedrift command-line interface.

Phases 0-1 expose ``init``, ``log``, ``sim``, ``canary run``, and
``signatures``. Later phases add ``check`` and ``report`` (see ROADMAP.md).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated

import typer

from dedrift.schema import AgentConfig, InteractionRecord
from dedrift.sim import SimAgent, SimConfig, drifted_profile
from dedrift.store import Store

app = typer.Typer(
    name="dedrift",
    help="Statistically rigorous behavioral drift detection for AI agents.",
    no_args_is_help=True,
)
canary_app = typer.Typer(help="Canary suite operations.", no_args_is_help=True)
app.add_typer(canary_app, name="canary")


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


if __name__ == "__main__":
    app()
