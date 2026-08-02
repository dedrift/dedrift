"""dedrift command-line interface.

Phase 0 exposes ``init``, ``log``, and ``sim``. Later phases add
``canary run``, ``check``, and ``report`` (see ROADMAP.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dedrift.schema import InteractionRecord
from dedrift.sim import SimAgent, SimConfig, drifted_profile
from dedrift.store import Store

app = typer.Typer(
    name="dedrift",
    help="Statistically rigorous behavioral drift detection for AI agents.",
    no_args_is_help=True,
)


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


if __name__ == "__main__":
    app()
