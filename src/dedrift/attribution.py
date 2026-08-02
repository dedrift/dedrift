"""Config-change attribution (SPEC.md §7).

Attribution is correlational, never causal. The report says "consistent
with", never "caused by". For each alert we estimate the drift onset (the
Page-Hinkley change-point when one exists for that family/signature, else
the current cycle) and rank config events by temporal proximity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from dedrift.check import CheckResult
from dedrift.schema import InteractionRecord
from dedrift.store import Store


@dataclass(frozen=True)
class Attribution:
    """Attribution of one alert group to nearby config events.

    Attributes:
        family: Canary family.
        signature: Signature name.
        onset_cycle: Estimated onset cycle ID.
        onset_ts: Timestamp of the onset cycle's first record (ISO).
        nearest_event_ts: Timestamp of the nearest config event (ISO), if any.
        nearest_event_delta_hours: Signed hours from event to onset
            (positive = event preceded onset).
        nearest_event_change: Human-readable fingerprint change.
        co_shifting: Number of other (family, signature) alert groups whose
            onset falls in the same cycle.
    """

    family: str
    signature: str
    onset_cycle: str
    onset_ts: str
    nearest_event_ts: str | None
    nearest_event_delta_hours: float | None
    nearest_event_change: str | None
    co_shifting: int


def _cycle_start_times(records: list[InteractionRecord]) -> dict[str, datetime]:
    starts: dict[str, datetime] = {}
    for r in records:
        if r.cycle_id is None:
            continue
        if r.cycle_id not in starts or r.ts < starts[r.cycle_id]:
            starts[r.cycle_id] = r.ts
    return starts


def attribute(store: Store, result: CheckResult) -> list[Attribution]:
    """Correlate each alert group with the nearest config event.

    Args:
        store: The project store (provides records and config events).
        result: The check result whose alerts need attribution.

    Returns:
        One attribution per alerting (family, signature) group,
        deterministically ordered.
    """
    alerts = result.alerts()
    if not alerts:
        return []
    records = [r for r in store.read_records() if r.cycle_id is not None]
    starts = _cycle_start_times(records)
    events = store.config_events()  # (ts, old_fp, new_fp), first event is project start
    real_events = [e for e in events if e[1] is not None]

    # Page-Hinkley onset estimates per (family, signature).
    ph_onsets: dict[tuple[str, str], str] = {}
    for flag in result.flags:
        if flag.kind == "page_hinkley" and flag.change_cycle_id is not None:
            ph_onsets[(flag.family, flag.signature)] = flag.change_cycle_id

    groups = sorted({(t.family, t.signature) for t in alerts})
    onset_by_group = {g: ph_onsets.get(g, result.current_cycle) for g in groups}
    onset_counts = pd.Series(list(onset_by_group.values())).value_counts()

    out: list[Attribution] = []
    for family, signature in groups:
        onset_cycle = onset_by_group[(family, signature)]
        onset_ts = starts.get(onset_cycle)
        if onset_ts is None:
            continue
        nearest_ts: str | None = None
        nearest_delta: float | None = None
        nearest_change: str | None = None
        for ts_str, old_fp, new_fp in real_events:
            event_ts = datetime.fromisoformat(ts_str)
            delta_hours = (onset_ts - event_ts).total_seconds() / 3600
            if nearest_delta is None or abs(delta_hours) < abs(nearest_delta):
                nearest_delta = delta_hours
                nearest_ts = ts_str
                nearest_change = f"{(old_fp or 'none')[:19]} -> {new_fp[:19]}"
        out.append(
            Attribution(
                family=family,
                signature=signature,
                onset_cycle=onset_cycle,
                onset_ts=onset_ts.isoformat(),
                nearest_event_ts=nearest_ts,
                nearest_event_delta_hours=(
                    round(nearest_delta, 2) if nearest_delta is not None else None
                ),
                nearest_event_change=nearest_change,
                co_shifting=int(onset_counts.get(onset_cycle, 1)) - 1,
            )
        )
    return out
