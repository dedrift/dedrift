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
from dedrift.schema import InteractionRecord, Source
from dedrift.store import Store


@dataclass(frozen=True)
class Attribution:
    """Attribution of one alert group to nearby config events.

    Attributes:
        family: Canary family.
        signature: Signature name.
        onset_cycle: Estimated onset cycle ID.
        onset_ts: Timestamp of the onset cycle's first record (ISO).
        nearest_event_ts: Timestamp of the nominated config event (ISO), if any.
        nearest_event_delta_hours: Signed hours from event to onset
            (positive = event preceded onset; negative only when
            ``event_relation`` is ``"precedes_detection"``).
        nearest_event_change: Human-readable fingerprint change.
        event_relation: How the nominated event relates to the onset
            estimate: ``"precedes_onset"`` (at or before the estimated
            onset — the only temporally consistent reading),
            ``"precedes_detection"`` (no event precedes the noisy onset
            estimate; this is the latest event before the check ran —
            weak evidence, labelled as such), or ``None`` when no event
            exists at or before the check (a silent drift).
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
    event_relation: str | None
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
    if result.snapshot_record_ids:
        snapshot = store.read_records_by_ids(result.snapshot_record_ids)
    else:
        logged = store.read_records()
        current_positions = [
            index
            for index, record in enumerate(logged)
            if record.source == Source.CANARY and record.cycle_id == result.current_cycle
        ]
        if not current_positions:
            return []
        snapshot = logged[: current_positions[-1] + 1]
    records = [
        record
        for record in snapshot
        if record.source == Source.CANARY and record.cycle_id is not None
    ]
    starts = _cycle_start_times(records)
    events = store.config_events({record.id for record in snapshot})
    real_events = [e for e in events if e[1] is not None]

    # Page-Hinkley onset estimates per (family, signature).
    ph_onsets: dict[tuple[str, str], str] = {}
    for flag in result.flags:
        if flag.kind == "page_hinkley" and flag.change_cycle_id is not None:
            ph_onsets[(flag.family, flag.signature)] = flag.change_cycle_id

    groups = sorted({(t.family, t.signature) for t in alerts})
    onset_by_group = {g: ph_onsets.get(g, result.current_cycle) for g in groups}
    onset_counts = pd.Series(list(onset_by_group.values())).value_counts()

    # The fallback window runs to the last record the check saw: detection
    # happened at check time, so any event before that predates detection.
    check_time = max((r.ts for r in records), default=None)
    out: list[Attribution] = []
    for family, signature in groups:
        onset_cycle = onset_by_group[(family, signature)]
        onset_ts = starts.get(onset_cycle)
        if onset_ts is None:
            continue
        nearest_ts: str | None = None
        nearest_delta: float | None = None
        nearest_change: str | None = None
        relation: str | None = None
        for ts_str, old_fp, new_fp in real_events:
            event_ts = datetime.fromisoformat(ts_str)
            delta_hours = (onset_ts - event_ts).total_seconds() / 3600
            # Only events at or before the estimated onset are candidates:
            # a drift cannot be "consistent with" a change that had not
            # happened yet. (An earlier version ranked by absolute time and
            # could nominate a post-onset event as "before onset".)
            if delta_hours < 0:
                continue
            if nearest_delta is None or delta_hours < nearest_delta:
                nearest_delta = delta_hours
                nearest_ts = ts_str
                nearest_change = f"{(old_fp or 'none')[:19]} -> {new_fp[:19]}"
        if nearest_ts is not None:
            relation = "precedes_onset"
        elif check_time is not None:
            # No event precedes the estimated onset. Onset estimates come
            # from Page-Hinkley and are noisy (measured null alarm rates are
            # documented), so an event after the estimate but before this
            # check is still worth listing -- labelled as weak evidence,
            # never described as preceding the onset.
            fallback: tuple[float, str, str] | None = None
            for ts_str, old_fp, new_fp in real_events:
                event_ts = datetime.fromisoformat(ts_str)
                if event_ts > check_time:
                    continue
                gap = (check_time - event_ts).total_seconds() / 3600
                if fallback is None or gap < fallback[0]:
                    fallback = (gap, ts_str, f"{(old_fp or 'none')[:19]} -> {new_fp[:19]}")
            if fallback is not None:
                _, nearest_ts, nearest_change = fallback
                nearest_delta = round(
                    (onset_ts - datetime.fromisoformat(nearest_ts)).total_seconds() / 3600, 2
                )
                relation = "precedes_detection"
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
                event_relation=relation,
                co_shifting=int(onset_counts.get(onset_cycle, 1)) - 1,
            )
        )
    return out
