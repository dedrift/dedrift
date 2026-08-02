"""Markdown report rendering (SPEC.md §8). Deterministic given logs + config."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from jinja2 import Environment

from dedrift.attribution import Attribution, attribute
from dedrift.check import CheckResult
from dedrift.signatures import signatures_frame
from dedrift.store import Store

SPARK_CHARS = "▁▂▃▄▅▆▇█"

_TEMPLATE = """\
# dedrift report — {{ verdict }}

Generated for cycle `{{ r.current_cycle }}` (check ts {{ r.ts }}; seed {{ r.seed }};
FDR q = {{ r.fdr_q }}). Same logs + same config reproduce this report exactly.

| Comparison | Reference | Verdict |
|---|---|---|
| Sudden (rolling window) | {{ r.rolling_cycles | join(', ') or '—' }} | **{{ r.verdict_sudden }}** |
| Cumulative (golden baseline) | {{ r.golden_cycles | join(', ') or '—' }} | **{{ r.verdict_cumulative }}** |
{% if r.degraded %}
> **DEGRADED DATA:** more than {{ degraded_pct }}% of current-cycle records carry
> errors. Drift conclusions are suppressed; fix data collection first.
{% endif %}

## Alerts ({{ alerts | length }})

{% if alerts %}
Alerts require BOTH Benjamini-Hochberg significance at q = {{ r.fdr_q }} AND a
material effect (configured per channel). Effects are shown in plain units.

| Baseline | Family | Signature | Test | Effect | p (BH-adjusted) |
|---|---|---|---|---|---|
{% for a in alerts -%}
| {{ a.baseline }} | {{ a.family }} | {{ a.signature }} | {{ a.outcome.test }} | {{ a | effect }} | {{ '%.4g' | format(a.p_adjusted) }} |
{% endfor %}
{% else %}
No alerts. Statistically significant results below materiality (if any) are
listed in the appendix — deliberately not alerted (principle: alert quality).
{% endif %}

## Attribution (correlational — "consistent with", never "caused by")

{% if attributions %}
{% for at in attributions -%}
- **{{ at.family }} / {{ at.signature }}** — drift onset ≈ cycle `{{ at.onset_cycle }}`
  ({{ at.onset_ts }}).{% if at.nearest_event_ts %} Nearest config event: {{ at.nearest_event_change }},
  {{ at.nearest_event_delta_hours }} h {% if at.nearest_event_delta_hours >= 0 %}before{% else %}after{% endif %} onset.{% else %} No config events recorded.{% endif %}
  {%- if at.co_shifting > 0 %} {{ at.co_shifting }} other signature group(s) shifted in the same window.{% endif %}
{% endfor %}
{% else %}
No alerts, so no attribution performed.
{% endif %}

## Heuristic & sequential flags (not hypothesis tests)

{% if r.flags %}
| Kind | Family | Signature | Value | Label / direction | Onset estimate |
|---|---|---|---|---|---|
{% for f in r.flags -%}
| {{ f.kind }} | {{ f.family }} | {{ f.signature }} | {{ '%.4g' | format(f.value) }} | {{ f.label }} | {{ f.change_cycle_id or '—' }} |
{% endfor %}
Flags are uncalibrated diagnostics with NO multiplicity control — a stable
agent will show occasional flags, and they never alert. PSI is a heuristic
index (0.1 / 0.25 conventional thresholds), not a p-value. Page-Hinkley
alarms localize drift onsets for attribution.
{% else %}
None.
{% endif %}

## Signature trajectories (per-family means across cycles)

{% for row in sparklines -%}
- `{{ row.family }}` / {{ row.signature }}: {{ row.spark }} ({{ '%.4g' | format(row.first) }} → {{ '%.4g' | format(row.last) }})
{% endfor %}

## Config timeline

{% if config_events %}
| Timestamp | Change |
|---|---|
{% for e in config_events -%}
| {{ e[0] }} | {{ (e[1] or 'project start')[:19] }} → {{ e[2][:19] }} |
{% endfor %}
{% else %}
No config events recorded.
{% endif %}

## Appendix — all non-alerting results

Results marked significant survived FDR but failed materiality; dedrift
deliberately does not alert on them. Anderson-Darling and Welch run as
corroboration OUTSIDE the FDR pool (they test the same hypothesis as KS on
the same data); their raw p-values are shown for context and never alert.

| Baseline | Family | Signature | Test | Effect | p | Status |
|---|---|---|---|---|---|---|
{% for t in non_alerts -%}
| {{ t.baseline }} | {{ t.family }} | {{ t.signature }} | {{ t.outcome.test }} | {{ t | effect }} | {{ ('raw %.4g' | format(t.outcome.p_value)) if not t.primary and t.outcome.p_value == t.outcome.p_value else ('NA' if t.p_adjusted != t.p_adjusted else 'adj %.4g' | format(t.p_adjusted)) }} | {{ 'corroboration' if not t.primary else ('significant, below materiality' if t.significant else 'not significant') }} |
{% endfor %}
"""


def _effect_str(t: object) -> str:
    from dedrift.check import TestRecord

    assert isinstance(t, TestRecord)
    o = t.outcome
    if o.test == "two_proportion_z":
        return f"{o.effect_raw * 100:+.2f} pp"
    if o.test == "levene":
        return f"var ratio {o.effect_size:.2f}"
    if o.test == "p95_perm":
        return f"P95 {o.effect_raw:+.4g} ({o.effect_size * 100:+.1f}%)"
    if o.test == "ks":
        # D is the gated effect; the mean shift is context. Cohen's d for the
        # same comparison appears on the Welch corroboration row.
        return f"D={o.effect_size:.2f} (Δmean {o.effect_raw:+.4g})"
    if o.test == "mmd":
        return f"MMD²={o.effect_size:.4g}"
    return f"d={o.effect_size:+.2f} (raw {o.effect_raw:+.4g})"


def _sparkline(values: npt.NDArray[np.float64]) -> str:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return ""
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    chars = []
    for v in values:
        if not np.isfinite(v):
            chars.append(" ")
            continue
        idx = 0 if span == 0 else int((v - lo) / span * (len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def render_report(store: Store, result: CheckResult) -> str:
    """Render the markdown report for a check result.

    Args:
        store: The project store.
        result: The check result to render.

    Returns:
        Markdown text; deterministic given identical logs and config.
    """
    records = [r for r in store.read_records() if r.cycle_id is not None]
    frame = signatures_frame(records)
    cycles = list(dict.fromkeys(frame["cycle_id"]))

    spark_rows = []
    key_signatures = ("output_words", "latency_ms", "refusal", "format_valid")
    for family in sorted(frame["family"].unique()):
        fam = frame[frame["family"] == family]
        for sig in key_signatures:
            means = fam.groupby("cycle_id")[sig].mean().reindex(cycles).to_numpy(dtype=float)
            spark_rows.append(
                {
                    "family": family,
                    "signature": sig,
                    "spark": _sparkline(means),
                    "first": float(means[0]),
                    "last": float(means[-1]),
                }
            )

    attributions: list[Attribution] = attribute(store, result)
    env = Environment(autoescape=False)
    env.filters["effect"] = _effect_str
    template = env.from_string(_TEMPLATE)

    if result.degraded:
        verdict = "DEGRADED DATA"
    elif result.n_alerts > 0:
        verdict = "DRIFT DETECTED"
    else:
        verdict = "OK"

    non_alerts = sorted(
        (t for t in result.tests if not t.alert),
        key=lambda t: (
            not t.significant,
            t.p_adjusted if t.p_adjusted == t.p_adjusted else 2.0,
            t.baseline,
            t.family,
            t.signature,
            t.outcome.test,
        ),
    )
    return template.render(
        r=result,
        verdict=verdict,
        alerts=result.alerts(),
        attributions=attributions,
        sparklines=spark_rows,
        config_events=store.config_events(),
        non_alerts=non_alerts,
        degraded_pct=20,
    )
