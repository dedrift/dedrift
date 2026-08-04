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
{% if r.composition_issues %}
> **COMPOSITION MISMATCH — {{ r.composition_issues | length }} family comparison(s) suppressed.**
> The windows below do not contain the same canaries at uniform repetition
> counts, so a two-sample test there would measure missing data, not drift.
> Fix the collection gap (or re-baseline after a suite change) and re-run.
>
{% for c in r.composition_issues -%}
> - `[{{ c.baseline }}] {{ c.family }}`: {{ c.detail }}
{% endfor %}
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


_ANYTIME_TEMPLATE = """\
# dedrift report (anytime-valid) — {{ r.verdict }}

Cycle `{{ r.current_cycle }}`, epoch fingerprint `{{ r.fingerprint }}`
(check ts {{ r.ts }}).

## The guarantee, stated exactly

Over an **unbounded** monitoring horizon, the probability of ever raising a
false alert on a stable agent is at most **α = {{ r.alpha }}** — *per epoch*.

| Component | Value | What it pays for |
|---|---|---|
| α (lifetime, battery-wide) | {{ r.alpha }} | the whole claim |
| α′ (e-BH level) | {{ '%.4g' | format(r.alpha_prime) }} | multiplicity across {{ r.n_processes }} e-processes |
| γ total | {{ r.gamma_total }} | nuisance-parameter coverage |
| γ per process | {{ '%.2e' | format(r.gamma_per_process) }} | γ total ÷ {{ r.n_processes }} |

{% if r.pool_declared_now %}
The pool of {{ r.n_processes }} e-processes was **declared at this check** —
this is the epoch's first. Membership is now frozen for the epoch, because
the pool size sets the coverage budget, which sets the nuisance interval,
which is part of the bet: for a frozen baseline the guarantee needs that
interval to be one fixed event rather than a sequence of them. A signature
with no reference data right now therefore waits for the next epoch to join.
{% else %}
Pool membership was frozen when this epoch began; only combinations with
reference data at that moment are in it.
{% endif %}

γ is divided by the pool size because e-BH requires every input to be a
valid e-value: a process whose coverage interval misses the truth is not
one, so coverage failures union-bound across the battery. Using the total
per process would state {{ r.alpha }} while delivering
{{ '%.2f' | format(r.alpha_prime + r.n_processes * r.gamma_total) }}.

**"Per epoch" is not a caveat to skim.** An epoch ends when the canary
suite, embedder, golden baseline or extractor changes — which changes the
hypothesis, so evidence gathered before it is not evidence about the null
being tested now. A guarantee spanning a hypothesis change would be
meaningless rather than stronger.
{% if r.resets %}
## ⚠ Epoch resets at this check ({{ r.resets | length }})

Wealth was returned to zero for these processes. Their guarantee restarts
from this cycle.
{% for note in r.resets %}
- {{ note }}
{% endfor %}
{% endif %}
{% if r.degraded %}
> **DEGRADED DATA:** too many current-cycle records carry errors. Alerts are
> suppressed; e-values were still accumulated because a suppressed cycle
> contributes `E_t = 1` exactly, which preserves the supermartingale.
{% endif %}

## Alerts ({{ alerts | length }})

{% if alerts %}
Rejected by e-BH at q = {{ '%.4g' | format(r.alpha_prime) }} over the running e-processes.

| Process | log-wealth | evidence 1/α reached | onset ≈ cycle | crossed at |
|---|---|---|---|---|
{% for p in alerts -%}
| {{ p.label }} | {{ '%.2f' | format(p.log_wealth) }} | {{ '%.3g' | format(p.evalue_capped) }} | {{ p.rise_cycle or '—' }} | {{ p.crossed_at or '—' }} |
{% endfor %}
{% else %}
No alerts. Wealth has not accumulated past the e-BH threshold on any
process.
{% endif %}

## All e-processes

Wealth is the accumulated evidence *since the epoch began*. Negative means
the bets have lost — evidence **for** stability, which the fixed-sample path
cannot express. "Bets" counts cycles where a bet was admissible; a
suppressed or degenerate cycle contributes exactly 1 (log 0) and is not a
missing update.

| Process | log-wealth | epoch | cycles | bets | onset | alert |
|---|---|---|---|---|---|---|
{% for p in processes -%}
| {{ p.label }} | {{ '%.2f' | format(p.log_wealth) }} | {{ p.epoch }} | {{ p.cycles }} | {{ p.bets_placed }} | {{ p.rise_cycle or '—' }} | {{ 'YES' if p.rejected else '' }} |
{% endfor %}

## Multiplicity spent, and on what

{{ idle }} of {{ r.n_processes }} processes in this epoch's pool have not
placed a bet yet. Combinations with no reference data at epoch start are
excluded from the pool entirely — they cannot produce evidence, so charging
the battery's coverage budget for them would be power given away. What
remains here are processes that are admissible but have had no usable
current-cycle data yet; they contribute ``E_t = 1`` exactly, which preserves
the supermartingale rather than skipping an update.

## What is proven, and what is measured

- **Proven, per process:** Ville's inequality bounds the probability of ever
  crossing at α′, for any stopping rule.
- **Proven, per check:** e-BH controls FDR under arbitrary dependence among
  the e-values.
- **Measured, not proven, over the trajectory:** applying e-BH at every
  cycle to running e-processes is anytime-valid under a causal condition
  (no unobserved confounding from the past) that our dependent streams
  plausibly but not provably satisfy. The realised rate is measured by
  simulation instead; see the statistics documentation.
- **Honest about power:** the nuisance worst-casing makes each bet
  conservative, so small shifts accumulate slowly or not at all. Detection
  delay by effect size is published rather than implied.
"""


def render_anytime_report(result: object) -> str:
    """Render the markdown report for an anytime-valid check.

    Deliberately a separate template rather than a branch inside the
    fixed-sample one: the two paths report different objects (wealth
    trajectories and epochs versus p-values and effect sizes), and merging
    them would blur exactly the distinction an operator needs to keep
    straight.

    Args:
        result: An :class:`dedrift.anytime.AnytimeCheckResult`.

    Returns:
        Markdown text; deterministic given identical state.
    """
    from dedrift.anytime import AnytimeCheckResult

    assert isinstance(result, AnytimeCheckResult)
    env = Environment(autoescape=False)
    template = env.from_string(_ANYTIME_TEMPLATE)
    return template.render(
        r=result,
        alerts=result.alerts(),
        processes=sorted(result.processes, key=lambda p: -p.log_wealth),
        idle=sum(1 for p in result.processes if p.bets_placed == 0),
    )


def _effect_str(t: object) -> str:
    from dedrift.check import TestRecord

    assert isinstance(t, TestRecord)
    o = t.outcome
    if o.test == "two_proportion_z":
        return f"{o.effect_raw * 100:+.2f} pp"
    if o.test == "levene":
        # NOT the variance ratio: Brown-Forsythe is computed on absolute
        # deviations from the median, and the effect is reported and gated on
        # that same robust scale (see detectors/scalar.py). Printing "var
        # ratio" here mislabelled the number by a squaring, which is exactly
        # the confusion the robust gate exists to prevent.
        return f"dispersion ratio (MAD) {o.effect_size:.2f}"
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
