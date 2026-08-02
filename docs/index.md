# dedrift {.dd-visually-hidden style="display:none"}

<div class="dd-hero" markdown>

<h1>Agents don't throw errors when they degrade.<br>They keep confidently producing <em>worse outputs</em>.</h1>

<p class="dd-sub">
dedrift catches silent behavioral drift in AI agents — with calibrated
false-alarm rates, FDR-controlled alerting, and config-change attribution.
Built by a statistician who got tired of monitoring tools whose p-values lie.
</p>

[Get started](#try-it-in-two-minutes){ .md-button .md-button--primary }
[The statistics](statistics.md){ .md-button }
[GitHub](https://github.com/dedrift/dedrift){ .md-button }

<div class="dd-pip"><span>$</span> pip install dedrift</div>

</div>

<div class="dd-section-title">Your observability catches crashes.<br>It doesn't catch character changes.</div>
<p class="dd-section-sub">
Model updates, prompt edits, tool-schema changes, RAG refreshes, provider-side
silent updates — all shift agent behavior without a single error in your logs.
</p>

<div class="grid cards" markdown>

-   :material-chart-bell-curve:{ .lg .middle } **Calibrated, not vibes-based**

    ---

    Every detector's false-alarm rate is validated by simulation tests that
    run in CI on every commit — and inside the release pipeline. A release
    cannot ship if the statistics fail their own audit.

-   :material-filter-check:{ .lg .middle } **Two gates before any alert**

    ---

    Benjamini–Hochberg FDR across every test in the check, then a
    materiality gate on effect size. Statistically significant but
    practically trivial? You don't get paged.

-   :material-bird:{ .lg .middle } **Frozen canaries, N repetitions**

    ---

    LLM outputs are random, so single-run comparisons are meaningless.
    Canaries run N times per cycle; dedrift compares distributions —
    against a rolling window *and* a frozen golden baseline.

-   :material-source-branch:{ .lg .middle } **Attribution, honestly framed**

    ---

    Every record carries a config fingerprint. When behavior shifts, the
    report correlates onset with the nearest config event — "consistent
    with the model change 5h earlier", never "caused by".

-   :material-laptop:{ .lg .middle } **Runs on a laptop**

    ---

    JSONL logs + SQLite. No servers, no docker-compose, no SaaS account.
    Core installs with zero ML dependencies; embeddings are an optional
    extra with a pinned-forever model.

-   :material-scale-balance:{ .lg .middle } **Honest about power**

    ---

    Small canary suites have limited detection power. The docs show the
    math — including which shifts you *won't* detect at default scale —
    instead of hiding it.

</div>

## Try it in two minutes

No API keys: the built-in simulator plays an agent whose model version is
swapped mid-history.

```bash
pip install dedrift
mkdir drift-demo && cd drift-demo
dedrift init
dedrift sim --cycles 8 --change-cycle 7
dedrift baseline set cycle-0000 cycle-0001 cycle-0002
dedrift check
```

<div class="dd-terminal">
<span class="p">$</span> dedrift check<br>
Current cycle: cycle-0007<br>
Sudden (vs rolling 4 cycles): <span class="a">DRIFT DETECTED</span><br>
Cumulative (vs golden 3 cycles): <span class="a">DRIFT DETECTED</span><br>
Alerts: 168 (q=0.05, materiality-gated)<br>
&nbsp;&nbsp;[golden] adversarial/refusal two_proportion_z: effect=<span class="a">+21.0 pp</span>, p_adj=0.0009<br>
&nbsp;&nbsp;[golden] adversarial/format_valid two_proportion_z: effect=<span class="a">-21.9 pp</span>, p_adj=0.0023<br>
&nbsp;&nbsp;[golden] edge_case/output_words ks: d=<span class="a">+1.89</span>, p_adj=1.4e-20<br>
<span class="c"># dedrift report --out report.md  →  attribution: nearest config event,</span><br>
<span class="c"># model fingerprint change, 0.0 h before onset — consistent with the swap.</span>
</div>

## With your own agent

One function and a YAML file of frozen canary inputs:

```python
# myagent.py
def agent_fn(input: dict) -> dict:
    response = my_agent.run(input["text"])
    return {"text": response.text, "structured": response.json}
```

```bash
dedrift canary run --suite canaries.yaml --agent myagent:agent_fn \
    --model 'anthropic/claude-sonnet-5@2026-05-01'
dedrift check && dedrift report
```

Run it on a schedule. When behavior shifts, you'll know what moved, by how
much, since when — and what changed in your stack at the same time.

## dedrift Pro

Anytime-valid sequential inference (e-processes), conditional
production-traffic drift, and importance weighting are part of a separate
commercial tier — contact [support@dedrift.ai](mailto:support@dedrift.ai).
