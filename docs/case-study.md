# Demonstration: a silent model swap, caught in one cycle

Simulations prove calibration. This page shows what a check looks like when
the ground shifts — a scripted silent model swap you can regenerate end to
end with four commands. Every number below is the output of those commands
on the current build.

## The setup

The synthetic agent (`dedrift sim`) runs a frozen suite of **30 canaries
across six behavioral families** — happy-path, edge-case, refusal-boundary,
tool-heavy JSON tasks, adversarial injections, long-context — at **7
repetitions per cycle** (210 records per cycle). Tier-2 semantic signatures
use the zero-dependency pinned hash embedder.

```bash
dedrift init
dedrift embedder pin hash
dedrift sim --cycles 8 --change-cycle 7   # scripted swap at the 8th cycle
dedrift baseline set --first 3            # golden = cycles 1–3
```

The scripted swap changes the model identifier from `simmodel@v1` to
`simmodel@v2` — **same prompt, same canaries, same tooling** — with the
classic silent-degradation profile: longer outputs, higher refusal and
format-error rates, a thicker latency tail. The only trace visible to the
system is a changed configuration fingerprint. Nobody told the detector
anything.

## The verdict

```text
$ dedrift check
Current cycle: cycle-0007
Overall: DRIFT DETECTED
Sudden (vs rolling 4 cycles): DRIFT DETECTED
Cumulative (vs golden 3 cycles): DRIFT DETECTED
Alerts: 124 (BH-adjusted equality tests q=0.05, observed-effect gated)
  [golden] adversarial/output_chars ks: effect=+0.762, p_adj=2.42e-14
  [golden] adversarial/output_words ks: effect=+0.771, p_adj=8.97e-15
  [golden] adversarial/semantic_displacement ks: effect=+0.390, p_adj=0.002136
  [golden] adversarial/tokens_out ks: effect=+0.676, p_adj=7.13e-11
  ...
```

Both baselines fired on the first post-swap check. Effects are reported on
the scales they were gated on: output-length KS statistics up to D = 0.93
(the distributions barely overlap), semantic displacement D ≈ 0.4–0.5 on
the hash embedder, and latency *dispersion* up ~2.1× on the robust
mean-absolute-deviation scale.

!!! note "How these effects are measured — and why MMD is silent here"
    The semantic centroid is fit on reference records alone, so
    displacement is an out-of-sample score, and latency dispersion is the
    robust mean-absolute-deviation ratio the test is computed on, not a
    variance ratio.

    The demo freezes a *three*-cycle golden baseline, and the MMD noise
    floor requires five reference cycles. The floor is therefore
    unavailable here, so MMD is diagnostic-only and cannot alert — that
    fail-closed behavior is deliberate: three pairwise values cannot
    calibrate a materiality floor. Use at least five known-good golden
    cycles or configure an externally justified floor.

![Per-record distributions by cycle: output tokens jump after the swap; latency medians hold while the tail thickens](assets/fig1_the_catch.png)

That right-hand panel is why dedrift tracks variance and P95 alongside
means: agents often go erratic before they go wrong on average.

## The attribution

Every alerting signature group was correlated with the recorded config
events:

> **adversarial / output_words** — drift onset ≈ cycle `cycle-0007`.
> Nearest preceding config event: model fingerprint change, **0.0 h before
> onset**. 21 other signature group(s) shifted in the same window.

Twenty-one signature groups co-shifting in one cycle, right after a
fingerprint change, is what a configuration change looks like — as opposed
to a single family decaying on its own. The report says *"consistent
with"*, never *"caused by"*; that's a design rule. Events that occur after
the estimated onset are never described as preceding it — they are labelled
weak evidence instead.

## Honest imperfections

- With eight cycles of history the Page–Hinkley change-point localizer is
  at the edge of its usable range, and a couple of signature groups
  localize the onset a cycle early; those attributions are labelled as
  weak evidence rather than preceding-onset. The estimate's measured false
  alarm behavior is documented in [Statistics](statistics.md).
- This is a scripted synthetic agent: it demonstrates detection and
  attribution mechanics, not production prevalence. Evidence on real
  production agents is what the design-partner program collects —
  [apply from the pricing section](https://dedrift.ai/#pricing) if you run
  one.

## Reproduce it

```bash
pip install dedrift
dedrift init && dedrift embedder pin hash
dedrift sim --cycles 8 --change-cycle 7 && dedrift baseline set --first 3
dedrift check
```

Runs in about a minute. The numbers above regenerate bit-for-bit (the sim
is seeded), and `dedrift report` renders the full markdown report with the
attribution table.
