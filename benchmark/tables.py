"""Render benchmark results into the web page and the paper table.

Reads ``benchmark/results/*.json`` and rewrites:

* ``web/benchmark/index.html`` — the results tables (both scales), between
  ``BENCHMARK:TABLE:BEGIN/END`` (suite) and ``BENCHMARK:TABLE2:BEGIN/END``
  (small) markers;
* ``launch/paper/benchmark_table.tex`` — the booktabs tabular the paper's
  validity-scale section inputs;
* ``launch/paper/benchmark_macros.tex`` — the prose macros the same section
  uses for inline numbers;

with the two LaTeX fragments copied to ``launch/paper/arxiv/`` for the
submission build. Everything published flows from the results JSONs, so
``make benchmark`` refreshes every number in place. Hand-edits between the
markers are lost on the next run — that is the point.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
WEB_PAGE = REPO / "web" / "benchmark" / "index.html"
PAPER_DIR = REPO / "launch" / "paper"
ARXIV_DIR = PAPER_DIR / "arxiv"

MARKERS = {
    "suite": ("<!-- BENCHMARK:TABLE:BEGIN -->", "<!-- BENCHMARK:TABLE:END -->"),
    "small": ("<!-- BENCHMARK:TABLE2:BEGIN -->", "<!-- BENCHMARK:TABLE2:END -->"),
}

SCALE_TITLES = {
    "suite": "18 canaries × 7 repetitions (documented default suite)",
    "small": "12 canaries × 5 repetitions (dedrift CI gate scale)",
}


def _load() -> dict[tuple[str, str], dict[str, Any]]:
    """Load all four results documents."""
    docs = {}
    for leg in ("percheck", "dedrift"):
        for scale in ("suite", "small"):
            path = RESULTS / f"{leg}_{scale}.json"
            if path.exists():
                docs[(leg, scale)] = json.loads(path.read_text())
    return docs


def _f(row: dict[str, Any]) -> str:
    """Format a rate_row as ``k/n = P% [lo, hi]``."""
    return (
        f"{row['k']:,}/{row['n']:,} = {100 * row['rate']:.1f}% "
        f"[{100 * row['wilson_low']:.1f}, {100 * row['wilson_high']:.1f}]"
    ).replace(",", "{,}")


def _html(row: dict[str, Any]) -> str:
    """HTML version of ``_f``."""
    return _f(row).replace("{,}", ",")


def _rows(scale: str, docs: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, str]]:
    """Assemble the shared row model for one scale (HTML and LaTeX both)."""
    pc = docs[("percheck", scale)]["methods"]
    dd = docs[("dedrift", scale)]["methods"]
    return [
        {"group": "General-purpose configurations"},
        {
            "name": "PSI, folk threshold 0.1",
            "config": "10 reference-quantile bins; flag at PSI ≥ 0.1",
            "nominal": "none stated",
            "percheck": pc["psi_folk"]["runs_any_moderate"],
            "history": None,
        },
        {
            "name": "PSI, folk threshold 0.25",
            "config": "same metric; flag at PSI ≥ 0.25 (“major”)",
            "nominal": "none stated",
            "percheck": pc["psi_folk"]["runs_any_major"],
            "history": None,
        },
        {
            "name": "Evidently — pooled table, ≥ 1 drifted column",
            "config": "DataDriftPreset 0.7.21, all defaults",
            "nominal": "none stated (per-column tests at 0.05)",
            "percheck": pc["evidently_pooled"]["runs_any_drifted_column"],
            "history": None,
        },
        {
            "name": "Evidently — pooled table, dataset verdict",
            "config": "defaults; drifted-column share ≥ 0.5",
            "nominal": "none stated",
            "percheck": pc["evidently_pooled"]["runs_dataset_drift"],
            "history": None,
        },
        # The per-family arm is deliberately NOT tabled. Running one report
        # per family and taking any-of is a user's choice, not a documented
        # default, so its rate is partly our own multiplicity rather than the
        # tool's calibration. It stays in the results JSON; it does not belong
        # in a table that reads as "what the defaults do".
        {
            "name": "Naive two-sample KS battery",
            "config": "α = 0.05 per (family × signature), no control",
            "nominal": "5% per test only",
            "percheck": pc["naive_ks"]["runs_any_rejection"],
            "history": None,
        },
        {"group": "dedrift, measured under the same protocol"},
        {
            "name": "dedrift PSI + validity guard",
            "config": "identical metric; refused where E[PSI] > 0.05",
            "nominal": "none stated (diagnostic)",
            "percheck": pc["psi_guarded"]["runs_any_flag"],
            "history": None,
        },
        {
            "name": "dedrift fixed-sample path (alerts)",
            "config": "BH-FDR q = 0.05 + materiality, dual baselines",
            "nominal": "FDR q = 0.05 per check",
            "percheck": dd["dedrift_fixed_percheck"]["runs_any_alert"],
            "history": dd["dedrift_fixed_cumulative"]["runs_ever_alerted_50_cycles"],
        },
        {
            "name": "dedrift flag channel (Page–Hinkley / guarded PSI)",
            "config": "diagnostic flags, deliberately uncalibrated",
            "nominal": "none stated — published limitation",
            "percheck": dd["dedrift_fixed_percheck"]["runs_any_flag"],
            "history": None,
        },
        {
            "name": "dedrift anytime-valid path",
            "config": "e-processes + e-BH, golden baseline, twosample rates",
            "nominal": "≤ 0.05 lifetime per epoch",
            # No per-check cell: this path makes no per-check claim, and its
            # per-fold rate has a 25,000 denominator against every other row's
            # 500. Printing it in the same column would buy a tighter interval
            # from a different estimand.
            "percheck": None,
            "history": dd["dedrift_anytime"]["runs_ever_alerted_50_cycles"],
        },
    ]


def render_html(docs: dict[tuple[str, str], dict[str, Any]], scale: str) -> str:
    """Render one scale's results table for the web page."""
    rows = _rows(scale, docs)
    parts = [
        '<div class="tbl-wrap"><table>',
        "<thead><tr><th>Configuration</th><th>Default setting used</th>"
        "<th>Nominal / stated</th><th class='num'>≥ 1 false alarm, per check</th>"
        "<th class='num'>≥ 1 false alarm, 50-cycle history</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        if "group" in row:
            parts.append(f"<tr class='grp'><td colspan='5'>{row['group']}</td></tr>")
            continue
        percheck = _html(row["percheck"]) if row["percheck"] else "—"
        history = _html(row["history"]) if row["history"] else "—"
        parts.append(
            f"<tr><td><b>{row['name']}</b></td><td>{row['config']}</td>"
            f"<td>{row['nominal']}</td><td class='num'>{percheck}</td>"
            f"<td class='num'>{history}</td></tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append(
        "<p class='tbl-note'>k/n = runs with at least one false alarm over 500 seeded stable "
        "histories; brackets are Wilson 95% intervals. “—”: the configuration produces no such "
        "estimand. The dedrift flag channel and the PSI rows are diagnostics, not alerts; they "
        "are shown because hiding them would break the symmetry this table exists for. The "
        "anytime-valid path has no per-check cell: it makes no per-check claim, and its per-fold "
        "rate is over 25,000 folds against every other row's 500 runs, so printing it in that "
        "column would compare different estimands. Per-run "
        "and per-comparison raw rows: "
        "<a href='https://github.com/dedrift/dedrift/tree/main/benchmark/results'>"
        "benchmark/results/</a>.</p>"
    )
    return "\n".join(parts)


def _tex_escape(text: str) -> str:
    """Escape the handful of LaTeX specials appearing in row labels."""
    return (
        text.replace("%", "\\%")
        .replace("≥", "$\\ge$")
        .replace("≤", "$\\le$")
        .replace(">", "$>$")
        .replace("<", "$<$")
        .replace("×", "$\\times$")
        .replace("α", "$\\alpha$")
        .replace("“", "``")
        .replace("”", "''")
        .replace("—", "---")
        .replace("–", "--")
        .replace("dedrift", "\\dedrift{}")
    )


def _tex_cell(row: dict[str, Any]) -> str:
    """One-line LaTeX cell: ``k/n``, rate, Wilson interval.

    Single line keeps the column boundaries clean -- stacked cells make
    adjacent rows run together and are hard to read across. Rates below 0.1%
    get three decimals so a rate of 0.008% is not rounded into
    indistinguishability from zero.
    """
    nd = 3 if row["rate"] < 0.001 else 1
    rate = f"{100 * row['rate']:.{nd}f}"
    lo = f"{100 * row['wilson_low']:.{nd}f}"
    hi = f"{100 * row['wilson_high']:.{nd}f}"
    k, n = f"{row['k']:,}".replace(",", "{,}"), f"{row['n']:,}".replace(",", "{,}")
    return f"{k}/{n}\\, ${rate}\\%$ \\footnotesize$[{lo}, {hi}]$"


def render_tex(docs: dict[tuple[str, str], dict[str, Any]], scale: str) -> str:
    """Render one scale's booktabs tabular for the paper.

    Four columns, not five: the per-configuration settings that used to sit
    in their own narrow column now travel in the caption, which stops the
    label columns from wrapping into each other.
    """
    lines = [
        "% GENERATED by benchmark/tables.py from benchmark/results/*.json.",
        "% Do not hand-edit; `make benchmark` rewrites this file.",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}p{4.4cm}p{2.5cm}l l@{}}",
        "\\toprule",
        "Configuration & Nominal / stated & $\\ge 1$ false alarm & $\\ge 1$ over 50 cycles \\\\",
        "\\midrule",
    ]
    for row in _rows(scale, docs):
        if "group" in row:
            if lines[-1].endswith("\\\\") and "midrule" not in lines[-1]:
                lines.append("\\midrule")
            lines.append("\\multicolumn{4}{@{}l}{\\emph{" + _tex_escape(row["group"]) + "}} \\\\")
            lines.append("\\addlinespace[2pt]")
            continue
        percheck = _tex_cell(row["percheck"]) if row["percheck"] else "---"
        history = _tex_cell(row["history"]) if row["history"] else "---"
        lines.append(
            f"{_tex_escape(row['name'])} & {_tex_escape(row['nominal'])} & "
            f"{percheck} & {history} \\\\"
        )
        lines.append("\\addlinespace[2pt]")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _macro(name: str, body: str) -> str:
    r"""One ``\newcommand`` line."""
    return "\\newcommand{\\" + name + "}{" + body + "}"


def render_macros(docs: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Render the prose-number macros used by the validity-scale section."""
    out = [
        "% GENERATED by benchmark/tables.py from benchmark/results/*.json.",
        "% Do not hand-edit; `make benchmark` rewrites this file.",
    ]
    for scale, tag in (("suite", "Suite"), ("small", "Small")):
        pc = docs[("percheck", scale)]["methods"]
        dd = docs[("dedrift", scale)]["methods"]

        def kn(row: dict[str, Any]) -> str:
            return f"${row['k']}/{row['n']}$ (${100 * row['rate']:.1f}\\%$)".replace(",", "{,}")

        out.append(_macro(f"benchPsiAny{tag}", kn(pc["psi_folk"]["runs_any_moderate"])))
        out.append(_macro(f"benchPsiGuard{tag}", kn(pc["psi_guarded"]["runs_any_flag"])))
        out.append(_macro(f"benchKsAny{tag}", kn(pc["naive_ks"]["runs_any_rejection"])))
        out.append(
            _macro(f"benchKsPerTest{tag}", f"${100 * pc['naive_ks']['rejections']['rate']:.1f}\\%$")
        )
        out.append(
            _macro(f"benchKsPerTest{tag}Bare", f"{pc['naive_ks']['rejections']['rate']:.4f}")
        )
        per_test = pc["naive_ks"]["rejections"]["rate"]
        k_med = pc["naive_ks"]["tests_per_run_median"]
        out.append(
            _macro(f"benchKsReconciled{tag}", f"${100 * (1 - (1 - per_test) ** k_med):.1f}\\%$")
        )
        out.append(
            _macro(f"benchEvPoolAny{tag}", kn(pc["evidently_pooled"]["runs_any_drifted_column"]))
        )
        out.append(
            _macro(f"benchEvPoolData{tag}", kn(pc["evidently_pooled"]["runs_dataset_drift"]))
        )
        out.append(
            _macro(f"benchDdPerCheck{tag}", kn(dd["dedrift_fixed_percheck"]["runs_any_alert"]))
        )
        out.append(_macro(f"benchDdFlag{tag}", kn(dd["dedrift_fixed_percheck"]["runs_any_flag"])))
        out.append(
            _macro(
                f"benchDdCum{tag}",
                kn(dd["dedrift_fixed_cumulative"]["runs_ever_alerted_50_cycles"]),
            )
        )
        out.append(
            _macro(f"benchAtEver{tag}", kn(dd["dedrift_anytime"]["runs_ever_alerted_50_cycles"]))
        )
    # The chi-square mechanism sentence needs the three worst columns'
    # measured rates at suite scale.
    chi = [
        (col, cell["rate"])
        for col, cell in docs[("percheck", "suite")]["methods"]["evidently_pooled"][
            "per_column"
        ].items()
        if any("chi" in m.lower() for m in cell["methods"])
    ]
    chi.sort(key=lambda kv: -kv[1])
    body = ", ".join(f"${100 * rate:.1f}\\%$" for _, rate in chi[:3])
    out.append(_macro("benchEvChiSuite", body))
    # The same sentence contrasts those with the columns Evidently routes to
    # a p-value test. Deriving the ceiling keeps the contrast honest if a
    # re-run moves it; hand-typing a range is how a claim goes stale.
    pval_max = max(
        (
            cell["rate"]
            for cell in docs[("percheck", "suite")]["methods"]["evidently_pooled"][
                "per_column"
            ].values()
            if not any("chi" in m.lower() for m in cell["methods"])
        ),
        default=0.0,
    )
    out.append(_macro("benchEvPvalMaxSuite", f"${100 * pval_max:.1f}\\%$"))
    out.append("")
    return "\n".join(out)


def render_column_html(docs: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Render the per-column Evidently mechanism table for the web page."""
    suite = docs[("percheck", "suite")]["methods"]["evidently_pooled"]["per_column"]
    small = docs[("percheck", "small")]["methods"]["evidently_pooled"]["per_column"]
    parts = [
        '<div class="tbl-wrap"><table>',
        "<thead><tr><th>Signature column</th><th>Evidently auto-selected test</th>"
        "<th class='num'>Null flag rate, suite scale</th>"
        "<th class='num'>Null flag rate, small scale</th></tr></thead>",
        "<tbody>",
    ]
    for col in suite:
        method = suite[col]["methods"][0] if suite[col]["methods"] else "?"
        cls = "bad" if "chi" in method.lower() else "y"
        parts.append(
            f"<tr><td><code>{col}</code></td><td>{method}</td>"
            f"<td class='num'><span class='{cls}'>{100 * suite[col]['rate']:.1f}%</span></td>"
            f"<td class='num'><span class='{cls}'>{100 * small[col]['rate']:.1f}%</span></td></tr>"
        )
    parts.append("</tbody></table></div>")
    parts.append(
        "<p class='tbl-note'>Per-column false-flag rate over 500 stable runs, pooled granularity, "
        "Evidently DataDriftPreset 0.7.21 defaults. The three chi-square-routed columns are the "
        "low-cardinality integer channels; had_error and retries are structurally all-zero on the "
        "null, so their 0% is degenerate, not calibrated.</p>"
    )
    return "\n".join(parts)


def sync(path: Path, begin: str, end: str, fragment: str) -> None:
    """Replace the marked region of ``path`` with ``fragment``."""
    text = path.read_text()
    pre, marker, rest = text.partition(begin)
    if not marker:
        raise ValueError(f"begin marker not found in {path}")
    _, marker2, post = rest.partition(end)
    if not marker2:
        raise ValueError(f"end marker not found in {path}")
    path.write_text(pre + begin + "\n" + fragment + "\n" + end + post)


def main() -> None:
    """Regenerate every published fragment from the results JSONs."""
    docs = _load()
    missing = [
        (leg, scale)
        for leg in ("percheck", "dedrift")
        for scale in ("suite", "small")
        if (leg, scale) not in docs
    ]
    if missing:
        raise SystemExit(f"missing results for: {missing}; run python -m benchmark.run first")
    for scale, (begin, end) in MARKERS.items():
        sync(WEB_PAGE, begin, end, render_html(docs, scale))
    sync(
        WEB_PAGE,
        "<!-- BENCHMARK:COLUMN:BEGIN -->",
        "<!-- BENCHMARK:COLUMN:END -->",
        render_column_html(docs),
    )
    (PAPER_DIR / "benchmark_table.tex").write_text(render_tex(docs, "suite"))
    (PAPER_DIR / "benchmark_table_small.tex").write_text(render_tex(docs, "small"))
    (PAPER_DIR / "benchmark_macros.tex").write_text(render_macros(docs))
    if ARXIV_DIR.exists():
        for name in ("benchmark_table.tex", "benchmark_table_small.tex", "benchmark_macros.tex"):
            shutil.copy2(PAPER_DIR / name, ARXIV_DIR / name)
    print("tables regenerated")


if __name__ == "__main__":
    main()
