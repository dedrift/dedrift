"""Headline measurements must agree across every artifact that prints them.

The project's claim is that its numbers are checkable. Those numbers live in
places edited at different times -- the MkDocs pages, the README, the marketing
site, the CHANGELOG, and (when present locally) the LaTeX paper. Re-running a
study and updating only some of them produces exactly the failure this package
exists to criticise: a published rate that no longer describes the build.

A partially-updated table is the specific thing this test catches. Add a row to
``CANONICAL`` whenever a measurement becomes load-bearing, and add its
superseded values to ``RETIRED`` so they cannot quietly return.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "launch" / "paper" / "dedrift_paper.tex"

#: quantity -> (values that must all appear, artifacts that must carry them)
CANONICAL: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "cycle-effect ladder, record-level (default) column",
        ("3.0", "36.8", "70.6", "88.3", "97.6"),
        ("docs/statistics.md", "web/index.html"),
    ),
    (
        "cycle-effect ladder, cluster-aware (auto) column",
        ("4.5", "34.4", "63.7", "80.6", "92.9"),
        ("docs/statistics.md", "web/index.html"),
    ),
    (
        "cycle-effect ladder, auto + alert_persistence=2 column",
        ("1.8", "7.6", "31.7", "47.1", "65.4"),
        ("docs/statistics.md", "web/index.html"),
    ),
    (
        "pipeline null at CI scale, default mode",
        ("16/500",),
        ("docs/statistics.md", "README.md"),
    ),
    (
        "anytime null, measured boundary at sigma=0.25, phi=0.9",
        ("36/500",),
        ("docs/anytime.md", "CHANGELOG.md"),
    ),
    (
        "anytime power, +10pp within 400 cycles",
        ("89/100",),
        ("docs/anytime.md", "README.md", "CHANGELOG.md"),
    ),
]

#: values superseded by a re-measurement, with the one file allowed to cite them
RETIRED: list[tuple[str, tuple[str, ...], str]] = [
    (
        "pre-tool_order cycle-effect ladder (battery was m~300, now m~336)",
        ("33.5%", "70.8%", "87.5%", "97.8%"),
        "docs/statistics.md",  # documents the supersession explicitly, once
    ),
]


def _normalise(text: str) -> str:
    """Strip LaTeX and HTML noise so the same number compares equal everywhere."""
    text = text.replace("\\%", "%").replace("\\,", "").replace("{,}", "").replace("$", "")
    text = text.replace("&nbsp;", " ").replace("\\mathbf{", "").replace("\\textbf{", "")
    return re.sub(r"<[^>]+>", " ", text)


def _read(rel: str) -> str | None:
    path = ROOT / rel
    return _normalise(path.read_text(encoding="utf-8")) if path.exists() else None


def _artifacts(files: tuple[str, ...]) -> list[str]:
    """Shipped artifacts plus the paper, which is gitignored and often absent."""
    return [*files, "launch/paper/dedrift_paper.tex"] if PAPER.exists() else list(files)


@pytest.mark.parametrize(
    ("quantity", "values", "files"),
    CANONICAL,
    ids=[q for q, _, _ in CANONICAL],
)
def test_quantity_is_consistent(
    quantity: str, values: tuple[str, ...], files: tuple[str, ...]
) -> None:
    """Every artifact carrying part of a measurement must carry all of it."""
    for rel in _artifacts(files):
        body = _read(rel)
        if body is None:
            continue
        missing = [v for v in values if v not in body]
        if missing and len(missing) < len(values):
            pytest.fail(
                f"{rel} carries part of '{quantity}' but is missing {missing}. "
                f"A partially-updated table means the artifact describes a build "
                f"that no longer exists; re-run the study or update every file."
            )


@pytest.mark.parametrize(
    ("label", "values", "allowed"),
    RETIRED,
    ids=[label for label, _, _ in RETIRED],
)
def test_retired_values_do_not_return(label: str, values: tuple[str, ...], allowed: str) -> None:
    """A superseded measurement may be cited as history in one place only."""
    for rel in ("docs/statistics.md", "web/index.html", "README.md", "CHANGELOG.md"):
        if rel == allowed:
            continue
        body = _read(rel)
        if body is None:
            continue
        hits = [v for v in values if v in body]
        assert len(hits) < len(values), (
            f"{rel} reprints the superseded values {hits} ({label}). "
            f"Only {allowed} may cite them, and only as explicit history."
        )
