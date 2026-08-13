"""A placeholder must never reach a live block on the published site.

The landing page carries a paper citation band that is authored before the
arXiv identifier exists, so it ships commented out with ``PENDING`` where the
identifier goes. The failure mode this guards is mechanical and easy: someone
uncomments the band to make the paper visible and forgets to substitute the
identifier, and the site advertises ``arXiv:PENDING`` to every reader.

Commented-out placeholders are fine -- that is the staging state. A placeholder
outside an HTML comment is not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

#: markers that are acceptable only inside an HTML comment
PLACEHOLDERS = ("PENDING", "TKTK", "XXXX.XXXXX", "LOREM")


def _strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _pages() -> list[Path]:
    return sorted(WEB.rglob("*.html"))


def test_pages_exist() -> None:
    """Guard the guard: a silent glob failure would make every check vacuous."""
    assert _pages(), f"no HTML found under {WEB}; this test would pass vacuously"


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_live_placeholder(page: Path) -> None:
    """No placeholder marker outside an HTML comment on any published page."""
    live = _strip_html_comments(page.read_text(encoding="utf-8"))
    found = [marker for marker in PLACEHOLDERS if marker in live]
    assert not found, (
        f"{page.relative_to(ROOT)} has {found} outside an HTML comment. "
        f"A staged block was activated without substituting the real value; "
        f"fill it in or re-comment the block."
    )
