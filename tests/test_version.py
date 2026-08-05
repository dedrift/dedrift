"""Package version has one source of truth."""

from __future__ import annotations

from dedrift import __version__


def test_runtime_version() -> None:
    assert __version__ == "0.3.1"
