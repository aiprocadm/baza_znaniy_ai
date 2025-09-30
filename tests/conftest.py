"""Pytest configuration for ensuring local stub packages are discoverable."""

from __future__ import annotations

import sys
from pathlib import Path

# The lightweight dependency stubs now live under ``tests/stubs``.  Prepending
# the directory to ``sys.path`` keeps the test suite hermetic while letting the
# real application import genuine third-party packages when they are installed.
STUBS_PATH = Path(__file__).resolve().parent / "stubs"
if STUBS_PATH.exists():
    sys.path.insert(0, str(STUBS_PATH))
