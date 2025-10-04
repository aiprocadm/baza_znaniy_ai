"""Pytest configuration for ensuring local stub packages are discoverable."""

from __future__ import annotations

import sys
from pathlib import Path

# The lightweight dependency stubs now live under ``tests/stubs``.  Prepending
# the directory to ``sys.path`` keeps the test suite hermetic while letting the
# real application import genuine third-party packages when they are installed.
STUBS_PATH = Path(__file__).resolve().parent / "stubs"

# Ensure heavy optional dependencies are imported before the stub directory takes
# precedence on ``sys.path``.  When the real packages are available they remain
# cached in ``sys.modules`` and continue to be used by the application code.
for module_name in (
    "pydantic",
    "sqlmodel",
    "fastapi",
    "fastapi.testclient",

    "psycopg",
    "psycopg.conninfo",

    "numpy",
    "openpyxl",
    "pptx",

    "sqlalchemy",

):
    try:  # pragma: no cover - exercised during integration tests
        __import__(module_name)
    except Exception:  # pragma: no cover - fallback to stubs when missing
        pass

if STUBS_PATH.exists():
    sys.path.insert(0, str(STUBS_PATH))
