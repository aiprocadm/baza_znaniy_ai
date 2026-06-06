"""Pytest configuration for ensuring local stub packages are discoverable."""

from __future__ import annotations

import os
import stat
import sys
import warnings
from pathlib import Path

import pytest

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

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type swigvarlink has no __module__ attribute",
)

# --- Committed-fixture write guard -----------------------------------------
# These files are *inputs* the suite must treat as read-only. Their writers all
# default to these very paths via relative defaults
# (``scripts/build_curated_golden.py`` → ``data/eval/golden_curated.*``;
# ``scripts/download_model.py`` → ``models/model_manifest.json``;
# ``scripts/eval_rag.py`` ``run --golden`` default), so a test that drives one
# of them without redirecting to ``tmp_path`` would silently regenerate a
# tracked fixture and dirty the working tree — invisible until ``git status``.
# The ``st``/``dim:1024`` signature and the normalised qwen manifest entry seen
# after some local runs are the fingerprints of those writers.
REPO_ROOT = Path(__file__).resolve().parents[1]
_PROTECTED_FIXTURES = (
    REPO_ROOT / "data" / "eval" / "golden_curated.jsonl",
    REPO_ROOT / "data" / "eval" / "golden_curated.sig.json",
    REPO_ROOT / "models" / "model_manifest.json",
)


def _set_writable(path: Path, writable: bool) -> None:
    """Toggle the read-only bit (the only ``chmod`` semantics Windows honours)."""
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IWRITE if writable else mode & ~stat.S_IWRITE)


@pytest.fixture(scope="session", autouse=True)
def _protect_committed_fixtures():
    """Fail loudly (and restore) if any test mutates a committed data fixture.

    Default mode snapshots the fixtures at session start; at session end, any
    that changed are rewritten with their original bytes — so ``git status``
    stays clean even when a writer leaks through — and the session fails so the
    regression is visible rather than silent.

    Set ``KB_TEST_LOCK_FIXTURES=1`` to instead mark the files read-only for the
    duration of the run: the offending test then fails in place with a
    ``PermissionError`` traceback that names it (the snapshot/restore path
    cannot, since teardown runs after every test). The lock self-heals a prior
    aborted run by clearing the read-only bit before snapshotting.
    """
    present = [p for p in _PROTECTED_FIXTURES if p.exists()]
    for p in present:  # self-heal: a killed lock-mode run may have left these read-only
        _set_writable(p, True)
    snapshots = {p: p.read_bytes() for p in present}

    lock = os.environ.get("KB_TEST_LOCK_FIXTURES", "").strip().lower() in {"1", "true", "yes", "on"}
    if lock:
        for p in present:
            _set_writable(p, False)

    yield

    if lock:
        for p in present:
            _set_writable(p, True)

    mutated: list[Path] = []
    for path, original in snapshots.items():
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            current = None
        if current != original:
            mutated.append(path)
            path.write_bytes(original)  # restore so the working tree stays clean

    if mutated:
        names = ", ".join(str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in mutated)
        pytest.fail(
            "Committed data fixtures were modified during the test run and have "
            f"been restored to keep `git status` clean: {names}. A test wrote to "
            "a tracked fixture instead of tmp_path. Re-run with "
            "KB_TEST_LOCK_FIXTURES=1 to get a PermissionError traceback that "
            "names the offending test.",
            pytrace=False,
        )
