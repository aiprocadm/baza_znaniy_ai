"""Regression tests for synchronous engine fallbacks in ``app.models.file``."""

from __future__ import annotations

from pathlib import Path

from app.models import file as file_module


def test_ensure_sync_engine_wraps_missing_dialect_and_dispose(tmp_path) -> None:
    """Ensure fallback attributes persist for engines lacking core APIs."""

    url = f"sqlite:///{Path(tmp_path) / 'fallback.sqlite'}"
    marker = getattr(file_module, "_FALLBACK_MARKER")

    class StubEngine:
        def __init__(self) -> None:
            self.url = url

    base_engine = StubEngine()

    wrapped = file_module._ensure_sync_engine(base_engine, url)

    assert getattr(wrapped.dialect, "name") == "sqlite"
    assert getattr(wrapped.dialect, "driver") == "sqlite"
    assert getattr(wrapped.dialect, marker, False)

    dispose = getattr(wrapped, "dispose")
    assert callable(dispose)
    assert getattr(dispose, marker, False)
    dispose()

    wrapped_again = file_module._ensure_sync_engine(wrapped, url)
    assert getattr(wrapped_again.dialect, "name") == "sqlite"
    assert getattr(wrapped_again.dialect, "driver") == "sqlite"
    assert getattr(wrapped_again.dialect, marker, False)

    dispose_again = getattr(wrapped_again, "dispose")
    assert callable(dispose_again)
    assert getattr(dispose_again, marker, False)
