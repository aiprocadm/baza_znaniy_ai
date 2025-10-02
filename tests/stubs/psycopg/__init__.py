"""Lightweight psycopg stub used by the test suite."""

from __future__ import annotations

from typing import Any, Optional

from . import conninfo, rows, sql


class Connection:  # pragma: no cover - placeholder for type compatibility
    """Placeholder connection type for type hints."""

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Stub Connection cannot be used directly")

    def commit(self) -> None:
        raise NotImplementedError("Stub Connection cannot be used directly")

    def rollback(self) -> None:
        raise NotImplementedError("Stub Connection cannot be used directly")

    def close(self) -> None:
        raise NotImplementedError("Stub Connection cannot be used directly")


def connect(dsn: str, row_factory: Optional[Any] = None, **_: Any) -> Connection:
    """Return a stub connection.

    Tests monkeypatch this function to provide fake connections. The default
    implementation fails fast to avoid accidental use outside the test suite.
    """

    raise RuntimeError("psycopg stub connect() should be patched in tests")


__all__ = ["connect", "Connection", "sql", "rows", "conninfo"]
