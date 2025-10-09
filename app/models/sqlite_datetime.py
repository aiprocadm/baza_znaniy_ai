"""SQLite datetime adapters compatible with Python 3.12+."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

import sqlite3

__all__ = ["register_sqlite_datetime_support"]


def _normalise(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _adapt_datetime(value: datetime) -> str:
    return _normalise(value).isoformat(timespec="microseconds")


_REGISTERED: Final[dict[str, bool]] = {"value": False}


def register_sqlite_datetime_support() -> None:
    """Install adapters ensuring SQLite handles timezone-aware datetimes."""

    if _REGISTERED["value"]:
        return

    sqlite3.register_adapter(datetime, _adapt_datetime)

    _REGISTERED["value"] = True
