"""Utilities for working with timezone-aware datetimes."""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["utc_now"]


def utc_now() -> datetime:
    """Return the current UTC time as an aware ``datetime`` instance."""

    return datetime.now(timezone.utc)
