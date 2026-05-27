"""Utilities for working with timezone-aware datetimes."""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["utc_now", "utc_now_naive"]


def utc_now() -> datetime:
    """Return the current UTC time as an aware ``datetime`` instance."""

    return datetime.now(timezone.utc)


def utc_now_naive() -> datetime:
    """Return the current UTC time stripped of ``tzinfo`` (naive).

    Bridge for callers that still write into SQLModel/SQLAlchemy columns
    declared as ``datetime`` without ``timezone=True`` (notably
    :class:`app.models.audit.AuditLog.timestamp`). The behaviour exactly
    matches the deprecated ``datetime.utcnow()`` so call sites can swap
    without semantic shift.

    TODO: once those columns are migrated to ``DateTime(timezone=True)``
    via Alembic, the call sites should switch to :func:`utc_now` and
    this helper can be deleted.
    """

    return datetime.now(timezone.utc).replace(tzinfo=None)
