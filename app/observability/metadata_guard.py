"""Runtime guard that tracks the integrity of ``SQLModel.metadata``."""

from __future__ import annotations

import logging
from typing import Protocol

from sqlmodel import SQLModel

from app.observability.metrics import (
    record_sqlmodel_metadata_alert,
    record_sqlmodel_metadata_state,
)

logger = logging.getLogger(__name__)


class _SupportsAddJob(Protocol):
    """Typed protocol for the subset of APScheduler we depend on."""

    def add_job(self, func, trigger, *, seconds: float, id: str, replace_existing: bool):
        ...


_DEFAULT_INTERVAL_SECONDS = 15.0
_GUARD_JOB_ID = "sqlmodel-metadata-guard"


def check_sqlmodel_metadata(*, origin: str = "metadata_guard") -> bool:
    """Validate the runtime ``SQLModel.metadata`` instance and emit alerts."""

    metadata = getattr(SQLModel, "metadata", None)
    healthy, reason = record_sqlmodel_metadata_state(metadata, origin=origin)
    if healthy:
        return True

    record_sqlmodel_metadata_alert(origin=origin, reason=reason)
    logger.warning(
        "SQLModel metadata integrity check failed", extra={"reason": reason}
    )
    return False


def schedule_sqlmodel_metadata_guard(
    scheduler: _SupportsAddJob | None,
    *,
    interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
) -> None:
    """Register the metadata guard as a periodic background task."""

    if scheduler is None:
        return

    try:
        seconds = max(float(interval_seconds), 1.0)
    except Exception:
        seconds = _DEFAULT_INTERVAL_SECONDS

    try:
        scheduler.add_job(
            check_sqlmodel_metadata,
            "interval",
            seconds=seconds,
            id=_GUARD_JOB_ID,
            replace_existing=True,
        )
    except Exception:  # pragma: no cover - scheduler configuration is best-effort
        logger.debug("Failed to schedule SQLModel metadata guard", exc_info=True)


__all__ = [
    "check_sqlmodel_metadata",
    "schedule_sqlmodel_metadata_guard",
]
