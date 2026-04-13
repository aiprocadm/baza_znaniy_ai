"""Helpers for tracking ingestion queue metrics in memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Dict, Sequence

from app.core.datetime_utils import utc_now
from app.services.files import FileRecord, FileStore, IngestQueue, IngestStatus


_KNOWN_STATUSES: tuple[str, ...] = (
    IngestStatus.PENDING,
    IngestStatus.PROCESSING,
    IngestStatus.COMPLETED,
    IngestStatus.FAILED,
)


@dataclass(frozen=True, slots=True)
class FailureSnapshot:
    """Minimal metadata about a failed ingestion attempt."""

    file_id: str
    filename: str
    status: str
    error: str | None
    uploaded_at: datetime


@dataclass(frozen=True, slots=True)
class IngestQueueMetrics:
    """Aggregated metrics about the in-memory ingest queue."""

    total_files: int
    queue_depth: int
    status_counts: Dict[str, int]
    oldest_pending_age_seconds: float | None
    average_pending_age_seconds: float | None
    recent_failures: tuple[FailureSnapshot, ...]
    last_activity_at: datetime | None


def _normalise_datetime(value: datetime | None) -> datetime | None:
    """Ensure *value* is timezone-aware in UTC for downstream consumers."""

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iter_records(store: FileStore, tenant: str | None) -> Sequence[FileRecord]:
    """Return file records scoped to *tenant* when provided."""

    records = list(store.all())
    if tenant is None:
        return records
    return [record for record in records if record.tenant == tenant]


def _compute_queue_depth(queue: IngestQueue, store: FileStore, tenant: str | None) -> int:
    """Return queue depth scoped to *tenant* when requested."""

    if tenant is None:
        return len(queue)
    depth = 0
    for file_id in queue.items():
        record = store.get(file_id)
        if record is not None and record.tenant == tenant:
            depth += 1
    return depth


def compute_ingest_queue_metrics(
    store: FileStore,
    queue: IngestQueue,
    *,
    tenant: str | None = None,
    now: datetime | None = None,
    failure_limit: int = 5,
) -> IngestQueueMetrics:
    """Return aggregated ingest queue metrics for *tenant*.

    The function inspects the shared :class:`~app.services.files.FileStore` and
    :class:`~app.services.files.IngestQueue` instances and derives metrics used by
    the Operations Console without mutating any state.
    """

    scoped_records = _iter_records(store, tenant)
    status_counts: Dict[str, int] = {status: 0 for status in _KNOWN_STATUSES}
    last_activity: datetime | None = None

    for record in scoped_records:
        status = record.status or "unknown"
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1
        uploaded_at = _normalise_datetime(record.uploaded_at)
        if uploaded_at is not None and (
            last_activity is None or uploaded_at > last_activity
        ):
            last_activity = uploaded_at

    now_dt = _normalise_datetime(now) or utc_now()
    pending_ages: list[float] = []
    for record in scoped_records:
        if record.status != IngestStatus.PENDING:
            continue
        uploaded_at = _normalise_datetime(record.uploaded_at)
        if uploaded_at is None:
            continue
        age_seconds = (now_dt - uploaded_at).total_seconds()
        if age_seconds < 0:
            age_seconds = 0.0
        pending_ages.append(age_seconds)

    oldest_pending = max(pending_ages) if pending_ages else None
    average_pending = mean(pending_ages) if pending_ages else None

    failed_records: list[FailureSnapshot] = []
    for record in scoped_records:
        if record.status != IngestStatus.FAILED:
            continue
        uploaded_at = _normalise_datetime(record.uploaded_at)
        if uploaded_at is None:
            continue
        failed_records.append(
            FailureSnapshot(
                file_id=record.id,
                filename=record.filename,
                status=record.status,
                error=record.error,
                uploaded_at=uploaded_at,
            )
        )

    failed_records.sort(key=lambda item: item.uploaded_at, reverse=True)
    if failure_limit < 0:
        limited_failures = failed_records
    elif failure_limit == 0:
        limited_failures = []
    else:
        limited_failures = failed_records[:failure_limit]
    recent_failures = tuple(limited_failures)

    return IngestQueueMetrics(
        total_files=len(scoped_records),
        queue_depth=_compute_queue_depth(queue, store, tenant),
        status_counts=status_counts,
        oldest_pending_age_seconds=oldest_pending,
        average_pending_age_seconds=average_pending,
        recent_failures=recent_failures,
        last_activity_at=last_activity,
    )


__all__ = [
    "FailureSnapshot",
    "IngestQueueMetrics",
    "compute_ingest_queue_metrics",
]
