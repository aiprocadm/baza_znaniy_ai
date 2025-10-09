"""Aggregations and summary helpers for tenant file statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict

from sqlalchemy import func, select
from sqlmodel import Session

from app.models.file import FileRecord, FileStatus


_KNOWN_FILE_STATUSES: tuple[str, ...] = (
    FileStatus.QUEUED,
    FileStatus.PROCESSING,
    FileStatus.COMPLETED,
    FileStatus.FAILED,
)


@dataclass(frozen=True, slots=True)
class FileStats:
    """Aggregated statistics about files for a single tenant."""

    total_files: int
    total_size_bytes: int
    total_chunks: int
    status_counts: Dict[str, int]
    oldest_upload: datetime | None
    newest_upload: datetime | None
    average_size_bytes: float | None


def _normalise_timestamp(value: datetime | None) -> datetime | None:
    """Return a timezone-aware UTC datetime for comparisons."""

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def compute_file_stats(session: Session, tenant_id: str) -> FileStats:
    """Return aggregated file statistics for the given tenant."""

    status_counts: Dict[str, int] = {status: 0 for status in _KNOWN_FILE_STATUSES}
    total_files = 0
    total_size_bytes = 0
    total_chunks = 0

    aggregation_stmt = (
        select(
            FileRecord.status,
            func.count(FileRecord.id),
            func.coalesce(func.sum(FileRecord.size), 0),
            func.coalesce(func.sum(FileRecord.chunks), 0),
        )
        .where(FileRecord.tenant_id == tenant_id)
        .group_by(FileRecord.status)
    )

    for status, count, size_sum, chunk_sum in session.exec(aggregation_stmt):
        normalized_status = str(status or "unknown").strip() or "unknown"
        if normalized_status not in status_counts:
            status_counts[normalized_status] = 0
        count_int = int(count or 0)
        size_int = int(size_sum or 0)
        chunks_int = int(chunk_sum or 0)

        status_counts[normalized_status] += count_int
        total_files += count_int
        total_size_bytes += size_int
        total_chunks += chunks_int

    timestamp_stmt = (
        select(
            func.min(FileRecord.created_at),
            func.max(FileRecord.created_at),
        )
        .where(FileRecord.tenant_id == tenant_id)
    )
    timestamp_row = session.exec(timestamp_stmt).first()
    if timestamp_row:
        oldest_raw, newest_raw = timestamp_row
    else:
        oldest_raw = None
        newest_raw = None

    average_size = None
    if total_files > 0 and total_size_bytes >= 0:
        average_size = float(total_size_bytes) / float(total_files)

    return FileStats(
        total_files=total_files,
        total_size_bytes=total_size_bytes,
        total_chunks=total_chunks,
        status_counts=status_counts,
        oldest_upload=_normalise_timestamp(oldest_raw),
        newest_upload=_normalise_timestamp(newest_raw),
        average_size_bytes=average_size,
    )


__all__ = ["FileStats", "compute_file_stats"]
