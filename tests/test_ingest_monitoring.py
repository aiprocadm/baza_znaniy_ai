from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from app.core.datetime_utils import utc_now
from app.services.files import FileRecord, FileStore, IngestQueue, IngestStatus
from app.services.ingest_monitoring import compute_ingest_queue_metrics


def _make_record(
    *,
    record_id: str,
    filename: str,
    tenant: str,
    path: Path,
    uploaded_at,
    status: str,
    size: int = 0,
    error: str | None = None,
) -> FileRecord:
    record = FileRecord(
        id=record_id,
        filename=filename,
        tenant=tenant,
        path=path,
        size=size,
        uploaded_at=uploaded_at,
        status=status,
    )
    if error is not None:
        record.error = error
    return record


def test_compute_ingest_queue_metrics_filters_by_tenant(tmp_path: Path) -> None:
    store = FileStore()
    queue = IngestQueue()

    payload_path = tmp_path / "payload.txt"
    payload_path.write_text("payload")

    base_time = utc_now() - timedelta(minutes=30)
    tenant = "tenant-a"

    pending = _make_record(
        record_id="pending-1",
        filename="pending.txt",
        tenant=tenant,
        path=payload_path,
        uploaded_at=base_time,
        status=IngestStatus.PENDING,
        size=payload_path.stat().st_size,
    )
    processing = _make_record(
        record_id="processing-1",
        filename="processing.txt",
        tenant=tenant,
        path=payload_path,
        uploaded_at=base_time + timedelta(minutes=5),
        status=IngestStatus.PROCESSING,
    )
    failed = _make_record(
        record_id="failed-1",
        filename="failed.txt",
        tenant=tenant,
        path=payload_path,
        uploaded_at=base_time + timedelta(minutes=20),
        status=IngestStatus.FAILED,
        error="boom",
    )
    other_tenant = _make_record(
        record_id="other-1",
        filename="other.txt",
        tenant="tenant-b",
        path=payload_path,
        uploaded_at=base_time + timedelta(minutes=1),
        status=IngestStatus.PENDING,
    )

    for record in (pending, processing, failed, other_tenant):
        store.add(record)

    queue.enqueue(pending.id)
    queue.enqueue(failed.id)
    queue.enqueue(other_tenant.id)

    snapshot_time = base_time + timedelta(minutes=45)
    metrics = compute_ingest_queue_metrics(
        store,
        queue,
        tenant=tenant,
        now=snapshot_time,
    )

    assert metrics.total_files == 3
    assert metrics.queue_depth == 2
    assert metrics.status_counts[IngestStatus.PENDING] == 1
    assert metrics.status_counts[IngestStatus.PROCESSING] == 1
    assert metrics.status_counts[IngestStatus.FAILED] == 1
    assert metrics.status_counts[IngestStatus.COMPLETED] == 0
    assert metrics.oldest_pending_age_seconds == pytest.approx(
        (snapshot_time - base_time).total_seconds()
    )
    assert metrics.average_pending_age_seconds == pytest.approx(
        (snapshot_time - base_time).total_seconds()
    )
    assert len(metrics.recent_failures) == 1
    assert metrics.recent_failures[0].file_id == failed.id
    assert metrics.recent_failures[0].error == "boom"
    assert metrics.recent_failures[0].uploaded_at.tzinfo is not None
    assert metrics.last_activity_at == metrics.recent_failures[0].uploaded_at


def test_compute_ingest_queue_metrics_handles_empty_state() -> None:
    store = FileStore()
    queue = IngestQueue()

    metrics = compute_ingest_queue_metrics(store, queue)

    assert metrics.total_files == 0
    assert metrics.queue_depth == 0
    assert metrics.status_counts[IngestStatus.PENDING] == 0
    assert metrics.status_counts[IngestStatus.PROCESSING] == 0
    assert metrics.status_counts[IngestStatus.COMPLETED] == 0
    assert metrics.status_counts[IngestStatus.FAILED] == 0
    assert metrics.oldest_pending_age_seconds is None
    assert metrics.average_pending_age_seconds is None
    assert metrics.recent_failures == ()
    assert metrics.last_activity_at is None
