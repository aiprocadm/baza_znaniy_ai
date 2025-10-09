"""Ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.core.datetime_utils import utc_now
from app.core.auth import ensure_tenant_access, get_current_active_user
from app.core.deps import (
    get_file_store,
    get_ingest_queue,
    get_ingest_service,
    get_ingest_session,
)
from app.models.user import UserRecord
from app.ingest.service import IngestQueueFullError, IngestService
from app.models import (
    IngestFailureInfo,
    IngestQueueMetricsResponse,
    IngestRequest,
    IngestResponse,
)
from app.models.file import DocumentRecord, DocumentStatus, FileRecord, FileStatus
from app.services.files import FileStore, IngestQueue
from app.services.ingest_monitoring import compute_ingest_queue_metrics

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    payload: IngestRequest,
    _: UserRecord = Depends(get_current_active_user),
    ingest_service: IngestService = Depends(get_ingest_service),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(ensure_tenant_access),
) -> IngestResponse:
    """Ensure the requested file has been ingested and return its status."""

    try:
        record_id = int(payload.file_id)
    except (TypeError, ValueError):  # pragma: no cover - invalid identifier
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND") from None

    record = session.get(FileRecord, record_id)
    if record is None or record.tenant_id != tenant:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND")

    if record.status == FileStatus.COMPLETED and not payload.force:
        return IngestResponse(
            file_id=str(record.id),
            status=record.status,
            chunks=record.chunks,
        )

    should_reset = payload.force or record.status not in {
        FileStatus.QUEUED,
        FileStatus.PROCESSING,
    }

    if should_reset:
        record.status = FileStatus.QUEUED
        record.retries = 0
        record.error = None
        record.chunks = None
        session.add(record)
        document = (
            session.get(DocumentRecord, record.document_id)
            if record.document_id is not None
            else None
        )
        if document:
            document.status = DocumentStatus.QUEUED
            document.error = None
            document.chunks = None
            document.updated_at = utc_now()
            session.add(document)
        session.commit()

    try:
        await ingest_service.enqueue_job(record)
    except IngestQueueFullError as exc:
        status_code = getattr(status, "HTTP_429_TOO_MANY_REQUESTS", 429)
        raise HTTPException(status_code, detail=str(exc)) from exc
    session.refresh(record)

    return IngestResponse(
        file_id=str(record.id),
        status=record.status,
        chunks=record.chunks,
        error=record.error,
    )


@router.get("/ingest/metrics", response_model=IngestQueueMetricsResponse)
def ingest_metrics(
    _: UserRecord = Depends(get_current_active_user),
    tenant: str = Depends(ensure_tenant_access),
    file_store: FileStore = Depends(get_file_store),
    ingest_queue: IngestQueue = Depends(get_ingest_queue),
) -> IngestQueueMetricsResponse:
    """Return real-time ingest metrics for the active tenant."""

    metrics = compute_ingest_queue_metrics(
        file_store,
        ingest_queue,
        tenant=tenant,
    )

    return IngestQueueMetricsResponse(
        total_files=metrics.total_files,
        queue_depth=metrics.queue_depth,
        status_counts=metrics.status_counts,
        oldest_pending_age_seconds=metrics.oldest_pending_age_seconds,
        average_pending_age_seconds=metrics.average_pending_age_seconds,
        recent_failures=[
            IngestFailureInfo(
                file_id=item.file_id,
                filename=item.filename,
                status=item.status,
                error=item.error,
                uploaded_at=item.uploaded_at,
            )
            for item in metrics.recent_failures
        ],
        last_activity_at=metrics.last_activity_at,
    )
