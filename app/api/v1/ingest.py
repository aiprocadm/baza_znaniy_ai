"""Ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.core.deps import get_ingest_service, get_ingest_session, get_tenant
from app.ingest.service import IngestService
from app.models import IngestRequest, IngestResponse
from app.models.file import FileRecord, FileStatus

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    payload: IngestRequest,
    ingest_service: IngestService = Depends(get_ingest_service),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(get_tenant),
) -> IngestResponse:
    """Ensure the requested file has been ingested and return its status."""

    try:
        record_id = int(payload.file_id)
    except (TypeError, ValueError):  # pragma: no cover - invalid identifier
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND") from None

    record = session.get(FileRecord, record_id)
    if record is None or record.tenant_id != tenant:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND")

    if record.status in {FileStatus.QUEUED, FileStatus.PROCESSING} and not payload.force:
        await ingest_service.queue.join()
        session.refresh(record)
        return IngestResponse(
            file_id=str(record.id),
            status=record.status,
            chunks=record.chunks,
            error=record.error,
        )

    if record.status == FileStatus.COMPLETED and not payload.force:
        return IngestResponse(
            file_id=str(record.id),
            status=record.status,
            chunks=record.chunks,
        )

    record.status = FileStatus.QUEUED
    record.retries = 0
    record.error = None
    record.chunks = None
    session.add(record)
    session.commit()
    session.refresh(record)

    await ingest_service.enqueue_job(record)
    await ingest_service.queue.join()
    session.refresh(record)

    return IngestResponse(
        file_id=str(record.id),
        status=record.status,
        chunks=record.chunks,
        error=record.error,
    )
