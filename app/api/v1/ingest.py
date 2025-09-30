"""Ingestion endpoint."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core.deps import get_file_store, get_ingest_queue
from app.ingest import parse_and_chunk
from app.models import IngestRequest, IngestResponse
from app.services.files import FileRecord, FileStore, IngestQueue, IngestStatus
from app.services.vectorstore import index_chunks

router = APIRouter(tags=["ingest"])


def _load_record(store: FileStore, file_id: str) -> FileRecord:
    record = store.get(file_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND")
    return record


def _process_file(record: FileRecord) -> tuple[int, Optional[str]]:
    try:
        data = record.path.read_bytes()
    except FileNotFoundError:
        return 0, "FILE_MISSING"

    chunks = parse_and_chunk(record.filename, data)
    if not chunks:
        return 0, "NO_TEXT_FOUND"

    indexed = index_chunks(chunks)
    return indexed, None


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    payload: IngestRequest,
    store: FileStore = Depends(get_file_store),
    queue: IngestQueue = Depends(get_ingest_queue),
) -> IngestResponse:
    """Trigger ingestion of the uploaded file and report its status."""

    record = _load_record(store, payload.file_id)

    if record.status == IngestStatus.PROCESSING:
        return IngestResponse(file_id=record.id, status=record.status, chunks=record.chunks, error=record.error)

    if record.status == IngestStatus.COMPLETED and not payload.force:
        return IngestResponse(file_id=record.id, status=record.status, chunks=record.chunks)

    queue.remove(record.id)
    store.update_status(record.id, status=IngestStatus.PROCESSING, error=None)

    indexed, error = _process_file(record)

    if error is not None:
        store.update_status(record.id, status=IngestStatus.FAILED, chunks=indexed, error=error)
        return IngestResponse(file_id=record.id, status=IngestStatus.FAILED, chunks=indexed, error=error)

    store.update_status(record.id, status=IngestStatus.COMPLETED, chunks=indexed, error=None)
    return IngestResponse(file_id=record.id, status=IngestStatus.COMPLETED, chunks=indexed)
