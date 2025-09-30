"""Endpoint for deleting uploaded files."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_file_store, get_ingest_queue, get_tenant
from app.models import DeleteResponse
from app.services.files import FileStore, IngestQueue

router = APIRouter(tags=["files"])


@router.delete("/file/{file_id}", response_model=DeleteResponse)
def delete_file(
    file_id: str,
    store: FileStore = Depends(get_file_store),
    queue: IngestQueue = Depends(get_ingest_queue),
    tenant: str = Depends(get_tenant),
) -> DeleteResponse:
    """Delete file metadata and remove the file from storage."""

    record = store.get(file_id)
    if record is None or record.tenant != tenant:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND")

    queue.remove(file_id)

    try:
        record.path.unlink(missing_ok=True)
    except TypeError:  # pragma: no cover - Python <3.8 compatibility
        try:
            record.path.unlink()
        except FileNotFoundError:
            pass

    store.remove(file_id)
    return DeleteResponse(ok=True, id=file_id)
