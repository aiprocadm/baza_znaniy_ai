from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.core.deps import get_ingest_session, get_tenant
from app.models import DeleteResponse
from app.models.file import FileRecord

router = APIRouter(tags=["files"])


@router.delete("/file/{file_id}", response_model=DeleteResponse)
def delete_file(
    file_id: str,
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(get_tenant),
) -> DeleteResponse:
    """Delete file metadata and remove the file from storage."""

    try:
        record_id = int(file_id)
    except (TypeError, ValueError):  # pragma: no cover - invalid identifier
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND") from None

    record = session.get(FileRecord, record_id)
    if record is None or record.tenant_id != tenant:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="FILE_NOT_FOUND")

    path = Path(record.path)
    try:
        path.unlink()
    except FileNotFoundError:  # pragma: no cover - best effort cleanup
        pass

    session.delete(record)
    session.commit()

    return DeleteResponse(ok=True, id=str(file_id))
