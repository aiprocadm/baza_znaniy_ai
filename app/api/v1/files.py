from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.core.auth import ensure_tenant_access, get_current_active_user
from app.core.deps import get_ingest_session
from app.models.user import UserRecord
from app.models import FileInfo, FilesResponse
from app.models.file import FileRecord

router = APIRouter(tags=["files"])


@router.get("/files", response_model=FilesResponse)
def list_files(
    _: UserRecord = Depends(get_current_active_user),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(ensure_tenant_access),
) -> FilesResponse:
    """Return metadata of files uploaded by the current tenant."""

    statement = (
        select(FileRecord)
        .where(FileRecord.tenant_id == tenant)
        .order_by(FileRecord.created_at.desc())
    )
    records = session.exec(statement).all()
    items = [
        FileInfo(
            id=str(record.id),
            filename=record.filename,
            tenant=record.tenant_id,
            status=record.status,
            uploaded_at=record.created_at,
            size=record.size,
            chunks=record.chunks,
            error=record.error,
        )
        for record in records
    ]
    return FilesResponse(files=items)
