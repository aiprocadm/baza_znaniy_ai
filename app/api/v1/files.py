from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.core.deps import get_ingest_session, get_tenant
from app.models import FileInfo, FilesResponse
from app.models.file import DocumentRecord, FileRecord

router = APIRouter(tags=["files"])


@router.get("/files", response_model=FilesResponse)
def list_files(
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(get_tenant),
) -> FilesResponse:
    """Return metadata of files uploaded by the current tenant."""

    statement = (
        select(FileRecord, DocumentRecord)
        .where(FileRecord.tenant_id == tenant)
        .join(DocumentRecord, FileRecord.document_id == DocumentRecord.id, isouter=True)
        .order_by(FileRecord.created_at.desc())
    )
    records = session.exec(statement).all()
    items = [
        FileInfo(
            id=str(file_obj.id),
            filename=file_obj.filename,
            tenant=file_obj.tenant_id,
            status=file_obj.status,
            uploaded_at=file_obj.created_at,
            size=file_obj.size,
            chunks=file_obj.chunks,
            error=file_obj.error,
            document_id=(str(file_obj.document_id) if file_obj.document_id else None),
            document_status=document.status if document is not None else None,
            mime_type=document.mime_type if document is not None else None,
        )
        for file_obj, document in records
    ]
    return FilesResponse(files=items)
