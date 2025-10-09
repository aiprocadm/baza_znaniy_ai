from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.core.auth import ensure_tenant_access, get_current_active_user
from app.core.deps import get_ingest_session
from app.models import FileInfo, FilesResponse, FileSummaryResponse
from app.models.user import UserRecord
from app.models.file import DocumentRecord, FileRecord
from app.services.file_stats import compute_file_stats

router = APIRouter(tags=["files"])


@router.get("/files", response_model=FilesResponse)
def list_files(
    _: UserRecord = Depends(get_current_active_user),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(ensure_tenant_access),
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


@router.get("/files/summary", response_model=FileSummaryResponse)
def file_summary(
    _: UserRecord = Depends(get_current_active_user),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(ensure_tenant_access),
) -> FileSummaryResponse:
    """Return aggregated statistics about files uploaded by the tenant."""

    stats = compute_file_stats(session, tenant)
    return FileSummaryResponse(
        total_files=stats.total_files,
        total_size_bytes=stats.total_size_bytes,
        total_chunks=stats.total_chunks,
        status_counts=stats.status_counts,
        oldest_upload=stats.oldest_upload,
        newest_upload=stats.newest_upload,
        average_size_bytes=stats.average_size_bytes,
    )
