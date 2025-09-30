"""Endpoints for working with uploaded file metadata."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.deps import get_file_store, get_tenant
from app.models import FileInfo, FilesResponse
from app.services.files import FileStore

router = APIRouter(tags=["files"])


@router.get("/files", response_model=FilesResponse)
def list_files(
    store: FileStore = Depends(get_file_store),
    tenant: str = Depends(get_tenant),
) -> FilesResponse:
    """Return metadata of files uploaded by the current tenant."""

    items = [
        FileInfo(
            id=record.id,
            filename=record.filename,
            tenant=record.tenant,
            status=record.status,
            uploaded_at=record.uploaded_at,
            size=record.size,
            chunks=record.chunks or None,
            error=record.error,
        )
        for record in store.all()
        if record.tenant == tenant
    ]
    items.sort(key=lambda entry: entry.uploaded_at, reverse=True)
    return FilesResponse(files=items)
