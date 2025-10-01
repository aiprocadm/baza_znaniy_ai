"""Upload endpoints."""

from __future__ import annotations

import secrets
import mimetypes
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.auth import ensure_tenant_access, get_current_active_user
from app.core.deps import (
    UploadLimits,
    get_data_dir,
    get_ingest_service,
    get_upload_limits,
)
from app.models.user import UserRecord
from app.models import UploadResponse
from app.ingest.service import IngestService
from app.api.upload_utils import create_upload_file

router = APIRouter(tags=["upload"])


def _normalise_extension(filename: str) -> str:
    name = (filename or "").strip()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


async def _read_file(upload: UploadFile, limits: UploadLimits) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="UPLOAD_EMPTY")
    if len(data) > limits.max_size:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="UPLOAD_TOO_LARGE")
    return data


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    *,
    file: Optional[List[UploadFile]] = File(None, alias="file"),
    files: Optional[List[UploadFile]] = File(None, alias="files"),
    limits: UploadLimits = Depends(get_upload_limits),
    data_dir: Path = Depends(get_data_dir),
    _: UserRecord = Depends(get_current_active_user),
    tenant: str = Depends(ensure_tenant_access),
    ingest_service: IngestService = Depends(get_ingest_service),
) -> UploadResponse:
    """Store an uploaded file on disk and register it for ingestion."""

    uploads = []
    if file:
        uploads.extend(file)
    if files:
        uploads.extend(files)

    if not uploads:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="UPLOAD_EMPTY")

    def _coerce(item: object) -> UploadFile:
        if isinstance(item, UploadFile):
            return item
        if isinstance(item, dict):  # pragma: no cover - compatibility for test stubs
            filename = item.get("filename")
            content = item.get("content", b"")
            content_type = item.get("content_type")
            return create_upload_file(filename, content, content_type)
        if isinstance(item, (list, tuple)):
            filename = item[0] if item else "uploaded"
            content = item[1] if len(item) > 1 else b""
            content_type = item[2] if len(item) > 2 else None
            return create_upload_file(filename, content, content_type)
        if isinstance(item, str):
            return create_upload_file(item, b"")
        return create_upload_file("uploaded", b"")

    coerced = [_coerce(item) for item in uploads]
    upload = next(
        (
            item
            for item in coerced
            if _normalise_extension((item.filename or "")) in limits.allowed_extensions
        ),
        coerced[0],
    )
    extension = _normalise_extension(upload.filename or "")
    if extension not in limits.allowed_extensions:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="UPLOAD_INVALID_EXT")

    payload = await _read_file(upload, limits)

    tenant_dir = data_dir / tenant
    tenant_dir.mkdir(parents=True, exist_ok=True)

    file_id = secrets.token_hex(16)
    target = tenant_dir / f"{file_id}.{extension}"
    target.write_bytes(payload)

    mime_type = getattr(upload, "content_type", None)
    if not mime_type:
        guessed, _ = mimetypes.guess_type(upload.filename or "")
        mime_type = guessed or "application/octet-stream"

    record, queued = await ingest_service.register_file(
        tenant,
        str(target),
        filename=upload.filename or target.name,
        size=len(payload),
        mime_type=mime_type,
    )

    if not queued and record.path != str(target):
        try:
            target.unlink()
        except FileNotFoundError:  # pragma: no cover - defensive cleanup
            pass

    file_identifier = str(record.id or "")
    if not file_identifier:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="UPLOAD_FAILED")

    return UploadResponse(
        file_id=file_identifier,
        filename=record.filename,
        tenant=tenant,
        status=record.status,
        queued=queued,
    )
