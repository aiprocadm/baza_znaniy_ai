"""Upload endpoints."""

from __future__ import annotations

        codex/update-upload-file-handling-and-tests
import io

        main
import mimetypes
import secrets
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

    coerced = [_coerce_upload_argument(item) for item in uploads]
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

    def _as_bytes(value: object) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):  # pragma: no cover - defensive branch
            return value.tobytes()
        if value is None:
            return b""
        return str(value).encode()

    def _spooled_file(data: object) -> SpooledTemporaryFile:
        stream = SpooledTemporaryFile(max_size=max(1, limits.max_size), mode="w+b")
        payload = _as_bytes(data)
        if payload:
            stream.write(payload)
        stream.seek(0)
        return stream


    def _coerce(item: object) -> UploadFile:
        if isinstance(item, UploadFile):
            return item

        filename: Optional[str] = None
        content: object = b""
        content_type: Optional[str] = None

        if isinstance(item, dict):  # pragma: no cover - compatibility for legacy clients
            filename = item.get("filename")
            content_type = item.get("content_type")
            content = item.get("file")
            if content is None:
                content = item.get("content", b"")
        elif isinstance(item, (list, tuple)):
            filename = str(item[0]) if item else "uploaded"
            content = item[1] if len(item) > 1 else b""
            third = item[2] if len(item) > 2 else None
            content_type = third if isinstance(third, str) else None
        elif isinstance(item, str):
            filename = item
            content = b""
        else:
            filename = "uploaded"
            content = b""


        return create_upload_file(filename, content, content_type)

        return UploadFile(filename=filename, file=file_obj, content_type=content_type)


    coerced = [_coerce(item) for item in uploads]
        main

    selected_extension = ""
    selected_filename: Optional[str] = None
    selected_content_type: Optional[str] = None
    payload: bytes

    try:
        upload = next(
            (
                item
                for item in coerced
                if _normalise_extension((item.filename or "")) in limits.allowed_extensions
            ),
            coerced[0],
        )

        selected_filename = upload.filename
        selected_content_type = getattr(upload, "content_type", None)
        selected_extension = _normalise_extension(selected_filename or "")
        if selected_extension not in limits.allowed_extensions:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="UPLOAD_INVALID_EXT")

        payload = await _read_file(upload, limits)
    finally:
        for item in coerced:
            close = getattr(item, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if result is not None and hasattr(result, "__await__"):
                    await result  # type: ignore[func-returns-value]
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    tenant_dir = data_dir / tenant
    tenant_dir.mkdir(parents=True, exist_ok=True)

    file_id = secrets.token_hex(16)
    target = tenant_dir / f"{file_id}.{selected_extension}"
    target.write_bytes(payload)

    mime_type = selected_content_type
    if not mime_type:
        guessed, _ = mimetypes.guess_type(selected_filename or "")
        mime_type = guessed or "application/octet-stream"

    record, queued = await ingest_service.register_file(
        tenant,
        str(target),
        filename=selected_filename or target.name,
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


def _coerce_upload_argument(item: object) -> UploadFile:
    if isinstance(item, UploadFile):
        return item
    if isinstance(item, dict):  # pragma: no cover - compatibility for test stubs
        filename = item.get("filename")
        content = item.get("content", b"")
        return UploadFile(filename=filename, file=io.BytesIO(_ensure_bytes(content)))
    if isinstance(item, (list, tuple)):
        filename = item[0] if item else "uploaded"
        content = item[1] if len(item) > 1 else b""
        return UploadFile(filename=filename, file=io.BytesIO(_ensure_bytes(content)))
    if isinstance(item, str):
        return UploadFile(filename=item, file=io.BytesIO())
    return UploadFile(filename="uploaded", file=io.BytesIO())


def _ensure_bytes(payload: object) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode()
    if payload is None:
        return b""
    read = getattr(payload, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode()
        try:
            return bytes(data)
        except Exception:
            return b""
    try:
        return bytes(payload)
    except Exception:
        return b""
