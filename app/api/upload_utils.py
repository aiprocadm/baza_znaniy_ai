"""Shared helpers for working with FastAPI ``UploadFile`` objects."""

from __future__ import annotations

from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Iterable, Sequence

from fastapi import HTTPException, UploadFile, status
import app.models  # noqa: F401 - ensure package registration for test stubs
import app.core.deps  # noqa: F401 - ensure dependency module registration
import app.core.auth  # noqa: F401 - ensure auth module registration
from starlette.datastructures import MutableHeaders
from starlette.requests import Request

from app.api.status_codes import HTTP_CONTENT_TOO_LARGE
from app.api.upload_policies import ALLOWED_CONTENT_TYPES_BY_EXTENSION

try:  # pragma: no cover - optional dependency when running with minimal stubs
    from app.core.deps import UploadLimits
except ImportError:  # pragma: no cover - fallback for stubbed environments
    UploadLimits = None  # type: ignore[assignment, misc]


_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    content_type
    for values in ALLOWED_CONTENT_TYPES_BY_EXTENSION.values()
    for content_type in values
)


def _normalise_header_size(value: str | None) -> int | None:
    if not value:
        return None
    try:
        size = int(value)
    except (TypeError, ValueError):
        return None
    if size < 0:
        return None
    return size


def validate_upload_request(
    request: Request | None,
    uploads: Sequence[UploadFile],
    limits: "UploadLimits",
    *,
    allowed_content_types: Iterable[str] | None = None,
) -> None:
    """Validate declared metadata for uploaded files."""

    if UploadLimits is None:  # pragma: no cover - sanity guard for tests
        raise RuntimeError("Upload limits are unavailable in the current environment")

    header_length = _normalise_header_size(
        request.headers.get("content-length") if request is not None else None
    )
    if header_length is not None and header_length > limits.max_size:
        raise HTTPException(HTTP_CONTENT_TOO_LARGE, detail="UPLOAD_TOO_LARGE")

    allowed = frozenset(allowed_content_types or _ALLOWED_CONTENT_TYPES)

    for upload in uploads:
        filename = getattr(upload, "filename", "") or ""
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension and extension not in limits.allowed_extensions:
            continue
        raw_type = getattr(upload, "content_type", None)
        content_type = (raw_type or "").split(";", 1)[0].strip().lower()
        if not content_type or content_type not in allowed:
            raise HTTPException(
                getattr(status, "HTTP_415_UNSUPPORTED_MEDIA_TYPE", 415),
                detail="UPLOAD_INVALID_TYPE",
            )


def _ensure_mutable_content_type() -> None:
    """Install a ``content_type`` setter on FastAPI's ``UploadFile``."""

    descriptor = getattr(UploadFile, "content_type", None)
    if not isinstance(descriptor, property) or descriptor.fset is not None:
        return

    def _coerce_headers(obj: UploadFile) -> MutableHeaders:
        headers = getattr(obj, "headers", None)
        if isinstance(headers, MutableHeaders):
            return headers
        if headers is None:
            mutable = MutableHeaders()
        else:
            try:
                raw = list(getattr(headers, "raw"))  # type: ignore[attr-defined]
            except Exception:
                raw = list(getattr(headers, "items")())  # type: ignore[attr-defined]
            mutable = MutableHeaders(raw=raw)
        try:
            setattr(obj, "headers", mutable)
        except Exception:  # pragma: no cover - defensive
            try:
                object.__setattr__(obj, "headers", mutable)
            except Exception:
                pass
        return mutable

    def _set_content_type(self: UploadFile, value: str | None) -> None:
        headers = _coerce_headers(self)
        if value is None:
            try:
                headers.pop("content-type", None)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
        else:
            headers["content-type"] = value

    setattr(UploadFile, "content_type", descriptor.setter(_set_content_type))


_ensure_mutable_content_type()


def create_upload_file(
    filename: str | None,
    content: Any,
    content_type: str | None = None,
) -> UploadFile:
    """Instantiate an ``UploadFile`` from raw content or an existing stream."""

    if isinstance(content, UploadFile):
        return content

    if hasattr(content, "read"):
        file_obj = content
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    else:
        if isinstance(content, (bytes, bytearray, memoryview)):
            data = bytes(content)
        elif content is None:
            data = b""
        else:
            data = str(content).encode()
        file_obj = SpooledTemporaryFile(max_size=max(len(data), 1024), mode="w+b")
        if data:
            file_obj.write(data)
        file_obj.seek(0)

    upload = UploadFile(filename=filename, file=file_obj)
    if content_type is not None:
        try:
            upload.content_type = content_type  # type: ignore[assignment]
        except AttributeError:
            headers = getattr(upload, "headers", None)
            if headers is not None:
                try:
                    raw = list(getattr(headers, "raw"))  # type: ignore[attr-defined]
                except Exception:
                    raw = list(getattr(headers, "items")())  # type: ignore[attr-defined]
                mutable = MutableHeaders(raw=raw)
                mutable["content-type"] = content_type
                upload.headers = mutable
    return upload
