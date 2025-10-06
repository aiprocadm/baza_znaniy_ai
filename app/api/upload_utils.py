"""Shared helpers for working with FastAPI ``UploadFile`` objects."""

from __future__ import annotations

from tempfile import SpooledTemporaryFile
from typing import Any

from fastapi import UploadFile
import app.models  # noqa: F401 - ensure package registration for test stubs
import app.core.deps  # noqa: F401 - ensure dependency module registration
import app.core.auth  # noqa: F401 - ensure auth module registration
from starlette.datastructures import MutableHeaders


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
                headers.pop("content-type", None)
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
