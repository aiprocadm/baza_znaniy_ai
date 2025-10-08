"""Helpers for coercing multipart upload payloads in FastAPI test stubs."""

from __future__ import annotations

from tempfile import SpooledTemporaryFile
from typing import Any

from . import UploadFile

__all__ = [
    "build_upload_file",
    "coerce_uploads",
    "ensure_list",
    "normalise_file_entries",
]


def normalise_file_entries(value: Any) -> list[Any]:
    """Normalise ``files=`` entries into a list of upload descriptors."""

    if isinstance(value, list):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], (list, tuple, UploadFile)):
        return list(value)
    return [value]


def ensure_list(value: Any) -> list[Any]:
    """Return ``value`` as a list without copying if already a list."""

    return value if isinstance(value, list) else [value]


def coerce_uploads(value: Any) -> list[UploadFile]:
    """Coerce ``files`` entries into ``UploadFile`` instances."""

    uploads: list[UploadFile] = []
    for entry in normalise_file_entries(value):
        uploads.append(build_upload_file(entry))
    return uploads


def build_upload_file(entry: Any) -> UploadFile:
    """Build an ``UploadFile`` compatible object from tuple/list descriptors."""

    if isinstance(entry, UploadFile):
        return entry

    if isinstance(entry, (list, tuple)):
        filename = entry[0] if entry else "uploaded"
        content = entry[1] if len(entry) > 1 else b""
        content_type = entry[2] if len(entry) > 2 else None
    else:
        filename = str(entry)
        content = b""
        content_type = None

    if hasattr(content, "read"):
        file_obj = content
        if hasattr(file_obj, "seek"):
            try:
                file_obj.seek(0)
            except Exception:  # pragma: no cover - defensive
                pass
    else:
        if isinstance(content, (bytes, bytearray, memoryview)):
            data = bytes(content)
        elif content is None:
            data = b""
        else:
            data = str(content).encode()
        file_obj = SpooledTemporaryFile(mode="w+b")
        if data:
            file_obj.write(data)
        file_obj.seek(0)

    kwargs: dict[str, Any] = {"filename": filename, "file": file_obj}
    if content_type is not None:
        kwargs["content_type"] = content_type
    return UploadFile(**kwargs)
