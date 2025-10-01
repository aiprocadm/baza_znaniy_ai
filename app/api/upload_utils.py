"""Shared helpers for working with FastAPI ``UploadFile`` objects."""

from __future__ import annotations

from tempfile import SpooledTemporaryFile
from typing import Any

from fastapi import UploadFile


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

    kwargs: dict[str, Any] = {"filename": filename, "file": file_obj}
    if content_type is not None:
        kwargs["content_type"] = content_type
    return UploadFile(**kwargs)
