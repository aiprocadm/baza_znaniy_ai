"""Tests for upload file coercion helpers."""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import HTTPException, UploadFile, status

from app.api.routes import _coerce_bytes, _coerce_upload_file
from app.api.v1.upload import _coerce_upload_argument


def _read(upload: UploadFile) -> bytes:
    return asyncio.run(upload.read())


def test_coerce_upload_file_from_dict_returns_original_instance() -> None:
    original = UploadFile(filename="sample.txt", file=io.BytesIO(b"payload"))
    wrapped = {"files": original}

    result = _coerce_upload_file(wrapped)

    assert result is original
    assert _read(result) == b"payload"


def test_coerce_upload_file_from_uploadfile_instance_is_passthrough() -> None:
    original = UploadFile(filename="passthrough.txt", file=io.BytesIO(b"data"))

    result = _coerce_upload_file(original)

    assert result is original
    assert _read(result) == b"data"


def test_coerce_upload_file_from_nested_list() -> None:
    result = _coerce_upload_file([["nested.txt", b"data"]])

    assert isinstance(result, UploadFile)
    assert result.filename == "nested.txt"
    assert _read(result) == b"data"


def test_coerce_upload_file_from_tuple_pair() -> None:
    result = _coerce_upload_file((("tuple.bin", b"binary"),))

    assert isinstance(result, UploadFile)
    assert result.filename == "tuple.bin"
    assert _read(result) == b"binary"


def test_coerce_upload_file_from_tuple_with_content_type() -> None:
    result = _coerce_upload_file(("typed.txt", b"payload", "text/plain"))

    assert isinstance(result, UploadFile)
    assert result.filename == "typed.txt"
    assert result.content_type == "text/plain"
    assert _read(result) == b"payload"


def test_coerce_upload_file_from_wrapped_dict_pair() -> None:
    result = _coerce_upload_file({"file": ("wrapped.txt", b"wrapped")})

    assert isinstance(result, UploadFile)
    assert result.filename == "wrapped.txt"
    assert _read(result) == b"wrapped"


def test_coerce_upload_file_with_empty_input_raises_http_exception() -> None:
    with pytest.raises(HTTPException) as excinfo:
        _coerce_upload_file([])

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "UPLOAD_INVALID_FILE"


def test_upload_v1_coerce_preserves_uploadfile() -> None:
    original = UploadFile(filename="keep.txt", file=io.BytesIO(b"payload"))

    result = _coerce_upload_argument(original)

    assert result is original
    assert _read(result) == b"payload"


def test_upload_v1_coerce_from_dict_builds_uploadfile() -> None:
    result = _coerce_upload_argument({"filename": "dict.txt", "content": b"bytes"})

    assert isinstance(result, UploadFile)
    assert result.filename == "dict.txt"
    assert _read(result) == b"bytes"


def test_upload_v1_coerce_from_tuple_builds_uploadfile() -> None:
    result = _coerce_upload_argument(("tuple.txt", b"tuple-bytes"))

    assert isinstance(result, UploadFile)
    assert result.filename == "tuple.txt"
    assert _read(result) == b"tuple-bytes"


def test_coerce_bytes_reads_from_start_and_restores_position() -> None:
    payload = io.BytesIO(b"abcdef")
    payload.seek(6)

    result = _coerce_bytes(payload)

    assert result == b"abcdef"
    assert payload.tell() == 6
