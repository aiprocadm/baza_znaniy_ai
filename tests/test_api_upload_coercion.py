"""Tests for upload file coercion helpers."""

from __future__ import annotations

import io

import pytest
from fastapi import HTTPException, UploadFile, status

from app.api.routes import _coerce_upload_file


@pytest.mark.anyio
async def test_coerce_upload_file_from_dict_returns_original_instance() -> None:
    original = UploadFile(filename="sample.txt", file=io.BytesIO(b"payload"))
    wrapped = {"files": original}

    result = _coerce_upload_file(wrapped)

    assert result is original
    assert await result.read() == b"payload"


@pytest.mark.anyio
async def test_coerce_upload_file_from_nested_list() -> None:
    result = _coerce_upload_file([["nested.txt", b"data"]])

    assert isinstance(result, UploadFile)
    assert result.filename == "nested.txt"
    assert await result.read() == b"data"


@pytest.mark.anyio
async def test_coerce_upload_file_from_tuple_pair() -> None:
    result = _coerce_upload_file((("tuple.bin", b"binary"),))

    assert isinstance(result, UploadFile)
    assert result.filename == "tuple.bin"
    assert await result.read() == b"binary"


def test_coerce_upload_file_with_empty_input_raises_http_exception() -> None:
    with pytest.raises(HTTPException) as excinfo:
        _coerce_upload_file([])

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "UPLOAD_INVALID_FILE"
