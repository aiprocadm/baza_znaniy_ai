"""Regression tests for the dynamic ``UploadFile.content_type`` setter.

These exercise ``app.api.upload_utils`` against the **real** Starlette
``MutableHeaders``. That distinction matters: the dict-based ``MutableHeaders``
stub in ``tests/test_api_v1_upload.py`` has a ``.pop`` method, but the real
Starlette type does not. The former ``headers.pop("content-type", None)``
implementation therefore raised ``AttributeError`` at runtime, which was
swallowed by a bare ``except Exception``, leaving the header silently in place.
The fix uses ``del headers["content-type"]`` instead.
"""

from __future__ import annotations

import pytest

# The bug only reproduces against the real Starlette MutableHeaders (no
# ``.pop``). If a dict-based stub is already installed (e.g. by another test
# module that ran first), skip rather than assert against the wrong type.
md = pytest.importorskip("starlette.datastructures")
if hasattr(md.MutableHeaders, "pop"):  # pragma: no cover - stub environment
    pytest.skip(
        "real Starlette MutableHeaders required (stub provides .pop)",
        allow_module_level=True,
    )

from fastapi import UploadFile  # noqa: E402
from app.api import upload_utils  # noqa: E402


def test_setting_content_type_none_removes_header() -> None:
    upload: UploadFile = upload_utils.create_upload_file("doc.txt", b"data", "text/plain")
    assert upload.content_type == "text/plain"
    assert "content-type" in upload.headers

    upload.content_type = None  # type: ignore[assignment]

    assert "content-type" not in upload.headers
    assert upload.content_type is None


def test_setting_content_type_none_is_idempotent_when_absent() -> None:
    upload: UploadFile = upload_utils.create_upload_file("doc.txt", b"data")

    # Clearing an already-absent header must not raise (KeyError is handled).
    upload.content_type = None  # type: ignore[assignment]

    assert "content-type" not in upload.headers
    assert upload.content_type is None
