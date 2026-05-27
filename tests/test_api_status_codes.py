"""Unit tests for compatibility helpers mapping HTTP status codes."""

from __future__ import annotations

import importlib
from http import HTTPStatus

import pytest


pytestmark = pytest.mark.filterwarnings("ignore:'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated")


@pytest.fixture
def status_module():
    """Expose the FastAPI status stub so tests can manipulate attributes."""

    status_codes = importlib.import_module("app.api.status_codes")

    try:
        yield status_codes.status
    finally:
        importlib.reload(status_codes)


def test_prefer_status_returns_first_available(monkeypatch, status_module):
    """When a named constant exists it should be preferred over the fallback."""

    sentinel_value = 499
    monkeypatch.setattr(status_module, "HTTP_CUSTOM_STATUS", sentinel_value, raising=False)

    from app.api import status_codes

    assert (
        status_codes._prefer_status("HTTP_CUSTOM_STATUS", fallback=int(HTTPStatus.OK))
        == sentinel_value
    )


def test_constants_fall_back_when_names_missing(monkeypatch, status_module):
    """Reloading the module without matching constants should use the HTTP fallback."""

    sentinel = object()
    for attribute in ("HTTP_422_UNPROCESSABLE_CONTENT", "HTTP_422_UNPROCESSABLE_ENTITY"):
        if hasattr(status_module, attribute):
            monkeypatch.setattr(status_module, attribute, sentinel, raising=False)

    status_codes = importlib.reload(importlib.import_module("app.api.status_codes"))

    assert status_codes.HTTP_UNPROCESSABLE_CONTENT == int(HTTPStatus.UNPROCESSABLE_ENTITY)
