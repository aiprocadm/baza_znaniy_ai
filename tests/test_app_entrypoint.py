"""Smoke tests for the FastAPI application entrypoint."""

from __future__ import annotations

import importlib
import sys
import types

import pytest
from fastapi import APIRouter, FastAPI


def test_uvicorn_entrypoint_exports_fastapi_app(monkeypatch: pytest.MonkeyPatch) -> None:
    stubbed_core_app = types.ModuleType("app.core.app")

    def _fake_create_app() -> FastAPI:
        return FastAPI()

    stubbed_core_app.create_app = _fake_create_app  # type: ignore[attr-defined]
    stubbed_core_app.__all__ = ["create_app"]  # type: ignore[attr-defined]

    stubbed_api_v1 = types.ModuleType("app.api.v1")
    stubbed_api_v1.router = APIRouter()

    monkeypatch.setitem(sys.modules, "app.core.app", stubbed_core_app)
    monkeypatch.setitem(sys.modules, "app.api.v1", stubbed_api_v1)
    if "app.api.main" in sys.modules:
        monkeypatch.delitem(sys.modules, "app.api.main", raising=False)

    module = importlib.import_module("app.api.main")

    assert hasattr(module, "app"), "Entry module must expose an 'app' attribute"
    assert isinstance(module.app, FastAPI)
