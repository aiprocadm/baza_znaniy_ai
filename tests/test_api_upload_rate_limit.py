from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import upload as upload_module


class _StubIngestService:
    async def register_file(
        self,
        tenant: str,
        path: str,
        *,
        filename: str,
        size: int,
        mime_type: str,
    ):
        record = SimpleNamespace(
            id="file-123",
            filename=filename,
            status="queued",
            path=path,
        )
        return record, True


@pytest.fixture(autouse=True)
def reset_rate_limit():
    upload_module._RATE_HISTORY.clear()
    yield
    upload_module._RATE_HISTORY.clear()


@pytest.fixture
def upload_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app = FastAPI()
    app.include_router(upload_module.router, prefix="/api/v1")

    app.dependency_overrides[upload_module.get_current_active_user] = lambda: SimpleNamespace(
        id="user"
    )
    app.dependency_overrides[upload_module.ensure_tenant_access] = lambda: "tenant"
    app.dependency_overrides[upload_module.get_data_dir] = lambda: tmp_path
    app.dependency_overrides[upload_module.get_ingest_service] = lambda: _StubIngestService()
    app.dependency_overrides[upload_module.get_upload_limits] = lambda: upload_module.UploadLimits(
        max_upload_mb=1,
        allowed_extensions={"pdf", "txt"},
    )

    return TestClient(app)


def test_upload_rate_limit_returns_429(
    upload_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upload_module, "_RATE_LIMIT", 1)
    monkeypatch.setattr(upload_module, "_RATE_WINDOW", 3600.0)

    response_ok = upload_client.post(
        "/api/v1/upload",
        files={"file": ("doc.pdf", b"hello", "application/pdf")},
    )
    assert response_ok.status_code == 201

    response_throttled = upload_client.post(
        "/api/v1/upload",
        files={"file": ("doc.pdf", b"hello", "application/pdf")},
    )
    assert response_throttled.status_code == 429
    assert response_throttled.json()["detail"] == "RATE_LIMIT"
