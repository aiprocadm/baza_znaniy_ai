import asyncio
from importlib import reload
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from tests.service_stubs import install_service_stubs

install_service_stubs()

try:  # pragma: no cover - optional dependency
    from starlette.datastructures import UploadFile as StarletteUploadFile
except ModuleNotFoundError:  # pragma: no cover - fallback when Starlette missing
    StarletteUploadFile = None  # type: ignore[assignment]

from app.api import routes as routes_module
from app.api import upload_utils


@pytest.fixture()
def docs_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path / 'docs.db'}")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.delenv("UPLOAD_MAX_SIZE", raising=False)
    monkeypatch.delenv("UPLOAD_ALLOWED_EXTS", raising=False)

    from app.core import config as config_module
    from app.models import file as file_models

    config_module.get_settings.cache_clear()
    file_models.get_engine.cache_clear()

    import app.main as app_main

    reload(app_main)
    app = app_main.app
    app.dependency_overrides = {}

    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
        file_models.get_engine.cache_clear()
        config_module.get_settings.cache_clear()


@pytest.mark.skipif(StarletteUploadFile is None, reason="starlette is not installed")
def test_coerce_upload_file_handles_tuple_with_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"example-bytes"

    # Ensure the helper constructs a real UploadFile from Starlette when available.
    monkeypatch.setattr(upload_utils, "UploadFile", StarletteUploadFile)
    monkeypatch.setattr(routes_module, "UploadFile", StarletteUploadFile)

    result = routes_module._coerce_upload_file([("document.txt", payload)])

    assert isinstance(result, StarletteUploadFile)
    assert result.filename == "document.txt"
    assert asyncio.run(result.read()) == payload


def test_upload_rejects_oversized_body(docs_client: TestClient) -> None:
    oversized = b"x" * (1024 * 1024 + 1)

    expected_status = getattr(status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413)
    response = docs_client.post(
        "/api/docs/upload",
        data={"user_id": "tester"},
        files={"file": ("big.txt", oversized, "text/plain")},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == "UPLOAD_TOO_LARGE"
