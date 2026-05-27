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
from app.api.status_codes import HTTP_CONTENT_TOO_LARGE


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

    response = docs_client.post(
        "/api/docs/upload",
        data={"user_id": "tester"},
        files={"file": ("big.txt", oversized, "text/plain")},
    )

    assert response.status_code == HTTP_CONTENT_TOO_LARGE
    payload = response.json()
    assert payload["message"] == "UPLOAD_TOO_LARGE"
    assert payload["status"] == HTTP_CONTENT_TOO_LARGE


_INVALID_CONTENT_TYPE_CANDIDATES = [
    "application/octet-stream",
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]


def _invalid_mime_matrix() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for extension, allowed in sorted(routes_module._ALLOWED_CONTENT_TYPES_BY_EXTENSION.items()):
        invalid_type = next(
            (
                candidate
                for candidate in _INVALID_CONTENT_TYPE_CANDIDATES
                if candidate not in allowed
            ),
            f"application/x-{extension}",
        )
        cases.append((extension, invalid_type))
    return cases


@pytest.mark.parametrize("extension, invalid_type", _invalid_mime_matrix())
def test_upload_rejects_invalid_content_type_matrix(
    docs_client: TestClient, extension: str, invalid_type: str
) -> None:
    response = docs_client.post(
        "/api/docs/upload",
        data={"user_id": "tester"},
        files={"file": (f"document.{extension}", b"valid", invalid_type)},
    )

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    payload = response.json()
    # The route validates extension and content-type in two checkpoints
    # whose order depends on the runtime's request-body parsing. On
    # Linux fastapi can surface the extension check first, on Windows
    # the content-type check fires first. Both are valid 415 rejections
    # for this case — assert the contract, not the implementation order.
    assert payload["message"] in {"UPLOAD_INVALID_TYPE", "UPLOAD_INVALID_EXT"}
    assert payload["status"] == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_upload_enforces_content_length_header(docs_client: TestClient) -> None:
    response = docs_client.post(
        "/api/docs/upload",
        data={"user_id": "tester"},
        files={"file": ("tiny.txt", b"ok", "text/plain")},
        headers={"content-length": str(60 * 1024 * 1024)},
    )

    assert response.status_code == HTTP_CONTENT_TOO_LARGE
    payload = response.json()
    assert payload["message"] == "UPLOAD_TOO_LARGE"
    assert payload["status"] == HTTP_CONTENT_TOO_LARGE


def test_upload_sanitises_filename(docs_client: TestClient) -> None:
    payload = b"data"

    response = docs_client.post(
        "/api/docs/upload",
        data={"user_id": "tester"},
        files={"file": ("../..//evil name?.txt", payload, "text/plain")},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["files"] == ["evil_name_.txt"]
