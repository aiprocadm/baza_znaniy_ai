from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.api.constants import MAX_UPLOAD_SIZE_BYTES
from backend.app.api.routes.knowledge_base import upload_file
from backend.app.db.utils import init_db
from backend.app.main import create_app


def test_console_endpoints_cover_main_flows() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    status_response = client.get("/api/v1/status")
    assert status_response.status_code == 200
    assert "services" in status_response.json()

    search_response = client.post("/api/v1/search", json={"query": "onboarding", "top_k": 5})
    assert search_response.status_code == 200
    assert "results" in search_response.json()

    upload_response = client.post(
        "/api/v1/upload",
        files={"file": ("guide.md", b"hello kb", "text/markdown")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["name"] == "guide.md"

    files_response = client.get("/api/v1/files")
    assert files_response.status_code == 200
    assert isinstance(files_response.json(), list)


def test_admin_and_auth_endpoints() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    users_response = client.get("/api/v1/admin/users")
    assert users_response.status_code == 200

    create_user_response = client.post(
        "/api/v1/admin/users",
        json={"name": "Alice", "email": "alice@kb.ai", "roles": ["user"]},
    )
    assert create_user_response.status_code == 201
    user_id = create_user_response.json()["id"]

    patch_response = client.patch(f"/api/v1/admin/users/{user_id}", json={"status": "active"})
    assert patch_response.status_code == 200
    assert patch_response.json()["status"] == "active"

    settings_response = client.put(
        "/api/v1/admin/settings",
        json={
            "qdrant_url": "http://localhost:6333",
            "llm_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "ingestion_parallelism": 6,
            "allow_guest_access": True,
        },
    )
    assert settings_response.status_code == 200
    assert settings_response.json()["ingestion_parallelism"] == 6

    session_response = client.get("/api/v1/auth/session")
    assert session_response.status_code == 200

    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@kb.ai", "password": "secret"},
    )
    assert login_response.status_code == 200
    assert login_response.json()["email"] == "admin@kb.ai"

    refresh_response = client.post("/api/v1/auth/refresh")
    assert refresh_response.status_code == 200
    assert "token" in refresh_response.json()

    delete_response = client.delete(f"/api/v1/admin/users/{user_id}")
    assert delete_response.status_code == 204


def test_upload_rejects_invalid_files() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    empty_upload = client.post(
        "/api/v1/upload",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert empty_upload.status_code == 400
    assert empty_upload.json()["detail"] == "Uploaded file is empty"

    unsupported_upload = client.post(
        "/api/v1/upload",
        files={"file": ("archive.zip", b"123", "application/zip")},
    )
    assert unsupported_upload.status_code == 415
    assert "Unsupported media type" in unsupported_upload.json()["detail"]

    large_upload = client.post(
        "/api/v1/upload",
        files={"file": ("big.txt", b"a" * (MAX_UPLOAD_SIZE_BYTES + 1), "text/plain")},
    )
    assert large_upload.status_code == 413
    assert "File is too large" in large_upload.json()["detail"]


class _ChunkedUploadFile:
    def __init__(self, chunks: list[bytes], *, filename: str = "chunked.txt", content_type: str = "text/plain") -> None:
        self._chunks = chunks
        self._index = 0
        self.filename = filename
        self.content_type = content_type
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def test_upload_file_stops_early_when_size_limit_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, str]] = []

    def _add_file(*, name: str, size: int, mime_type: str):  # type: ignore[no-untyped-def]
        calls.append((name, size, mime_type))
        return {"id": "x", "name": name, "size": size, "mime_type": mime_type}

    monkeypatch.setattr("backend.app.api.routes.knowledge_base.runtime_store.add_file", _add_file)
    upload = _ChunkedUploadFile(chunks=[b"a" * MAX_UPLOAD_SIZE_BYTES, b"b"])

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(upload_file(file=upload))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 413
    assert calls == []
    assert upload.read_sizes == [1024 * 1024, 1024 * 1024]


def test_upload_file_accepts_small_chunked_file(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _add_file(*, name: str, size: int, mime_type: str):  # type: ignore[no-untyped-def]
        captured.update({"name": name, "size": size, "mime_type": mime_type})
        return {"id": "ok", "name": name, "size": size, "mime_type": mime_type}

    monkeypatch.setattr("backend.app.api.routes.knowledge_base.runtime_store.add_file", _add_file)
    upload = _ChunkedUploadFile(chunks=[b"hello", b" ", b"kb"])

    result = asyncio.run(upload_file(file=upload))  # type: ignore[arg-type]

    assert result["size"] == 8
    assert captured == {"name": "chunked.txt", "size": 8, "mime_type": "text/plain"}
