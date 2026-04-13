from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.api.constants import MAX_UPLOAD_SIZE_BYTES
from backend.app.db.utils import init_db
from backend.app.main import create_app

ADMIN_AUTH_HEADERS = {"Authorization": "Bearer kb_admin_token"}


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

    users_response = client.get("/api/v1/admin/users", headers=ADMIN_AUTH_HEADERS)
    assert users_response.status_code == 200

    create_user_response = client.post(
        "/api/v1/admin/users",
        json={"name": "Alice", "email": "alice@kb.ai", "roles": ["user"]},
        headers=ADMIN_AUTH_HEADERS,
    )
    assert create_user_response.status_code == 201
    user_id = create_user_response.json()["id"]

    patch_response = client.patch(
        f"/api/v1/admin/users/{user_id}",
        json={"status": "active"},
        headers=ADMIN_AUTH_HEADERS,
    )
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
        headers=ADMIN_AUTH_HEADERS,
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

    delete_response = client.delete(f"/api/v1/admin/users/{user_id}", headers=ADMIN_AUTH_HEADERS)
    assert delete_response.status_code == 204


def test_login_rejects_invalid_password() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@kb.ai", "password": "wrong-secret"},
    )

    assert login_response.status_code == 401
    assert login_response.json()["detail"] == "Invalid credentials"


def test_admin_endpoints_require_token() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    no_token_response = client.get("/api/v1/admin/users")
    assert no_token_response.status_code in (401, 403)


def test_admin_endpoints_allow_admin_role() -> None:
    init_db()
    app = create_app()
    client = TestClient(app)

    users_response = client.get("/api/v1/admin/users", headers=ADMIN_AUTH_HEADERS)
    assert users_response.status_code == 200

    create_user_response = client.post(
        "/api/v1/admin/users",
        json={"name": "Bob", "email": "bob@kb.ai", "roles": ["user"]},
        headers=ADMIN_AUTH_HEADERS,
    )
    assert create_user_response.status_code == 201
    created_user_id = create_user_response.json()["id"]

    delete_user_response = client.delete(
        f"/api/v1/admin/users/{created_user_id}",
        headers=ADMIN_AUTH_HEADERS,
    )
    assert delete_user_response.status_code == 204


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
