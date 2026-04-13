from __future__ import annotations

from fastapi.testclient import TestClient

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
