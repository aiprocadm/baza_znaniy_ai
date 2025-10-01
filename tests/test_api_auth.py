from __future__ import annotations

from importlib import reload
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from tests.service_stubs import install_service_stubs

install_service_stubs()

from app.core import deps as core_deps

from app.models.tenant import TenantRecord
from app.models.user import UserRecord, UserRole
from app.security import hash_password


@pytest.fixture()
def auth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path / 'auth.db'}")
    monkeypatch.setenv("SECRET_KEY", "integration-secret")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "5")

    from app.core import config as config_module
    from app.models import file as file_models

    config_module.get_settings.cache_clear()
    file_models.get_engine.cache_clear()

    import app.main as app_main

    reload(app_main)
    app = app_main.app

    client = TestClient(app)

    def session_override():
        service = app.state.ingest_service
        with Session(service.engine) as session:
            yield session

    app.dependency_overrides = {
        core_deps.get_ingest_service: lambda: app.state.ingest_service,
        core_deps.get_ingest_session: session_override,
    }

    with Session(app.state.ingest_service.engine) as session:
        tenant = TenantRecord(slug="default", name="Default", is_active=True)
        session.add(tenant)
        session.commit()

        user = UserRecord(
            email="admin@example.com",
            full_name="Admin",
            tenant_slug="default",
            role=UserRole.ADMIN,
            is_active=True,
            hashed_password=hash_password("secret"),
        )
        session.add(user)
        session.commit()

    try:
        yield client
    finally:
        client.close()
        file_models.get_engine.cache_clear()
        config_module.get_settings.cache_clear()


def test_login_refresh_and_logout_flow(auth_client: TestClient) -> None:
    login_response = auth_client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "secret"},
    )
    assert login_response.status_code == 200
    tokens = login_response.json()
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    tenants_response = auth_client.get(
        "/api/v1/tenants",
        headers={
            "Authorization": f"Bearer {access_token}",
            "X-Tenant": "default",
        },
    )
    assert tenants_response.status_code == 200

    refresh_response = auth_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_response.status_code == 200
    new_tokens = refresh_response.json()
    new_access = new_tokens["access_token"]

    logout_response = auth_client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": refresh_token},
        headers={"Authorization": f"Bearer {new_access}"},
    )
    assert logout_response.status_code == 200

    reuse_response = auth_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert reuse_response.status_code == 401

    tenants_after_logout = auth_client.get(
        "/api/v1/tenants",
        headers={
            "Authorization": f"Bearer {new_access}",
            "X-Tenant": "default",
        },
    )
    assert tenants_after_logout.status_code == 401

    tenants_unauth = auth_client.get("/api/v1/tenants")
    assert tenants_unauth.status_code == 401
