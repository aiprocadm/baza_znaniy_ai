from __future__ import annotations

from importlib import reload
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url
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
        member = UserRecord(
            email="member@example.com",
            full_name="Member",
            tenant_slug="default",
            role=UserRole.MEMBER,
            is_active=True,
            hashed_password=hash_password("member-secret"),
        )
        session.add(member)
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


@pytest.mark.parametrize("db_scheme", ["sqlite", "sqlite+aiosqlite"])
def test_get_engine_sync_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db_scheme: str
) -> None:
    from app.models import file as file_models

    db_path = tmp_path / f"auth-{db_scheme.replace('+', '-')}.db"
    monkeypatch.setenv("DB_URL", f"{db_scheme}:///{db_path}")
    file_models.get_engine.cache_clear()

    engine = file_models.get_engine(create_schema=False)

    try:
        assert hasattr(engine, "dialect")
        assert hasattr(engine.dialect, "name")
        assert hasattr(engine.dialect, "driver")
        assert engine.dialect.name == "sqlite"
        assert engine.dialect.driver != "aiosqlite"
        assert getattr(engine.dialect, "is_async", False) is False

        assert hasattr(engine, "url")
        engine_url = engine.url
        engine_url_str = str(engine_url)
        assert engine_url_str.startswith("sqlite:")

        if hasattr(engine_url, "get_backend_name"):
            backend_name = engine_url.get_backend_name()
            driver_name = engine_url.get_driver_name()
        else:
            resolved_url = make_url(engine_url_str)
            if hasattr(resolved_url, "get_backend_name"):
                backend_name = resolved_url.get_backend_name()
                driver_name = resolved_url.get_driver_name()
            else:
                backend_name = engine_url_str.split(":", 1)[0]
                driver_name = backend_name.split("+", 1)[0]

        assert backend_name == "sqlite"
        assert driver_name in {"sqlite", "pysqlite"}

        assert hasattr(engine, "dispose")
        assert callable(engine.dispose)
        assert hasattr(engine, "connect")
        assert callable(engine.connect)
        connection = engine.connect()
        assert hasattr(connection, "execute")
    finally:
        engine.dispose()
        file_models.get_engine.cache_clear()


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


def test_login_bruteforce_rate_limit(auth_client: TestClient) -> None:
    for _ in range(5):
        response = auth_client.post(
            "/api/v1/auth/login",
            json={"email": "admin@example.com", "password": "wrong"},
        )
        assert response.status_code == 401
    limited = auth_client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "wrong"},
    )
    assert limited.status_code == 429


def test_revoked_access_token_rejected(auth_client: TestClient) -> None:
    login = auth_client.post("/api/v1/auth/login", json={"email": "admin@example.com", "password": "secret"})
    token = login.json()["access_token"]
    refresh = login.json()["refresh_token"]
    auth_client.post("/api/v1/auth/logout", json={"refresh_token": refresh}, headers={"Authorization": f"Bearer {token}"})
    response = auth_client.get("/api/v1/tenants", headers={"Authorization": f"Bearer {token}", "X-Tenant": "default"})
    assert response.status_code == 401


def test_privilege_escalation_blocked_for_member(auth_client: TestClient) -> None:
    login = auth_client.post(
        "/api/v1/auth/login",
        json={"email": "member@example.com", "password": "member-secret"},
    )
    member_access = login.json()["access_token"]
    response = auth_client.get(
        "/api/v1/tenants",
        headers={"Authorization": f"Bearer {member_access}", "X-Tenant": "default"},
    )
    assert response.status_code == 403
