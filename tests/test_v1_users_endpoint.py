"""Characterization test for the v1 create_user endpoint.

Pins the UserResponse shape so the mypy-cleanup refactor in app/api/v1/users.py
(UserRole coercion + casts) provably does not change caller-visible behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api.v1.users import router
from app.models.entities import TenantRecord


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def app(engine):
    from app.core.auth import require_admin_user
    from app.core.deps import get_ingest_session

    fastapi_app = FastAPI()
    fastapi_app.include_router(router)

    def fake_admin():
        return {"id": "test-admin", "role": "admin"}

    def fake_session():
        with Session(engine) as s:
            yield s

    fastapi_app.dependency_overrides[require_admin_user] = fake_admin
    fastapi_app.dependency_overrides[get_ingest_session] = fake_session

    # Order-independence for the full-suite run: other tests reload MVP app
    # modules and globally monkeypatch fastapi's TestClient, which can leave this
    # app's dependency_overrides keyed by a stale get_ingest_session object so the
    # *real* dependency runs. Pointing state.ingest_service at the same in-memory
    # engine makes create_user get a session on THIS test's DB whether or not the
    # override is honoured. (In isolation the override wins; this is belt-and-braces.)
    fastapi_app.state.ingest_service = SimpleNamespace(engine=engine)
    return fastapi_app


def test_create_user_returns_expected_response(app, engine):
    with Session(engine) as s:
        s.add(TenantRecord(tenant_id="acme", slug="acme", name="Acme Inc"))
        s.commit()

    client = TestClient(app)
    resp = client.post(
        "/users",
        json={
            "email": "alice@example.com",
            "full_name": "Alice Example",
            "password": "hunter2hunter2",
            "role": "manager",
            "is_active": True,
            "tenant_slug": "acme",
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "alice@example.com"
    assert body["full_name"] == "Alice Example"
    assert body["role"] == "manager"
    assert body["is_active"] is True
    assert body["tenant_slug"] == "acme"
    assert isinstance(body["id"], int) and body["id"] >= 1
    # Password must never be echoed back.
    assert "password" not in body and "hashed_password" not in body
