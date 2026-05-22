"""Test the admin audit log endpoint."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api.v1.admin_audit import router
from app.core.audit_db import persist_audit_event


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
    return fastapi_app


def test_get_audit_returns_empty_on_fresh_db(app):
    client = TestClient(app)
    resp = client.get("/admin/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


def test_get_audit_returns_persisted_rows(app, engine):
    with Session(engine) as s:
        for i in range(3):
            persist_audit_event(s, event="api_request", user_id=f"u{i}")

    client = TestClient(app)
    resp = client.get("/admin/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    assert data["total"] == 3


def test_get_audit_filter_by_event(app, engine):
    with Session(engine) as s:
        persist_audit_event(s, event="login_success", user_id="alice")
        persist_audit_event(s, event="api_request", user_id="alice")

    client = TestClient(app)
    resp = client.get("/admin/audit?event=login_success")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["event"] == "login_success"


def test_get_audit_pagination(app, engine):
    with Session(engine) as s:
        for i in range(15):
            persist_audit_event(s, event="api_request", user_id=f"u{i}")

    client = TestClient(app)
    resp = client.get("/admin/audit?limit=10&offset=0")
    data = resp.json()
    assert len(data["items"]) == 10

    resp = client.get("/admin/audit?limit=10&offset=10")
    data = resp.json()
    assert len(data["items"]) == 5
