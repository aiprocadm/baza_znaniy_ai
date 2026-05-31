"""Test the admin audit log endpoint."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api.v1.admin_audit import router
from app.core.audit_db import persist_audit_event
from app.core.datetime_utils import utc_now_naive


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


def test_purge_removes_entries_older_than_days(app, engine):
    with Session(engine) as s:
        persist_audit_event(s, event="stale", timestamp=utc_now_naive() - timedelta(days=40))
        persist_audit_event(s, event="fresh", timestamp=utc_now_naive() - timedelta(days=5))

    client = TestClient(app)
    resp = client.post("/admin/audit/purge?days=30")

    assert resp.status_code == 200
    data = resp.json()
    assert data["removed"] == 1
    assert data["retention_days"] == 30

    events = {item["event"] for item in client.get("/admin/audit").json()["items"]}
    assert "fresh" in events
    assert "stale" not in events


def test_purge_defaults_to_configured_retention(app, engine, monkeypatch):
    from app.core import config

    monkeypatch.setenv("AUDIT_LOG_RETENTION_DAYS", "30")
    config.get_settings.cache_clear()
    try:
        with Session(engine) as s:
            persist_audit_event(s, event="stale", timestamp=utc_now_naive() - timedelta(days=40))
            persist_audit_event(s, event="fresh", timestamp=utc_now_naive() - timedelta(days=5))

        client = TestClient(app)
        resp = client.post("/admin/audit/purge")

        assert resp.status_code == 200
        data = resp.json()
        assert data["retention_days"] == 30
        assert data["removed"] == 1
    finally:
        config.get_settings.cache_clear()


def test_purge_with_zero_days_is_noop(app, engine):
    with Session(engine) as s:
        persist_audit_event(s, event="stale", timestamp=utc_now_naive() - timedelta(days=400))

    client = TestClient(app)
    resp = client.post("/admin/audit/purge?days=0")

    assert resp.status_code == 200
    assert resp.json()["removed"] == 0
    assert len(client.get("/admin/audit").json()["items"]) == 1


def test_purge_records_audit_event(app, engine):
    with Session(engine) as s:
        persist_audit_event(s, event="stale", timestamp=utc_now_naive() - timedelta(days=40))
        persist_audit_event(s, event="fresh", timestamp=utc_now_naive() - timedelta(days=5))

    client = TestClient(app)
    resp = client.post("/admin/audit/purge?days=30")
    assert resp.json()["removed"] == 1

    # Destroying audit history is itself an auditable action.
    items = client.get("/admin/audit").json()["items"]
    purges = [item for item in items if item["event"] == "audit_log_purged"]
    assert len(purges) == 1
    assert purges[0]["user_id"] == "test-admin"


def test_purge_noop_records_no_audit_event(app, engine):
    with Session(engine) as s:
        persist_audit_event(s, event="stale", timestamp=utc_now_naive() - timedelta(days=400))

    client = TestClient(app)
    client.post("/admin/audit/purge?days=0")

    items = client.get("/admin/audit").json()["items"]
    assert [item["event"] for item in items] == ["stale"]
