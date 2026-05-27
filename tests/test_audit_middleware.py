"""Test the request audit middleware."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.audit_middleware import AuditMiddleware
from app.core.datetime_utils import utc_now_naive
from app.models.audit import AuditLog


@pytest.fixture
def engine():
    # StaticPool so all sessions share the same in-memory DB connection.
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def app(engine):
    fastapi_app = FastAPI()

    def session_factory():
        return Session(engine)

    fastapi_app.add_middleware(
        AuditMiddleware, session_factory=session_factory, path_prefix="/api/kb"
    )

    @fastapi_app.get("/api/kb/health")
    def health():
        return {"status": "ok"}

    @fastapi_app.get("/api/v1/admin/ping")
    def admin_ping():
        return {"pong": True}

    return fastapi_app


def test_middleware_logs_kb_request(app, engine):
    client = TestClient(app)
    before = utc_now_naive() - timedelta(seconds=1)
    resp = client.get("/api/kb/health")
    assert resp.status_code == 200

    with Session(engine) as s:
        rows = s.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "api_request"
    assert rows[0].request_path == "/api/kb/health"
    assert rows[0].request_method == "GET"
    assert rows[0].status_code == 200
    assert rows[0].timestamp >= before


def test_middleware_ignores_non_kb_paths(app, engine):
    client = TestClient(app)
    resp = client.get("/api/v1/admin/ping")
    assert resp.status_code == 200

    with Session(engine) as s:
        rows = s.exec(select(AuditLog)).all()
    assert len(rows) == 0
