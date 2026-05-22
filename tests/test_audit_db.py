"""Test the audit_db persistence helpers."""
from __future__ import annotations

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.audit_db import persist_audit_event, query_audit_log
from app.models.audit import AuditLog


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_persist_audit_event_writes_to_db(session: Session) -> None:
    persist_audit_event(
        session,
        event="login_success",
        user_id="alice",
        tenant="acme",
        ip="10.0.0.1",
        request_path="/api/v1/auth/login",
        request_method="POST",
        status_code=200,
        payload={"detail": "ok"},
        correlation_id="req-1",
    )
    rows = session.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "login_success"
    assert rows[0].user_id == "alice"
    assert rows[0].payload_json == '{"detail": "ok"}'


def test_persist_audit_event_minimal(session: Session) -> None:
    persist_audit_event(session, event="api_request")
    rows = session.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "api_request"
    assert rows[0].user_id is None


def test_query_audit_log_pagination(session: Session) -> None:
    for i in range(15):
        persist_audit_event(session, event="api_request", user_id=f"u{i}")
    page1 = query_audit_log(session, limit=10, offset=0)
    page2 = query_audit_log(session, limit=10, offset=10)
    assert len(page1) == 10
    assert len(page2) == 5


def test_query_audit_log_filter_by_event(session: Session) -> None:
    persist_audit_event(session, event="login_success", user_id="alice")
    persist_audit_event(session, event="api_request", user_id="alice")
    persist_audit_event(session, event="login_fail", user_id="bob")
    rows = query_audit_log(session, event="login_success")
    assert len(rows) == 1
    assert rows[0].user_id == "alice"


def test_query_audit_log_filter_by_user(session: Session) -> None:
    persist_audit_event(session, event="api_request", user_id="alice")
    persist_audit_event(session, event="api_request", user_id="bob")
    rows = query_audit_log(session, user_id="alice")
    assert len(rows) == 1
