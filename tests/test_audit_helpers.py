"""Test the extended audit helpers (log + optional DB persistence)."""
from __future__ import annotations

import logging

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.core.audit import log_security_event
from app.models.audit import AuditLog


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_log_security_event_emits_log_record(caplog: pytest.LogCaptureFixture) -> None:
    """Existing behaviour: log line is always written."""
    with caplog.at_level(logging.INFO, logger="security.audit"):
        log_security_event("login_success", user_id="alice")
    assert any(
        record.name == "security.audit"
        and getattr(record, "security_event", None) is not None
        for record in caplog.records
    )


def test_log_security_event_persists_when_session_provided(session) -> None:
    """New behaviour: when session= is passed, persist to DB too."""
    log_security_event(
        "login_fail",
        session=session,
        user_id="alice",
        ip="10.0.0.1",
    )
    rows = session.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "login_fail"
    assert rows[0].user_id == "alice"
    assert rows[0].ip == "10.0.0.1"


def test_log_security_event_no_db_when_no_session(caplog: pytest.LogCaptureFixture) -> None:
    """Backwards compat: no session means log-only, no DB call."""
    with caplog.at_level(logging.INFO, logger="security.audit"):
        log_security_event("login_success", user_id="alice")
    assert len(caplog.records) >= 1


def test_log_security_event_extra_fields_via_kwargs(session) -> None:
    """Existing callers pass arbitrary kwargs (email=, revoked_jti=) — must still work."""
    log_security_event(
        "login_fail",
        session=session,
        email="alice@example.com",
    )
    rows = session.exec(select(AuditLog)).all()
    assert len(rows) == 1
    assert rows[0].event == "login_fail"
    assert rows[0].payload_json is not None
    assert "alice@example.com" in rows[0].payload_json
