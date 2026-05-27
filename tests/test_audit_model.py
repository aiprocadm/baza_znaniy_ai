"""Test the AuditLog SQLModel definition."""

from __future__ import annotations

from datetime import datetime

from app.core.datetime_utils import utc_now_naive
from app.models.audit import AuditLog


def test_audit_log_instantiation() -> None:
    """AuditLog can be constructed with required fields."""
    entry = AuditLog(
        timestamp=datetime(2026, 5, 22, 12, 0, 0),
        event="login_success",
        user_id="alice",
        tenant="acme",
        ip="192.168.1.1",
        request_path="/api/v1/auth/login",
        request_method="POST",
        status_code=200,
        payload_json='{"detail": "ok"}',
        correlation_id="req-abc-123",
    )
    assert entry.event == "login_success"
    assert entry.user_id == "alice"
    assert entry.timestamp.year == 2026


def test_audit_log_minimal_fields() -> None:
    """AuditLog requires only event and timestamp."""
    entry = AuditLog(
        timestamp=utc_now_naive(),
        event="api_request",
    )
    assert entry.user_id is None
    assert entry.tenant is None
