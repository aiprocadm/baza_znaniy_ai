"""Audit log database helpers — persistence and querying."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlmodel import Session, select

from app.models.audit import AuditLog


def persist_audit_event(
    session: Session,
    *,
    event: str,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    ip: Optional[str] = None,
    request_path: Optional[str] = None,
    request_method: Optional[str] = None,
    status_code: Optional[int] = None,
    payload: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> AuditLog:
    """Write one row to audit_log. Commits immediately."""
    entry = AuditLog(
        timestamp=timestamp or datetime.utcnow(),
        event=event,
        user_id=user_id,
        tenant=tenant,
        ip=ip,
        request_path=request_path,
        request_method=request_method,
        status_code=status_code,
        payload_json=json.dumps(payload) if payload is not None else None,
        correlation_id=correlation_id,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def query_audit_log(
    session: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    event: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[AuditLog]:
    """Return audit entries matching filters, newest first."""
    stmt = select(AuditLog)
    if event:
        stmt = stmt.where(AuditLog.event == event)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if tenant:
        stmt = stmt.where(AuditLog.tenant == tenant)
    if since:
        stmt = stmt.where(AuditLog.timestamp >= since)
    if until:
        stmt = stmt.where(AuditLog.timestamp <= until)
    stmt = stmt.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)
    return list(session.exec(stmt).all())
