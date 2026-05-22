"""Admin endpoint: GET /api/v1/admin/audit — read audit_log entries."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.audit_db import query_audit_log
from app.core.auth import require_admin_user
from app.core.deps import get_ingest_session
from app.models.audit import AuditLog

router = APIRouter(prefix="/admin", tags=["admin"])


class AuditLogItem(BaseModel):
    id: int
    timestamp: datetime
    event: str
    user_id: Optional[str] = None
    tenant: Optional[str] = None
    ip: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    status_code: Optional[int] = None
    correlation_id: Optional[str] = None

    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    items: list[AuditLogItem]
    total: int
    limit: int
    offset: int


@router.get("/audit", response_model=AuditLogResponse)
def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    event: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    session: Session = Depends(get_ingest_session),
    _admin=Depends(require_admin_user),
) -> AuditLogResponse:
    """Return paginated audit entries, newest first.

    Requires admin role. Filters: event name, user_id, tenant, time range.
    """
    items = query_audit_log(
        session,
        limit=limit,
        offset=offset,
        event=event,
        user_id=user_id,
        tenant=tenant,
        since=since,
        until=until,
    )

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
    total = len(list(session.exec(stmt).all()))

    return AuditLogResponse(
        items=[AuditLogItem.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )
