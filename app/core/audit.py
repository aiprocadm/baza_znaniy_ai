"""Security audit logging helpers.

log_security_event writes to the security.audit logger always. When a
DB session is provided via session=, it also persists to the audit_log
table for queryable history.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlmodel import Session

logger = logging.getLogger("security.audit")


def log_security_event(
    event: str,
    *,
    session: Optional["Session"] = None,
    user_id: Optional[str] = None,
    tenant: Optional[str] = None,
    ip: Optional[str] = None,
    request_path: Optional[str] = None,
    request_method: Optional[str] = None,
    status_code: Optional[int] = None,
    correlation_id: Optional[str] = None,
    **extra_fields: Any,
) -> None:
    """Record a security event to log + (optionally) DB.

    Backwards compatible: callers that pass only event=... and **fields
    behave as before — log-only. Callers that pass session= also persist
    to the audit_log table.
    """
    payload = {
        "event": event,
        "user_id": user_id,
        "tenant": tenant,
        "ip": ip,
        "request_path": request_path,
        "request_method": request_method,
        "status_code": status_code,
        "correlation_id": correlation_id,
        **extra_fields,
    }
    payload_clean = {k: v for k, v in payload.items() if v is not None}
    logger.info("security_event", extra={"security_event": payload_clean})

    if session is not None:
        from app.core.audit_db import persist_audit_event

        persist_audit_event(
            session,
            event=event,
            user_id=user_id,
            tenant=tenant,
            ip=ip,
            request_path=request_path,
            request_method=request_method,
            status_code=status_code,
            payload=extra_fields if extra_fields else None,
            correlation_id=correlation_id,
        )
