"""SQLModel for audit_log table."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(index=True)
    event: str = Field(max_length=64, index=True)
    user_id: Optional[str] = Field(default=None, max_length=64, index=True)
    tenant: Optional[str] = Field(default=None, max_length=64, index=True)
    ip: Optional[str] = Field(default=None, max_length=45)
    request_path: Optional[str] = Field(default=None, max_length=512)
    request_method: Optional[str] = Field(default=None, max_length=8)
    status_code: Optional[int] = None
    payload_json: Optional[str] = None
    correlation_id: Optional[str] = Field(default=None, max_length=64, index=True)
