"""SQLModel tables describing tenants, users, jobs and settings."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel


class TenantStatus(str):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class TenantRecord(SQLModel, table=True):
    __tablename__ = "tenants"

    tenant_id: str = Field(primary_key=True)
    name: Optional[str] = Field(default=None)
    status: str = Field(default=TenantStatus.ACTIVE, index=True)
    storage_quota: int = Field(default=0, ge=0)
    storage_used: int = Field(default=0, ge=0)
    document_quota: int = Field(default=0, ge=0)
    document_count: int = Field(default=0, ge=0)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class UserStatus(str):
    ACTIVE = "active"
    INVITED = "invited"
    DISABLED = "disabled"


class UserRecord(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_id", name="uq_users_tenant_external"),
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    external_id: str = Field(index=True)
    email: Optional[str] = Field(default=None, index=True)
    role: str = Field(default="member", index=True)
    status: str = Field(default=UserStatus.ACTIVE, index=True)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    last_login_at: Optional[datetime] = Field(default=None)


class JobStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(SQLModel, table=True):
    __tablename__ = "jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    tenant_slug: Optional[str] = Field(default=None, index=True)
    job_type: str = Field(default="generic", index=True)
    status: str = Field(default=JobStatus.QUEUED, index=True)
    priority: int = Field(default=0)
    error: Optional[str] = Field(default=None)
    resource_id: Optional[str] = Field(default=None, index=True)
    attempt: int = Field(default=0, ge=0)
    payload: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)


class SettingRecord(SQLModel, table=True):
    __tablename__ = "settings"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_settings_tenant_name"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    name: str = Field(index=True)
    value: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    status: str = Field(default="active", index=True)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


__all__ = [
    "JobRecord",
    "JobStatus",
    "SettingRecord",
    "TenantRecord",
    "TenantStatus",
    "UserRecord",
    "UserStatus",
]
