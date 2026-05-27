"""Database models shared across the ingestion and admin services."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import field_validator, model_validator
from sqlalchemy import Column, JSON, UniqueConstraint
from sqlmodel import Field, SQLModel

from app.core.datetime_utils import utc_now
from app.core.email import EmailValidationError, normalise_email
from app.models.sqlmodel_compat import install_stub_model_initializers


class TenantStatus(str):
    """Lifecycle states tracked for tenants."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class TenantRecord(SQLModel, table=True):
    """Primary tenant persistence model used throughout the application."""

    __tablename__ = "tenants"
    __table_args__ = (UniqueConstraint("slug", name="uq_tenants_slug"),)

    def __init__(self, **data: Any) -> None:
        slug = data.get("slug")
        tenant_id = data.get("tenant_id")
        if tenant_id in (None, "") and slug:
            data["tenant_id"] = str(slug)
        elif slug in (None, "") and tenant_id:
            data["slug"] = str(tenant_id)
        try:
            super().__init__(**data)
        except TypeError:  # pragma: no cover - SQLModel stubs can expose object.__init__
            for key, value in data.items():
                setattr(self, key, value)

    tenant_id: Optional[str] = Field(default=None, primary_key=True)
    slug: str = Field(index=True)
    name: Optional[str] = Field(default=None, index=True)
    status: str = Field(default=TenantStatus.ACTIVE, index=True)
    is_active: bool = Field(default=True, index=True)
    contact_email: Optional[str] = Field(default=None, index=True)
    storage_quota: int = Field(default=0, ge=0)
    storage_used: int = Field(default=0, ge=0)
    document_quota: int = Field(default=0, ge=0)
    document_count: int = Field(default=0, ge=0)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)

    @model_validator(mode="before")
    @classmethod
    def _populate_identifiers(cls, values: dict[str, Any]) -> dict[str, Any]:
        slug = values.get("slug")
        tenant_id = values.get("tenant_id")
        if tenant_id and not slug:
            values["slug"] = str(tenant_id)
        elif slug and not tenant_id:
            values["tenant_id"] = str(slug)
        return values

    @model_validator(mode="after")
    def _sync_identifiers(self) -> "TenantRecord":
        tenant_id = getattr(self, "tenant_id", None)
        slug = getattr(self, "slug", None)
        if tenant_id in (None, "") and slug:
            object.__setattr__(self, "tenant_id", str(slug))
        elif slug in (None, "") and tenant_id:
            object.__setattr__(self, "slug", str(tenant_id))
        return self

    @field_validator("slug")
    @classmethod
    def _normalise_slug(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("slug must not be empty")
        return value

    @field_validator("status")
    @classmethod
    def _normalise_status(cls, value: str) -> str:
        value = (value or TenantStatus.ACTIVE).strip().lower()
        if value not in {TenantStatus.ACTIVE, TenantStatus.INACTIVE, TenantStatus.SUSPENDED}:
            raise ValueError(f"invalid tenant status: {value}")
        return value


class UserStatus(str):
    """Lifecycle states tracked for users."""

    ACTIVE = "active"
    INVITED = "invited"
    DISABLED = "disabled"


class UserRecord(SQLModel, table=True):
    """User model associated with a tenant."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_id", name="uq_users_tenant_external"),
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
    )

    def __init__(self, **data: Any) -> None:
        tenant_id = data.get("tenant_id")
        tenant_slug = data.get("tenant_slug")
        if tenant_id in (None, "") and tenant_slug:
            data["tenant_id"] = str(tenant_slug)
        elif tenant_slug in (None, "") and tenant_id:
            data["tenant_slug"] = str(tenant_id)
        try:
            super().__init__(**data)
        except TypeError:  # pragma: no cover - SQLModel stubs can expose object.__init__
            for key, value in data.items():
                setattr(self, key, value)

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    tenant_slug: Optional[str] = Field(default=None, foreign_key="tenants.slug", index=True)
    external_id: Optional[str] = Field(default=None, index=True)
    email: Optional[str] = Field(default=None, index=True)
    full_name: Optional[str] = Field(default=None)
    role: str = Field(default="member", index=True)
    is_active: bool = Field(default=True, index=True)
    status: str = Field(default=UserStatus.ACTIVE, index=True)
    hashed_password: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    last_login_at: Optional[datetime] = Field(default=None)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: Optional[str]) -> Optional[str]:
        if value in (None, ""):
            return None
        try:
            return normalise_email(value)
        except EmailValidationError as exc:
            raise ValueError("INVALID_EMAIL_FORMAT") from exc


class JobStatus(str):
    """Ingest job lifecycle states."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(SQLModel, table=True):
    """Long-running background jobs tracked by the service."""

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
    payload: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)


class SettingRecord(SQLModel, table=True):
    """Tenant specific configuration stored in the database."""

    __tablename__ = "settings"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_settings_tenant_name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(foreign_key="tenants.tenant_id", index=True)
    name: str = Field(index=True)
    value: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    status: str = Field(default="active", index=True)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)
    updated_at: datetime = Field(default_factory=utc_now, nullable=False)


install_stub_model_initializers([TenantRecord, UserRecord, JobRecord, SettingRecord])


__all__ = [
    "JobRecord",
    "JobStatus",
    "SettingRecord",
    "TenantRecord",
    "TenantStatus",
    "UserRecord",
    "UserStatus",
]
