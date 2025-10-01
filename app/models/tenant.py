"""Tenant data models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel, UniqueConstraint


class TenantBase(SQLModel):
    """Shared attributes for tenant persistence and API schemas."""

    name: str = SQLField(index=True, min_length=1, max_length=200)
    is_active: bool = SQLField(default=True)
    contact_email: Optional[str] = SQLField(default=None, max_length=255)


class TenantRecord(TenantBase, table=True):
    """SQLModel table describing tenants."""

    __tablename__ = "tenants"
    __table_args__ = (UniqueConstraint("slug", name="uq_tenants_slug"),)

    slug: str = SQLField(
        primary_key=True,
        index=True,
        min_length=1,
        max_length=100,
        regex=r"^[a-zA-Z0-9_-]+$",
    )
    created_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)


class TenantCreate(BaseModel):
    """Payload for creating a new tenant via the API."""

    name: str = Field(..., min_length=1, max_length=200)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    is_active: bool = Field(default=True)
    contact_email: Optional[str] = Field(default=None, max_length=255)


class TenantUpdate(BaseModel):
    """Payload for updating tenant attributes."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    is_active: Optional[bool] = None
    contact_email: Optional[str] = Field(default=None, max_length=255)


class TenantResponse(BaseModel):
    """Tenant data exposed by the API."""

    slug: str
    name: str
    is_active: bool
    contact_email: Optional[str]
    created_at: datetime
    updated_at: datetime


__all__ = [
    "TenantBase",
    "TenantCreate",
    "TenantRecord",
    "TenantResponse",
    "TenantUpdate",
]

