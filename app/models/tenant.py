"""Tenant API schemas and persistence exports."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.entities import TenantRecord, TenantStatus


class TenantBase(BaseModel):
    """Shared fields for tenant API payloads."""

    name: str = Field(..., min_length=1, max_length=200)
    is_active: bool = Field(default=True)
    contact_email: Optional[str] = Field(default=None, max_length=255)


class TenantCreate(TenantBase):
    """Payload for creating a new tenant via the API."""

    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


class TenantUpdate(BaseModel):
    """Payload for updating tenant attributes."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    is_active: Optional[bool] = None
    contact_email: Optional[str] = Field(default=None, max_length=255)


class TenantResponse(BaseModel):
    """Tenant data exposed by the API."""

    model_config = ConfigDict(from_attributes=True)

    slug: str
    name: str
    is_active: bool
    contact_email: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


__all__ = [
    "TenantBase",
    "TenantCreate",
    "TenantRecord",
    "TenantResponse",
    "TenantStatus",
    "TenantUpdate",
]
