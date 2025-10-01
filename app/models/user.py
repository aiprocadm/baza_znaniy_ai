"""User models for persistence and API payloads."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.entities import UserRecord as UserRecord


class UserRole(str, Enum):
    """Available roles for application users."""

    ADMIN = "admin"
    MANAGER = "manager"
    MEMBER = "member"


class UserCreate(BaseModel):
    """Payload used to create a new user."""

    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=200)
    password: str = Field(..., min_length=8, max_length=200)
    role: UserRole = Field(default=UserRole.MEMBER)
    is_active: bool = Field(default=True)
    tenant_slug: str = Field(..., min_length=1, max_length=100)


class UserUpdate(BaseModel):
    """Payload used to update an existing user."""

    full_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    password: Optional[str] = Field(default=None, min_length=8, max_length=200)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """Representation of a user returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    tenant_slug: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]


__all__ = [
    "UserCreate",
    "UserRecord",
    "UserResponse",
    "UserRole",
    "UserUpdate",
]
