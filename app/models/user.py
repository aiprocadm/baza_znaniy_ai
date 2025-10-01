"""User data models and helpers."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field
from sqlmodel import Field as SQLField, SQLModel


class UserRole(str, Enum):
    """Available roles for application users."""

    ADMIN = "admin"
    MANAGER = "manager"
    MEMBER = "member"


class UserBase(SQLModel):
    """Common fields shared by persistence and API models."""

    email: EmailStr = SQLField(index=True, unique=True)
    full_name: str = SQLField(min_length=1, max_length=200)
    role: UserRole = SQLField(default=UserRole.MEMBER)
    is_active: bool = SQLField(default=True)
    tenant_slug: str = SQLField(foreign_key="tenants.slug", index=True)


class UserRecord(UserBase, table=True):
    """SQLModel table storing user accounts."""

    __tablename__ = "users"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    hashed_password: str = SQLField(min_length=1)
    last_login_at: Optional[datetime] = SQLField(default=None, nullable=True)
    created_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow, nullable=False)



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

    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    is_active: bool
    tenant_slug: str
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

