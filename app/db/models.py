from __future__ import annotations

        codex/add-postgresql-access-layer-and-auth-features
import enum

from sqlalchemy import Boolean, Enum, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    STAFF = "staff"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, name="user_role"), nullable=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(255), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    response_summary: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Any] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ChatLog(id={self.id!r}, user_id={self.user_id!r}, conversation_id={self.conversation_id!r}, "
            f"latency_ms={self.latency_ms!r})"
        )
        main
