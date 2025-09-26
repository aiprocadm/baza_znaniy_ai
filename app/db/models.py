from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.session import Base

try:  # pragma: no cover - optional optimisation for PostgreSQL
    from sqlalchemy.dialects.postgresql import JSONB as JSONType  # type: ignore
except Exception:  # pragma: no cover - fallback when dialect not available
    JSONType = JSON  # type: ignore


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    STAFF = "staff"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, name="user_role"), nullable=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    logs: Mapped[List["ChatLog"]] = relationship("ChatLog", back_populates="user")

    @property
    def login(self) -> str:
        return self.username

    @login.setter
    def login(self, value: str) -> None:
        self.username = value


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    response_summary: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Any] = mapped_column(JSON().with_variant(JSONType, "postgresql"), nullable=False, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    user: Mapped[User] = relationship("User", back_populates="logs")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            "ChatLog(id={!r}, user_id={!r}, conversation_id={!r}, latency_ms={!r})".format(
                self.id,
                self.user_id,
                self.conversation_id,
                self.latency_ms,
            )
        )


__all__ = ["User", "UserRole", "ChatLog"]
