"""Pydantic models used by chat endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class ChatIn(BaseModel):
    user_id: str
    message: str
    conversation_id: str | None = None


__all__ = ["ChatIn"]
