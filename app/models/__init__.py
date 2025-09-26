"""Pydantic models shared across the service and tests."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Representation of a stored document chunk used in tests."""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document body")
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentCreate(BaseModel):
    """Payload for creating a document entry."""

    id: Optional[str] = Field(None, description="Optional explicit identifier")
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)


__all__ = ["Document", "DocumentCreate"]
