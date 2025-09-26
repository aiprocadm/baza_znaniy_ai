"""Pydantic models and domain objects for the knowledge base."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single knowledge base document."""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Raw document text")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentCreate(BaseModel):
    """Payload for creating a document."""

    id: Optional[str] = Field(None, description="Optional document id")
    content: str = Field(..., min_length=1, description="Document body")
    tags: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Query payload for retrieval."""

    question: str = Field(..., min_length=1, description="User question")
    limit: int = Field(default=3, ge=1, le=10, description="Number of hits to return")


class QueryResponse(BaseModel):
    """Retrieval results."""

    question: str
    matches: List[Document]
