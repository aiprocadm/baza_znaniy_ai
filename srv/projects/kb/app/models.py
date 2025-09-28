"""Pydantic models used by the service API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    ok: bool
    files: List[str]
    chunks: int


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    file: Optional[str]
    page: Optional[int]
    score: Optional[float]


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    conversation_id: str
    citations_insufficient: bool
    latency_ms: float
