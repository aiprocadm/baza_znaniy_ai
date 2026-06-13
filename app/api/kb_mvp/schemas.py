"""Pydantic request/response models for the MVP /api/kb endpoints."""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from app.services.kb_store import (
    DEFAULT_HISTORY_LIMIT,
    MAX_CONVERSATION_TITLE,
    MAX_QUERY_LEN,
    MAX_TEXT_LEN,
)


class DocumentCreate(BaseModel):
    """Payload accepted by ``POST /api/kb/documents``."""

    title: str = Field(default="", max_length=300)
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("text is empty")
        if len(cleaned) > MAX_TEXT_LEN:
            raise ValueError(f"text exceeds {MAX_TEXT_LEN} characters")
        return cleaned

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        return (value or "").strip()


class DocumentOut(BaseModel):
    id: int
    title: str
    text: Optional[str] = None
    created_at: str
    chunks: int
    source: str = "text"
    filename: Optional[str] = None
    mime_type: Optional[str] = None


class DocumentListItem(BaseModel):
    id: int
    title: str
    created_at: str
    chunks: int
    source: str = "text"
    filename: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("query is empty")
        return cleaned


class HitOut(BaseModel):
    """A single ranked chunk — used by both ``/search`` and ``/ask``."""

    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
    source: str = "text"
    filename: Optional[str] = None
    page: Optional[int] = None
    has_original: bool = False


class RerankInfo(BaseModel):
    """Reranker diagnostics returned with /search and /ask responses."""

    enabled: bool
    used: bool = False
    model: Optional[str] = None
    candidates: int = 0
    elapsed_ms: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    hits: List[HitOut]
    rerank: Optional[RerankInfo] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
    top_k: int = Field(default=4, ge=1, le=20)
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation UUID. Omit to start a new one and receive its id in the response.",
        max_length=64,
    )
    history_limit: int = Field(
        default=DEFAULT_HISTORY_LIMIT,
        ge=0,
        le=50,
        description="How many previous messages to include in the LLM prompt context.",
    )

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("question is empty")
        return cleaned

    @field_validator("conversation_id")
    @classmethod
    def _strip_conv_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class RetrievalReasonOut(BaseModel):
    reason: str
    severity: str
    detail: str = ""


class RetrievalReportOut(BaseModel):
    degraded: bool
    severity: str
    reasons: List[RetrievalReasonOut] = Field(default_factory=list)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[HitOut]
    provider: str
    model: Optional[str] = None
    elapsed_ms: Optional[float] = None
    rerank: Optional[RerankInfo] = None
    retrieval: Optional[RetrievalReportOut] = None
    conversation_id: str = Field(
        ..., description="Conversation UUID — same as request, or freshly created."
    )


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=MAX_CONVERSATION_TITLE)


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=MAX_CONVERSATION_TITLE)


class MessageOut(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: str
    sources: List[HitOut] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None


class ConversationDetail(ConversationOut):
    messages: List[MessageOut] = Field(default_factory=list)
