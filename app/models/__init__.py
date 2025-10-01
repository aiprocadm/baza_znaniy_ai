"""Pydantic models shared across the service and tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from .lora import LoraLoadRequest, LoraStatusResponse, LoraUnloadRequest

class Document(BaseModel):
    """Representation of a stored document chunk used across the service."""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document body")
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentCreate(BaseModel):
    """Payload for creating a document entry."""

    id: Optional[str] = Field(None, description="Optional explicit identifier")
    content: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response returned after an upload request."""

    file_id: str = Field(..., description="Assigned file identifier")
    filename: str = Field(..., description="Original file name")
    tenant: str = Field(..., description="Tenant that owns the file")
    status: str = Field(..., description="Current ingestion status")
    queued: bool = Field(True, description="Whether the file is queued for ingestion")


class IngestRequest(BaseModel):
    """Payload for triggering ingestion of a file."""

    file_id: str = Field(..., description="Identifier returned by the upload endpoint")
    force: bool = Field(False, description="Re-run ingestion even if already completed")


class IngestResponse(BaseModel):
    """Status information about an ingestion job."""

    file_id: str
    status: str
    chunks: Optional[int] = None
    error: Optional[str] = None


class SearchHit(BaseModel):
    """Representation of a single similarity search hit."""

    file: Optional[str] = None
    page: Optional[int] = None
    score: float
    text: str


class SearchResponse(BaseModel):
    """Response model for search requests."""

    query: Any
    hits: List[SearchHit]


class ChatRequest(BaseModel):
    """Input payload for chat interaction."""

    user_id: str
    message: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)


class Citation(BaseModel):
    """Citation entry returned with chat responses."""

    file: Optional[str] = None
    page: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    """Response payload for chat endpoint."""

    answer: str
    citations: List[Citation]
    conversation_id: str
    citations_insufficient: bool
    latency_ms: float
    max_context_tokens: Optional[int] = Field(
        None, description="Configured maximum size of the model context window"
    )
    max_generation_tokens: Optional[int] = Field(
        None, description="Upper bound for generated tokens per response"
    )


class FileInfo(BaseModel):
    """Metadata returned by the files endpoint."""

    id: str
    filename: str
    tenant: str
    status: str
    uploaded_at: datetime
    size: int
    chunks: Optional[int] = None
    error: Optional[str] = None
    document_id: Optional[str] = None
    document_status: Optional[str] = None
    mime_type: Optional[str] = None


class FilesResponse(BaseModel):
    """Wrapper for listing files."""

    files: List[FileInfo]


class JobInfo(BaseModel):
    """Representation of an asynchronous job entry."""

    id: str
    tenant_id: str
    job_type: str
    status: str
    priority: int
    error: Optional[str] = None
    attempt: int = 0
    resource_id: Optional[str] = None
    payload: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class JobsResponse(BaseModel):
    """Wrapper returned by the admin jobs endpoint."""

    jobs: List[JobInfo]


class DeleteResponse(BaseModel):
    """Confirmation returned after deleting a file."""

    ok: bool
    id: str


__all__ = [
    "Citation",
    "ChatRequest",
    "ChatResponse",
    "DeleteResponse",
    "Document",
    "DocumentCreate",
    "FileInfo",
    "FilesResponse",
    "JobInfo",
    "JobsResponse",
    "IngestRequest",
    "IngestResponse",
    "SearchHit",
    "SearchResponse",
    "UploadResponse",
    "LoraLoadRequest",
    "LoraStatusResponse",
    "LoraUnloadRequest",
]
