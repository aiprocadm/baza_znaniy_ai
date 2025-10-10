"""Pydantic models shared across the service and tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.datetime_utils import utc_now
from .lora import LoraAdapterInfo, LoraAdapterName, LoraStatusResponse

import importlib

_LAZY_MODELS = {
    "UserCreate": ("app.models.user", "UserCreate"),
    "UserRecord": ("app.models.user", "UserRecord"),
    "UserResponse": ("app.models.user", "UserResponse"),
    "UserRole": ("app.models.user", "UserRole"),
    "UserUpdate": ("app.models.user", "UserUpdate"),
    "TenantCreate": ("app.models.tenant", "TenantCreate"),
    "TenantRecord": ("app.models.tenant", "TenantRecord"),
    "TenantResponse": ("app.models.tenant", "TenantResponse"),
    "TenantUpdate": ("app.models.tenant", "TenantUpdate"),
}
class Document(BaseModel):
    """Representation of a stored document chunk used across the service."""

    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document body")
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


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


class FileSummaryResponse(BaseModel):
    """Aggregated statistics returned by the files summary endpoint."""

    total_files: int = Field(..., ge=0)
    total_size_bytes: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    status_counts: Dict[str, int] = Field(default_factory=dict)
    oldest_upload: Optional[datetime] = None
    newest_upload: Optional[datetime] = None
    average_size_bytes: Optional[float] = Field(default=None, ge=0)


class IngestFailureInfo(BaseModel):
    """Representation of a failed ingestion attempt surfaced via the API."""

    file_id: str
    filename: str
    status: str
    error: Optional[str] = None
    uploaded_at: datetime


class IngestQueueMetricsResponse(BaseModel):
    """Real-time ingest queue metrics exposed for the Operations Console."""

    total_files: int = Field(..., ge=0)
    queue_depth: int = Field(..., ge=0)
    status_counts: Dict[str, int] = Field(default_factory=dict)
    oldest_pending_age_seconds: Optional[float] = Field(default=None, ge=0)
    average_pending_age_seconds: Optional[float] = Field(default=None, ge=0)
    recent_failures: List[IngestFailureInfo] = Field(default_factory=list)
    last_activity_at: Optional[datetime] = None


class JobInfo(BaseModel):
    """Representation of an asynchronous job entry."""

    id: str
    tenant_slug: Optional[str]
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


def __getattr__(name: str) -> Any:  # pragma: no cover - import side effect helper
    """Lazily import models that define additional SQLModel tables."""

    module_info = _LAZY_MODELS.get(name)
    if module_info is None:
        raise AttributeError(f"module 'app.models' has no attribute {name!r}")
    module = importlib.import_module(module_info[0])
    value = getattr(module, module_info[1])
    globals()[name] = value
    return value


__all__ = [
    "Citation",
    "ChatRequest",
    "ChatResponse",
    "DeleteResponse",
    "Document",
    "DocumentCreate",
    "FileInfo",
    "FilesResponse",
    "FileSummaryResponse",
    "IngestFailureInfo",
    "IngestQueueMetricsResponse",
    "JobInfo",
    "JobsResponse",
    "IngestRequest",
    "IngestResponse",
    "SearchHit",
    "SearchResponse",
    "UploadResponse",
    "LoraAdapterInfo",
    "LoraAdapterName",
    "LoraStatusResponse",
    "TenantCreate",
    "TenantRecord",
    "TenantResponse",
    "TenantUpdate",
    "UserCreate",
    "UserRecord",
    "UserResponse",
    "UserRole",
    "UserUpdate",
]
