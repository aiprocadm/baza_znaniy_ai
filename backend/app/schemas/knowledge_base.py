from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ServiceHealth(BaseModel):
    name: str
    status: Literal["healthy", "degraded", "offline"]
    latency_ms: int = Field(ge=0)
    last_error: str | None = None


class SystemStats(BaseModel):
    documents: int = Field(ge=0)
    ingestions: int = Field(ge=0)
    errors: int = Field(ge=0)


class SystemStatusResponse(BaseModel):
    services: list[ServiceHealth]
    stats: SystemStats


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=10, ge=1, le=50)
    tags: list[str] | None = None
    owner: str | None = None


class SearchResultItem(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    source: str
    updated_at: datetime


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int


class ActivityItem(BaseModel):
    id: str
    type: Literal["upload", "ingest", "chat", "search"]
    title: str
    description: str
    created_at: datetime


class FileMeta(BaseModel):
    id: str
    name: str
    size: int = Field(ge=0)
    mime_type: str
    status: Literal["processing", "indexed", "error"]
    created_at: datetime


class UserPayload(BaseModel):
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    roles: list[Literal["user", "admin"]] = Field(default_factory=lambda: ["user"])


class UserUpdatePayload(BaseModel):
    name: str | None = None
    email: str | None = None
    roles: list[Literal["user", "admin"]] | None = None
    status: Literal["active", "invited", "blocked"] | None = None


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    roles: list[Literal["user", "admin"]]
    status: Literal["active", "invited", "blocked"]


class ApiKey(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None = None


class SessionResponse(BaseModel):
    user_id: str
    email: str
    name: str
    roles: list[Literal["user", "admin"]]
    token_expires_at: datetime


class LoginPayload(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class RefreshResponse(BaseModel):
    token: str


class SystemSettings(BaseModel):
    qdrant_url: str
    llm_model: str
    ingestion_parallelism: int = Field(ge=1, le=32)
    allow_guest_access: bool
