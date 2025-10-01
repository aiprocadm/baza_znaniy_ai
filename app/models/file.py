"""SQLModel models for storing ingestion metadata."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Optional

from sqlalchemy import Column, JSON, UniqueConstraint
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, Relationship, SQLModel, Session, create_engine


class FileStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRecord(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("sha256", name="uq_documents_sha"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    sha256: str = Field(index=True)
    mime_type: str = Field(default="application/octet-stream")
    status: str = Field(default=DocumentStatus.QUEUED, index=True)
    error: Optional[str] = Field(default=None)
    chunks: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

class FileRecord(SQLModel, table=True):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("tenant_id", "sha256", name="uq_files_tenant_sha"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True)
    sha256: str = Field(index=True)
    document_id: Optional[int] = Field(
        default=None, foreign_key="documents.id", index=True
    )
    path: str
    filename: str
    size: int = Field(default=0, ge=0)
    status: str = Field(default=FileStatus.QUEUED, index=True)
    retries: int = Field(default=0)
    error: Optional[str] = Field(default=None)
    chunks: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

class PageRecord(SQLModel, table=True):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("file_id", "number", name="uq_pages_file_number"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="files.id", index=True)
    number: int = Field(index=True)
    sha256: str = Field(index=True)
    text: str
    tokens: int = Field(default=0, ge=0)
    meta: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)



class ChunkRecord(SQLModel, table=True):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("page_id", "index", name="uq_chunks_page_index"),
        UniqueConstraint("page_id", "sha256", name="uq_chunks_page_sha"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    page_id: int = Field(foreign_key="pages.id", index=True)
    index: int = Field(index=True)
    sha256: str = Field(index=True)
    text: str
    batch: Optional[int] = Field(default=None, index=True)
    tokens: int = Field(default=0, ge=0)
    meta: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)



def _connect_args(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


@lru_cache(maxsize=1)
def get_engine(url: Optional[str] = None) -> Engine:
    db_url = url or os.getenv("DB_URL", "sqlite+aiosqlite:///./var/data/kb.sqlite")
    dialect = make_url(db_url)

    # Ensure additional SQLModel definitions are imported before metadata creation
    __import__("app.models.entities")

    if dialect.drivername.endswith("+aiosqlite"):
        async_engine = create_async_engine(db_url, echo=False)
        sync_engine = async_engine.sync_engine
        SQLModel.metadata.create_all(sync_engine)
        return sync_engine

    engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url))
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(url: Optional[str] = None) -> Session:
    """Create a new session bound to the configured engine."""

    engine = get_engine(url)
    return Session(engine)


__all__ = [
    "ChunkRecord",
    "DocumentRecord",
    "DocumentStatus",
    "FileRecord",
    "FileStatus",
    "PageRecord",
    "get_engine",
    "get_session",
]
