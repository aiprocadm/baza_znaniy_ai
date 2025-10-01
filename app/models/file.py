"""SQLModel models for storing ingestion metadata."""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Optional

from sqlalchemy import Column, Text, UniqueConstraint
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Field, SQLModel, Session, create_engine

# Ensure metadata is aware of tenant/user tables when engines are initialised
from app.models.tenant import TenantRecord  # noqa: F401
from app.models.user import UserRecord  # noqa: F401


class FileStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileRecord(SQLModel, table=True):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("tenant_id", "sha256", name="uq_files_tenant_sha"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True)
    sha256: str = Field(index=True)
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
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)



class TenantRecord(SQLModel, table=True):
    __tablename__ = "tenants"
    __table_args__ = (
        UniqueConstraint("name", name="uq_tenants_name"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class UserRecord(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    email: str = Field(index=True)
    full_name: Optional[str] = Field(default=None)
    role: Optional[str] = Field(default="member")
    hashed_password: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class DocumentRecord(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_documents_tenant_slug"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: int = Field(foreign_key="tenants.id", index=True)
    file_id: Optional[int] = Field(default=None, foreign_key="files.id", index=True)
    slug: Optional[str] = Field(default=None, index=True)
    title: Optional[str] = Field(default=None)
    content: str = Field(default="", sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class JobRecord(SQLModel, table=True):
    __tablename__ = "jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: Optional[int] = Field(default=None, foreign_key="tenants.id", index=True)
    job_type: str = Field(index=True)
    status: str = Field(default="pending", index=True)
    payload: Optional[str] = Field(default=None)
    scheduled_at: Optional[datetime] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    finished_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class SettingRecord(SQLModel, table=True):
    __tablename__ = "settings"
    __table_args__ = (
        UniqueConstraint("tenant_id", "key", name="uq_settings_tenant_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: Optional[int] = Field(default=None, foreign_key="tenants.id", index=True)
    key: str = Field(index=True)
    value: str = Field(sa_column=Column(Text, nullable=False))
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)



def _connect_args(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


@lru_cache(maxsize=1)
def get_engine(url: Optional[str] = None, *, create_schema: bool = True) -> Engine:
    db_url = url or os.getenv("DB_URL", "sqlite+aiosqlite:///./var/data/kb.sqlite")
    dialect = make_url(db_url)

    if dialect.drivername.endswith("+aiosqlite"):
        async_engine = create_async_engine(db_url, echo=False)
        sync_engine = async_engine.sync_engine
        if create_schema:
            SQLModel.metadata.create_all(sync_engine)
        return sync_engine

    engine = create_engine(db_url, echo=False, connect_args=_connect_args(db_url))
    if create_schema:
        SQLModel.metadata.create_all(engine)
    return engine


def get_session(url: Optional[str] = None) -> Session:
    """Create a new session bound to the configured engine."""

    engine = get_engine(url)
    return Session(engine)


__all__ = [
    "DocumentRecord",
    "ChunkRecord",
    "FileRecord",
    "FileStatus",
    "JobRecord",
    "PageRecord",
    "SettingRecord",
    "TenantRecord",
    "UserRecord",
    "get_engine",
    "get_session",
]
