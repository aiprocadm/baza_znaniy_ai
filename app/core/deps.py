"""Common FastAPI dependencies shared across routers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from fastapi import Request

from sqlmodel import Session

from app.ingest.service import IngestService


@dataclass
class UploadLimits:
    """Configuration describing upload restrictions."""

    max_size: int = 10 * 1024 * 1024
    allowed_extensions: set[str] = field(default_factory=lambda: {"pdf", "docx", "txt"})

    def __post_init__(self) -> None:
        self.max_size = self._normalise_max_size(self.max_size)
        self.allowed_extensions = self._normalise_extensions(self.allowed_extensions)

    @staticmethod
    def _normalise_max_size(value: object) -> int:
        try:
            size = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError("max_size must be an integer") from exc
        if size <= 0:
            raise ValueError("max_size must be greater than zero")
        return size

    @staticmethod
    def _normalise_extensions(value: object) -> set[str]:
        if isinstance(value, str):
            return {piece.strip().lower() for piece in value.split(",") if piece.strip()}
        if isinstance(value, Iterable):
            return {str(item).lower() for item in value}
        raise ValueError("allowed_extensions must be a string or iterable")


def get_data_dir() -> Path:
    """Return the directory where uploaded files are stored."""

    root = Path(os.getenv("DATA_DIR", "/opt/knowlab/data/files")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_upload_limits() -> UploadLimits:
    """Provide upload limit configuration from environment variables."""

    max_size = os.getenv("UPLOAD_MAX_SIZE", UploadLimits().max_size)
    extensions = os.getenv("UPLOAD_ALLOWED_EXTS", "pdf,docx,txt")
    return UploadLimits(max_size=max_size, allowed_extensions=extensions)


def get_tenant(request: Request = None) -> str:
    """Resolve tenant identifier from headers (defaulting to ``"default"``)."""

    header_value = request.headers.get("x-tenant") if request and hasattr(request, "headers") else None
    tenant = (header_value or os.getenv("DEFAULT_TENANT", "default")).strip()
    return tenant or "default"


def get_ingest_service(request: Request = None) -> IngestService:
    """Access the shared :class:`~app.ingest.service.IngestService` instance."""

    if request is None:
        raise RuntimeError("Request context is required for ingest service access")
    return request.app.state.ingest_service


def get_ingest_session(request: Request = None) -> Iterator[Session]:
    """Provide a database session tied to the ingest service engine."""

    service = get_ingest_service(request)
    with Session(service.engine) as session:
        yield session


__all__ = [
    "UploadLimits",
    "get_data_dir",
    "get_ingest_service",
    "get_ingest_session",
    "get_tenant",
    "get_upload_limits",
]
