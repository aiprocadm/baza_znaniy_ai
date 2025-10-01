"""Common FastAPI dependencies shared across routers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from fastapi import Request

        codex/refactor-upload-and-ingest-apis-to-use-ingestservice
from sqlmodel import Session

from app.ingest.service import IngestService

from app.core.config import get_settings
from app.services.files import FileStore, IngestQueue
        main


DEFAULT_ALLOWED_EXTENSIONS = frozenset({"pdf", "docx", "pptx", "xlsx", "txt", "md"})


@dataclass
class UploadLimits:
    """Configuration describing upload restrictions."""

    max_upload_mb: int = 40
    allowed_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_EXTENSIONS))

    max_bytes: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_upload_mb = self._normalise_max_mb(self.max_upload_mb)
        self.max_bytes = self.max_upload_mb * 1024 * 1024
        self.allowed_extensions = self._normalise_extensions(self.allowed_extensions)

    @staticmethod
    def _normalise_max_mb(value: object) -> int:
        try:
            size = int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError("max_upload_mb must be an integer") from exc
        if size <= 0:
            raise ValueError("max_upload_mb must be greater than zero")
        return size

    @property
    def max_size(self) -> int:
        """Backwards-compatible alias exposing the limit in bytes."""

        return self.max_bytes

    @staticmethod
    def _normalise_extensions(value: object) -> set[str]:
        if isinstance(value, str):
            return {piece.strip().lower() for piece in value.split(",") if piece.strip()}
        if isinstance(value, Iterable):
            return {str(item).lower() for item in value}
        raise ValueError("allowed_extensions must be a string or iterable")


def get_data_dir() -> Path:
    """Return the directory where uploaded files are stored."""

    settings = get_settings()
    root = settings.data_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_upload_limits() -> UploadLimits:
    """Provide upload limit configuration from environment variables."""

        codex/expand-env.example-and-update-configuration
    settings = get_settings()

    def _mb_to_bytes(value: float | int) -> int:
        return int(float(value) * 1024 * 1024)

    raw_max_mb = os.getenv("MAX_UPLOAD_MB")
    if raw_max_mb not in {None, ""}:
        try:
            max_size = _mb_to_bytes(float(raw_max_mb))
        except ValueError as exc:  # pragma: no cover - defensive conversion
            raise ValueError("MAX_UPLOAD_MB must be a number") from exc
    else:
        legacy = os.getenv("UPLOAD_MAX_SIZE")
        if legacy not in {None, ""}:
            max_size = UploadLimits._normalise_max_size(legacy)
        else:
            max_size = _mb_to_bytes(settings.max_upload_mb)

    extensions = os.getenv("UPLOAD_ALLOWED_EXTS", "pdf,docx,txt")
    return UploadLimits(max_size=max_size, allowed_extensions=extensions)

    defaults = UploadLimits()
    max_upload_mb = os.getenv("MAX_UPLOAD_MB", defaults.max_upload_mb)
    return UploadLimits(max_upload_mb=max_upload_mb)
        main


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
    "DEFAULT_ALLOWED_EXTENSIONS",
    "UploadLimits",
    "get_data_dir",
    "get_ingest_service",
    "get_ingest_session",
    "get_tenant",
    "get_upload_limits",
]
