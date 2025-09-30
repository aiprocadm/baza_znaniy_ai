"""Common FastAPI dependencies shared across routers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from fastapi import Request

from app.services.files import FileStore, IngestQueue


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

    root = Path(os.getenv("DATA_DIR", "/opt/knowlab/data/files")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_upload_limits() -> UploadLimits:
    """Provide upload limit configuration from environment variables."""

    defaults = UploadLimits()
    max_upload_mb = os.getenv("MAX_UPLOAD_MB", defaults.max_upload_mb)
    return UploadLimits(max_upload_mb=max_upload_mb)


def get_tenant(request: Request = None) -> str:
    """Resolve tenant identifier from headers (defaulting to ``"default"``)."""

    header_value = request.headers.get("x-tenant") if request and hasattr(request, "headers") else None
    tenant = (header_value or os.getenv("DEFAULT_TENANT", "default")).strip()
    return tenant or "default"


def get_file_store(request: Request = None) -> FileStore:
    """Access the shared :class:`~app.services.files.FileStore` instance."""

    if request is None:
        raise RuntimeError("Request context is required for file store access")
    return request.app.state.file_store


def get_ingest_queue(request: Request = None) -> IngestQueue:
    """Access the shared :class:`~app.services.files.IngestQueue` instance."""

    if request is None:
        raise RuntimeError("Request context is required for ingest queue access")
    return request.app.state.ingest_queue


__all__ = [
    "DEFAULT_ALLOWED_EXTENSIONS",
    "UploadLimits",
    "get_data_dir",
    "get_file_store",
    "get_ingest_queue",
    "get_tenant",
    "get_upload_limits",
]
