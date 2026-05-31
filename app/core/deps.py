"""Common FastAPI dependencies shared across routers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from fastapi import Request
from sqlmodel import Session

from app.core.config import get_settings
from app.llm.manager import LlamaLoraManager
from app.ingest.service import IngestService
from app.services.files import FileStore, IngestQueue


ALLOWED_EXTENSION_WHITELIST = frozenset({"pdf", "docx", "pptx", "xlsx", "txt", "md"})

DEFAULT_ALLOWED_EXTENSIONS = frozenset(sorted(ALLOWED_EXTENSION_WHITELIST))


@dataclass
class UploadLimits:
    """Configuration describing upload restrictions."""

    max_upload_mb: int = 50
    allowed_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_EXTENSIONS))

    max_bytes: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_upload_mb = self._normalise_max_mb(self.max_upload_mb)
        self.max_bytes = self.max_upload_mb * 1024 * 1024
        self.allowed_extensions = self._normalise_extensions(self.allowed_extensions)

    @staticmethod
    def _normalise_max_mb(value: str | int | float) -> int:
        try:
            size = int(float(value))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError("max_upload_mb must be a number") from exc
        if size <= 0:
            raise ValueError("max_upload_mb must be greater than zero")
        return size

    @staticmethod
    def _bytes_to_mb(value: str | int | float) -> int:
        try:
            bytes_value = int(float(value))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
            raise ValueError("max_upload_bytes must be a number") from exc
        if bytes_value <= 0:
            raise ValueError("max_upload_bytes must be greater than zero")
        return max(1, math.ceil(bytes_value / (1024 * 1024)))

    @property
    def max_size(self) -> int:
        """Backwards-compatible alias exposing the limit in bytes."""

        return self.max_bytes

    @staticmethod
    def _normalise_extensions(value: object) -> set[str]:
        if isinstance(value, str):
            candidates = {piece.strip().lower() for piece in value.split(",") if piece.strip()}
        elif isinstance(value, Iterable):
            candidates = {str(item).lower() for item in value}
        else:
            raise ValueError("allowed_extensions must be a string or iterable")

        allowed = candidates & ALLOWED_EXTENSION_WHITELIST
        if not allowed:
            raise ValueError("allowed_extensions must contain at least one supported extension")
        return allowed


def get_data_dir() -> Path:
    """Return the directory where uploaded files are stored."""

    settings = get_settings()
    root = settings.data_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_upload_limits() -> UploadLimits:
    """Provide upload limit configuration from environment variables."""

    settings = get_settings()

    raw_max_mb = os.getenv("MAX_UPLOAD_MB")
    if raw_max_mb:
        max_upload_mb = UploadLimits._normalise_max_mb(raw_max_mb)
    else:
        legacy = os.getenv("UPLOAD_MAX_SIZE")
        if legacy:
            max_upload_mb = UploadLimits._bytes_to_mb(legacy)
        else:
            max_upload_mb = settings.max_upload_mb

    default_extensions = ",".join(sorted(DEFAULT_ALLOWED_EXTENSIONS))
    raw_extensions = os.getenv("UPLOAD_ALLOWED_EXTS")
    extensions = raw_extensions if raw_extensions not in {None, ""} else default_extensions

    return UploadLimits(
        max_upload_mb=max_upload_mb,
        allowed_extensions=UploadLimits._normalise_extensions(extensions),
    )


# FastAPI requires a bare ``Request`` annotation to inject the request object;
# ``Request | None`` breaks route registration. ``= None`` enables direct calls,
# so the implicit-Optional default is suppressed intentionally on these deps.
def get_tenant(request: Request = None) -> str:  # type: ignore[assignment]
    """Resolve tenant identifier from headers (defaulting to ``"default"``)."""

    header_value = (
        request.headers.get("x-tenant") if request and hasattr(request, "headers") else None
    )
    tenant = (header_value or os.getenv("DEFAULT_TENANT", "default") or "default").strip()
    return tenant or "default"


def get_ingest_service(request: Request = None) -> IngestService:  # type: ignore[assignment]
    """Access the shared :class:`~app.ingest.service.IngestService` instance."""

    if request is None:
        raise RuntimeError("Request context is required for ingest service access")
    return request.app.state.ingest_service


def get_ingest_session(request: Request = None) -> Iterator[Session]:  # type: ignore[assignment]
    """Provide a database session tied to the ingest service engine."""

    service = get_ingest_service(request)
    with Session(service.engine) as session:
        yield session


def get_file_store(request: Request = None) -> FileStore:  # type: ignore[assignment]
    """Access the in-memory :class:`~app.services.files.FileStore`."""

    if request is None:
        raise RuntimeError("Request context is required for file store access")
    app = getattr(request, "app", None)
    if app is None:
        raise RuntimeError("File store is not available outside application context")
    store = getattr(getattr(app, "state", None), "file_store", None)
    if store is None:
        raise RuntimeError("File store has not been initialised")
    return store


def get_ingest_queue(request: Request = None) -> IngestQueue:  # type: ignore[assignment]
    """Access the shared :class:`~app.services.files.IngestQueue` instance."""

    if request is None:
        raise RuntimeError("Request context is required for ingest queue access")
    app = getattr(request, "app", None)
    if app is None:
        raise RuntimeError("Ingest queue is not available outside application context")
    queue = getattr(getattr(app, "state", None), "ingest_queue", None)
    if queue is None:
        raise RuntimeError("Ingest queue has not been initialised")
    return queue


def get_lora_manager(request: Request = None) -> LlamaLoraManager:  # type: ignore[assignment]
    """Access the shared LoRA manager instance."""

    app_state = None
    if request is not None:
        app = getattr(request, "app", None)
        if app is None and hasattr(request, "scope"):
            app = request.scope.get("app")
        if app is not None:
            app_state = getattr(app, "state", None)
    if app_state is None:
        try:  # pragma: no cover - fallback path for stubbed requests
            from app.main import app as main_app  # type: ignore
        except Exception:  # pragma: no cover - defensive
            main_app = None
        if main_app is not None:
            app_state = getattr(main_app, "state", None)
    if app_state is None:
        raise RuntimeError("LoRA manager is not configured")
    manager = getattr(app_state, "lora_manager", None)
    if isinstance(manager, LlamaLoraManager):
        return manager

    if manager is None or not all(
        hasattr(manager, attr) for attr in ("load_adapter", "unload_adapter")
    ):
        raise RuntimeError("LoRA manager is not configured")

    return manager  # type: ignore[return-value]


__all__ = [
    "DEFAULT_ALLOWED_EXTENSIONS",
    "UploadLimits",
    "get_data_dir",
    "get_ingest_service",
    "get_ingest_session",
    "get_file_store",
    "get_ingest_queue",
    "get_lora_manager",
    "get_tenant",
    "get_upload_limits",
]
