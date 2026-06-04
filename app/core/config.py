"""Application configuration models and helpers."""

from __future__ import annotations

import math
import os

from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Python 3.10 compatibility shim
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - fallback for older interpreters
    import importlib_metadata  # type: ignore[import-not-found]

from decimal import Decimal

from pydantic import AliasChoices, Field

from pydantic import BaseModel, computed_field, field_validator, model_validator


try:  # pragma: no cover - ``pydantic-settings`` is optional
    from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - minimal shim for tests

    PydanticBaseSettings = None  # type: ignore[assignment]


def _default_auth_disabled() -> bool:
    """Return whether authentication should be bypassed by default."""

    raw_value = (
        os.getenv("AUTH_DISABLED_FOR_TESTS")
        or os.getenv("AUTH_DISABLED")
        or os.getenv("DISABLE_AUTH")
        or os.getenv("AUTH_DISABLE")
        or os.getenv("KB_DISABLE_AUTH")
    )
    if raw_value is None:
        return False

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _flatten_aliases(source: object) -> list[str]:
    """Return a flattened list of alias names from arbitrary structures."""

    if source is None or source is Ellipsis:  # pragma: no cover - defensive guard
        return []
    if isinstance(source, AliasChoices):
        items: list[str] = []
        iterable: Iterable[object]
        if isinstance(source, Iterable):
            iterable = source  # older Pydantic — AliasChoices was Iterable
        else:  # Pydantic >=2.9 — AliasChoices exposes only ``.choices``
            iterable = getattr(source, "choices", ())  # type: ignore[attr-defined]
        for choice in iterable:
            items.extend(_flatten_aliases(choice))
        return items
    if isinstance(source, bytes):
        try:
            return [source.decode("utf-8")]
        except Exception:  # pragma: no cover - defensive guard
            return [str(source)]
    if isinstance(source, str):
        return [source]
    if isinstance(source, Iterable):
        items: list[str] = []
        for item in source:
            items.extend(_flatten_aliases(item))
        return items
    return [str(source)]


def _candidate_env_names(field_name: str, field: object) -> list[str]:
    """Return environment variable names to probe for a field."""

    candidates: list[str] = []
    seen: set[str] = set()

    def _register(raw: object) -> None:
        for alias_name in _flatten_aliases(raw):
            name = alias_name.strip()
            if not name:
                continue
            if name not in seen:
                candidates.append(name)
                seen.add(name)
            upper_name = name.upper()
            if upper_name not in seen:
                candidates.append(upper_name)
                seen.add(upper_name)

    _register(field_name)
    _register(field_name.upper())
    _register(getattr(field, "alias", None))

    field_info = getattr(field, "field_info", None)
    if field_info is not None:  # pragma: no branch - simple attribute lookups
        _register(getattr(field_info, "alias", None))
        _register(getattr(field_info, "validation_alias", None))
        _register(getattr(field_info, "serialization_alias", None))

    _register(getattr(field, "validation_alias", None))
    _register(getattr(field, "serialization_alias", None))

    return candidates


def _environment_overrides(
    model_cls: type[BaseModel],
    *,
    skip: Iterable[str] | None = None,
) -> tuple[dict[str, object], dict[str, list[str]]]:
    """Collect overrides and the env variable names used for each field."""

    excluded = set(skip or ())
    overrides: dict[str, object] = {}
    consumed: dict[str, list[str]] = {}
    model_fields = getattr(model_cls, "model_fields", None)

    if isinstance(model_fields, dict):
        for name, field in model_fields.items():
            if name in excluded:
                continue
            candidates = _candidate_env_names(name, field)
            for env_name in candidates:
                env_value = os.getenv(env_name)
                if env_value is not None:
                    overrides[name] = env_value
                    consumed[name] = candidates
                    break
    else:  # pragma: no cover - fallback for Pydantic v1 style models
        fields_map = getattr(model_cls, "__fields__", None)
        if isinstance(fields_map, dict):
            iterable: Iterable[tuple[str, object]] = fields_map.items()
        else:
            annotations = getattr(model_cls, "__annotations__", {})
            iterable = ((name, getattr(model_cls, name, None)) for name in annotations)
        for name, field in iterable:
            if name in excluded:
                continue

            candidates = _candidate_env_names(name, field)
            if not candidates:
                candidates = [name, name.upper()]

            field_info = getattr(model_cls, name, None)
            candidates = _candidate_env_names(name, field_info)

            for env_name in candidates:
                env_value = os.getenv(env_name)
                if env_value is not None:
                    overrides[name] = env_value
                    consumed[name] = candidates
                    break

    return overrides, consumed


if PydanticBaseSettings is None:

    import os

    def _flatten_aliases(source: object) -> list[str]:
        """Return a flattened list of alias names from arbitrary structures."""

        if source is None or source is Ellipsis:  # pragma: no cover - defensive guard
            return []
        if isinstance(source, AliasChoices):
            items: list[str] = []
            iterable: Iterable[object]
            if isinstance(source, Iterable):
                iterable = source
            else:  # pragma: no cover - fallback for unexpected shims
                iterable = getattr(source, "choices", ())  # type: ignore[attr-defined]
            for choice in iterable:
                items.extend(_flatten_aliases(choice))
            return items
        if isinstance(source, bytes):
            try:
                return [source.decode("utf-8")]
            except Exception:  # pragma: no cover - defensive guard
                return [str(source)]
        if isinstance(source, str):
            return [source]
        if isinstance(source, Iterable):
            items: list[str] = []
            for item in source:
                items.extend(_flatten_aliases(item))
            return items
        return [str(source)]

    def _candidate_env_names(field_name: str, field: object) -> list[str]:
        """Return environment variable names to probe for a field."""

        candidates: list[str] = []

        def _add(name: object) -> None:
            if name is None or name is Ellipsis or name == "":  # pragma: no cover - guard
                return
            text = str(name)
            if text not in candidates:
                candidates.append(text)

        def _add_all(value: object) -> None:
            for alias_name in _flatten_aliases(value):
                _add(alias_name)

        _add(field_name.upper())

        alias_value = getattr(field, "alias", None)
        if alias_value is None:
            alias_value = getattr(getattr(field, "metadata", {}), "get", lambda *_: None)("alias")
        _add_all(alias_value)

        validation_alias = getattr(field, "validation_alias", None)
        if validation_alias is None and hasattr(field, "metadata"):
            validation_alias = getattr(field.metadata, "get", lambda *_: None)("validation_alias")
        _add_all(validation_alias)

        return candidates

    class SettingsConfigDict(dict):  # type: ignore[override]
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    class BaseSettings(BaseModel):  # type: ignore[misc]
        model_config = SettingsConfigDict()

        def __init__(self, **data: object) -> None:

            overrides, consumed = _environment_overrides(self.__class__, skip=data.keys())
            merged = {**overrides, **data}
            restored: list[tuple[str, str | None]] = []
            seen: set[str] = set()
            for env_names in consumed.values():
                for env_name in env_names:
                    if env_name in seen:
                        continue
                    seen.add(env_name)
                    restored.append((env_name, os.environ.get(env_name)))
                    os.environ.pop(env_name, None)
            config_obj = getattr(self.__class__, "model_config", None)
            previous_env_file = getattr(config_obj, "env_file", None)
            if config_obj is not None:
                try:
                    config_obj["env_file"] = ()  # type: ignore[index]
                except Exception:
                    pass
                try:
                    setattr(config_obj, "env_file", ())
                except Exception:
                    pass
            try:
                super().__init__(**merged)
                if overrides:
                    payload = self.model_dump(mode="python")
                    payload.update(overrides)
                    updated = self.__class__.model_validate(payload)
                    self.__dict__.update(updated.__dict__)
                    self.__pydantic_fields_set__ = updated.__pydantic_fields_set__
            finally:
                if config_obj is not None:
                    try:
                        config_obj["env_file"] = previous_env_file  # type: ignore[index]
                    except Exception:
                        pass
                    try:
                        setattr(config_obj, "env_file", previous_env_file)
                    except Exception:
                        pass
                for env_name, original in reversed(restored):
                    if original is None:
                        os.environ.pop(env_name, None)
                    else:
                        os.environ[env_name] = original

else:

    class BaseSettings(PydanticBaseSettings):  # type: ignore[misc]
        """Augmented settings class with eager environment alias resolution."""

        model_config = SettingsConfigDict()

        def __init__(self, **data: object) -> None:
            env_overrides, consumed = _environment_overrides(self.__class__, skip=data.keys())
            merged = {**env_overrides, **data}
            restored: list[tuple[str, str | None]] = []
            seen: set[str] = set()
            for env_names in consumed.values():
                for env_name in env_names:
                    if env_name in seen:
                        continue
                    seen.add(env_name)
                    restored.append((env_name, os.environ.get(env_name)))
                    os.environ.pop(env_name, None)
            config_obj = getattr(self.__class__, "model_config", None)
            previous_env_file = getattr(config_obj, "env_file", None)
            if config_obj is not None:
                try:
                    config_obj["env_file"] = ()  # type: ignore[index]
                except Exception:
                    pass
                try:
                    setattr(config_obj, "env_file", ())
                except Exception:
                    pass
            try:
                super().__init__(**merged)
                if env_overrides:
                    payload = self.model_dump(mode="python")
                    payload.update(env_overrides)
                    updated = self.__class__.model_validate(payload)
                    self.__dict__.update(updated.__dict__)
                    self.__pydantic_fields_set__ = updated.__pydantic_fields_set__
            finally:
                if config_obj is not None:
                    try:
                        config_obj["env_file"] = previous_env_file  # type: ignore[index]
                    except Exception:
                        pass
                    try:
                        setattr(config_obj, "env_file", previous_env_file)
                    except Exception:
                        pass
                for env_name, original in reversed(restored):
                    if original is None:
                        os.environ.pop(env_name, None)
                    else:
                        os.environ[env_name] = original

            values: dict[str, object] = {}
            model_fields = getattr(self.__class__, "model_fields", None)
            if isinstance(model_fields, dict):
                for name, field in model_fields.items():
                    for env_name in _candidate_env_names(name, field):
                        env_value = os.getenv(env_name)
                        if env_value is not None:
                            values[name] = env_value
                            break
            else:  # pragma: no cover - fallback for extremely small shims

                fields_map = getattr(self.__class__, "__fields__", None)
                if isinstance(fields_map, dict):
                    iterable: Iterable[tuple[str, object]] = fields_map.items()
                else:
                    annotations = getattr(self.__class__, "__annotations__", {})
                    iterable = ((name, getattr(self.__class__, name, None)) for name in annotations)
                for name, field_info in iterable:
                    for env_name in _candidate_env_names(name, field_info):
                        env_value = os.getenv(env_name)
                        if env_value is not None:
                            values[name] = env_value
                            break
            values.update(data)
            super().__init__(**values)


def _default_app_version() -> str:
    """Return the best effort application version string."""

    try:
        return importlib_metadata.version("kb-ai")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - editable installs in tests
        return "0.1.0"
    except Exception:  # pragma: no cover - defensive guard
        return "0.1.0"


class Settings(BaseSettings):
    """Configuration loaded from environment variables and ``.env`` files."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Core application ---------------------------------------------------
    app_env: str = Field(
        default="development",
        validation_alias=AliasChoices("APP_ENV", "ENV", "ENVIRONMENT"),
    )
    app_version: str = Field(
        default_factory=_default_app_version,
        validation_alias=AliasChoices("APP_VERSION"),
    )
    app_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("APP_HOST", "HOST"),
    )
    app_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("APP_PORT", "PORT"),
    )
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    rate_limit: str | None = Field(default=None, validation_alias=AliasChoices("RATE_LIMIT"))
    rate_limit_backend: str = Field(
        default="memory",
        validation_alias=AliasChoices("RATE_LIMIT_BACKEND"),
    )
    rate_burst: int = Field(default=0, validation_alias=AliasChoices("RATE_BURST"))
    auth_disabled: bool = Field(
        default_factory=_default_auth_disabled,
        validation_alias=AliasChoices(
            "AUTH_DISABLED",
            "DISABLE_AUTH",
            "AUTH_DISABLE",
            "KB_DISABLE_AUTH",
            "AUTH_DISABLED_FOR_TESTS",
        ),
    )

    auth_provider: str = Field(
        default="local-jwt",
        validation_alias=AliasChoices("AUTH_PROVIDER"),
    )
    keycloak_server_url: str | None = Field(
        default=None, validation_alias=AliasChoices("KEYCLOAK_SERVER_URL")
    )
    keycloak_realm: str | None = Field(
        default=None, validation_alias=AliasChoices("KEYCLOAK_REALM")
    )
    keycloak_client_id: str | None = Field(
        default=None, validation_alias=AliasChoices("KEYCLOAK_CLIENT_ID")
    )
    supabase_url: str | None = Field(default=None, validation_alias=AliasChoices("SUPABASE_URL"))
    supabase_jwt_secret: str | None = Field(
        default=None, validation_alias=AliasChoices("SUPABASE_JWT_SECRET")
    )

    # Core paths ---------------------------------------------------------
    data_dir: Path = Field(
        default=Path("./var/data"),
        validation_alias=AliasChoices("DATA_DIR", "FILES_ROOT"),
    )
    files_subdir: str = Field(default="files", validation_alias=AliasChoices("FILES_SUBDIR"))
    db_url: str = Field(
        default="sqlite+aiosqlite:///./var/data/kb.sqlite",
        validation_alias=AliasChoices("DB_URL", "INGEST_DB_URL"),
    )
    postgres_dsn: str | None = Field(default=None, validation_alias=AliasChoices("POSTGRES_DSN"))
    max_upload_mb: int = Field(
        default=40,
        validation_alias=AliasChoices("MAX_UPLOAD_MB", "UPLOAD_MAX_MB"),
    )
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        validation_alias=AliasChoices(
            "CORS_ALLOW_ORIGINS",
            "CORS_ALLOWED_ORIGINS",
            "ALLOWED_ORIGINS",
        ),
    )

    # Chat storage -------------------------------------------------------
    chat_db_backend: str = Field(default="sqlite", validation_alias=AliasChoices("CHAT_DB_BACKEND"))
    chat_db_path: Path | None = Field(default=None, validation_alias=AliasChoices("CHAT_DB_PATH"))
    chat_db_dsn: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CHAT_DB_DSN", "CHAT_DB_URL", "DATABASE_URL"),
    )
    chat_db_schema: str | None = Field(
        default=None, validation_alias=AliasChoices("CHAT_DB_SCHEMA")
    )
    chat_history_limit: int = Field(default=12, validation_alias=AliasChoices("CHAT_HISTORY_LIMIT"))
    chat_summary_trigger: int = Field(
        default=10,
        validation_alias=AliasChoices("CHAT_SUMMARY_TRIGGER", "MEMORY_SUMMARY_TRIGGER"),
    )
    chat_min_citations: int = Field(default=3, validation_alias=AliasChoices("CHAT_MIN_CITATIONS"))
    chat_max_citations: int = Field(default=5, validation_alias=AliasChoices("CHAT_MAX_CITATIONS"))

    # Retrieval ----------------------------------------------------------
    retrieve_topk: int = Field(default=10, validation_alias=AliasChoices("RETRIEVE_TOPK"))
    rerank_enabled: bool = Field(default=True, validation_alias=AliasChoices("RERANK_ENABLED"))
    rerank_topk: int | None = Field(
        default=50,
        validation_alias=AliasChoices("RERANK_TOPK", "RERANK_TOP_K"),
    )
    rag_tokenizer_name: str = Field(
        default="cl100k_base", validation_alias=AliasChoices("RAG_TOKENIZER_NAME")
    )
    rag_chunk: int = Field(default=900, validation_alias=AliasChoices("RAG_CHUNK"))
    rag_overlap: int = Field(default=140, validation_alias=AliasChoices("RAG_OVERLAP"))
    ingest_max_retries: int = Field(default=3, validation_alias=AliasChoices("INGEST_MAX_RETRIES"))
    ingest_backoff_seconds: float = Field(
        default=1.0,
        validation_alias=AliasChoices("INGEST_BACKOFF_SECONDS", "INGEST_BACKOFF_BASE"),
    )
    ingest_queue_size: int = Field(
        default=64,
        validation_alias=AliasChoices("INGEST_QUEUE_SIZE", "INGEST_MAX_QUEUE"),
    )
    ingest_use_local_queue: bool = Field(
        default=True,
        validation_alias=AliasChoices("INGEST_USE_LOCAL_QUEUE"),
    )
    ingest_autostart_worker: bool = Field(
        default=True,
        validation_alias=AliasChoices("INGEST_AUTOSTART_WORKER", "INGEST_AUTO_START_WORKER"),
    )
    ingest_worker_interval_seconds: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "INGEST_WORKER_INTERVAL_SECONDS",
            "INGEST_POLL_INTERVAL",
            "INGEST_SCHEDULER_INTERVAL",
        ),
    )
    ingest_processing_timeout_seconds: float = Field(
        default=900.0,
        validation_alias=AliasChoices(
            "INGEST_PROCESSING_TIMEOUT_SECONDS",
            "INGEST_STUCK_PROCESSING_TIMEOUT_SECONDS",
        ),
    )
    ingest_maintenance_cron: str = Field(
        default="0 * * * *",
        validation_alias=AliasChoices(
            "INGEST_MAINTENANCE_CRON",
            "INGEST_MAINTENANCE_SCHEDULE",
        ),
    )
    ingest_job_retention_days: int = Field(
        default=7,
        validation_alias=AliasChoices(
            "INGEST_JOB_RETENTION_DAYS",
            "INGEST_RETENTION_DAYS",
        ),
    )

    @field_validator("ingest_queue_size", mode="before")
    @classmethod
    def _validate_ingest_queue_size(cls, value: object) -> int:
        if isinstance(value, (int, float)):
            if math.isinf(value):
                return 0
            if isinstance(value, float) and math.isnan(value):
                raise ValueError("ingest_queue_size cannot be NaN")
        if isinstance(value, Decimal):
            if value.is_infinite():
                return 0
            if not value.is_finite():
                raise ValueError("ingest_queue_size must be finite")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "unbounded",
                "unlimited",
                "infinite",
                "inf",
                "none",
                "no-limit",
                "nolimit",
            }:
                return 0
        candidate = int(value)
        if candidate < 0:
            raise ValueError("ingest_queue_size must be zero or positive")
        return candidate

    @field_validator("ingest_worker_interval_seconds", mode="before")
    @classmethod
    def _validate_ingest_worker_interval(cls, value: object) -> float:
        candidate = float(value)
        if candidate <= 0:
            raise ValueError("ingest_worker_interval_seconds must be positive")
        return candidate

    @field_validator("ingest_processing_timeout_seconds", mode="before")
    @classmethod
    def _validate_ingest_processing_timeout(cls, value: object) -> float:
        candidate = float(value)
        if candidate <= 0:
            raise ValueError("ingest_processing_timeout_seconds must be positive")
        return candidate

    @field_validator("ingest_job_retention_days", mode="before")
    @classmethod
    def _validate_ingest_retention(cls, value: object) -> int:
        candidate = int(value)
        if candidate <= 0:
            raise ValueError("ingest_job_retention_days must be positive")
        return candidate

    # OCR configuration --------------------------------------------------
    ocr_tesseract_cmd: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OCR_TESSERACT_CMD", "TESSERACT_CMD"),
    )
    ocr_dpi: int = Field(default=300, validation_alias=AliasChoices("OCR_DPI"))
    ocr_page_limit: int | None = Field(
        default=None,
        validation_alias=AliasChoices("OCR_PAGE_LIMIT"),
    )
    ocr_timeout_seconds: float | None = Field(
        default=None,
        validation_alias=AliasChoices("OCR_TIMEOUT_SECONDS"),
    )

    @field_validator("ocr_dpi", mode="before")
    @classmethod
    def _validate_ocr_dpi(cls, value: object) -> int:
        candidate = int(value)
        return 72 if candidate < 72 else candidate

    @field_validator("ocr_page_limit", mode="before")
    @classmethod
    def _validate_ocr_page_limit(cls, value: object | None) -> int | None:
        if value in (None, "", 0, "0"):
            return None
        candidate = int(value)
        return candidate if candidate > 0 else None

    @field_validator("ocr_timeout_seconds", mode="before")
    @classmethod
    def _validate_ocr_timeout(cls, value: object | None) -> float | None:
        if value in (None, "", 0, 0.0, "0", "0.0"):
            return None
        candidate = float(value)
        return candidate if candidate > 0 else None

    # HTML conversion --------------------------------------------------
    document_parser_backend: str = Field(
        default="legacy",
        validation_alias=AliasChoices("DOCUMENT_PARSER_BACKEND"),
        description="Document parser backend: legacy|docling|auto.",
    )
    docling_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("DOCLING_ENABLED"),
        description="Enable Docling parser integration path.",
    )
    docling_timeout: float = Field(
        default=60.0,
        validation_alias=AliasChoices("DOCLING_TIMEOUT"),
        description="Docling conversion timeout in seconds.",
    )
    docling_ocr_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("DOCLING_OCR_ENABLED"),
        description="Enable OCR in Docling parser when supported.",
    )
    docling_max_pages: int | None = Field(
        default=None,
        validation_alias=AliasChoices("DOCLING_MAX_PAGES"),
        description="Maximum number of pages to parse via Docling.",
    )

    @field_validator("document_parser_backend", mode="before")
    @classmethod
    def _validate_document_parser_backend(cls, value: object) -> str:
        backend = str(value or "legacy").strip().lower()
        return backend if backend in {"legacy", "docling", "auto"} else "legacy"

    @field_validator("docling_timeout", mode="before")
    @classmethod
    def _validate_docling_timeout(cls, value: object) -> float:
        candidate = float(value or 60.0)
        return candidate if candidate > 0 else 60.0

    @field_validator("docling_max_pages", mode="before")
    @classmethod
    def _validate_docling_max_pages(cls, value: object | None) -> int | None:
        if value in (None, "", 0, "0"):
            return None
        candidate = int(value)
        return candidate if candidate > 0 else None

    html2text_bodywidth: int = Field(
        default=0,
        validation_alias=AliasChoices("HTML2TEXT_BODYWIDTH"),
        description="Maximum column width for html2text output. Zero disables wrapping.",
    )
    html2text_links: bool = Field(
        default=False,
        validation_alias=AliasChoices("HTML2TEXT_LINKS", "HTML2TEXT_INCLUDE_LINKS"),
        description="Include link targets in converted HTML output when set to true.",
    )
    html2text_ignore_images: bool = Field(
        default=True,
        validation_alias=AliasChoices("HTML2TEXT_IGNORE_IMAGES"),
        description="Ignore image tags during HTML conversion when set to true.",
    )
    html2text_ignore_emphasis: bool = Field(
        default=True,
        validation_alias=AliasChoices("HTML2TEXT_IGNORE_EMPHASIS"),
        description="Strip emphasis markers such as bold and italics when converting HTML.",
    )
    html2text_inline_links: bool = Field(
        default=False,
        validation_alias=AliasChoices("HTML2TEXT_INLINE_LINKS"),
        description="Render links inline rather than collecting references at the end of paragraphs.",
    )
    html2text_single_line_break: bool = Field(
        default=False,
        validation_alias=AliasChoices("HTML2TEXT_SINGLE_LINE_BREAK"),
        description="Collapse consecutive line breaks into a single break during HTML conversion.",
    )
    html2text_wrap_links: bool = Field(
        default=True,
        validation_alias=AliasChoices("HTML2TEXT_WRAP_LINKS"),
        description="Allow html2text to wrap long link targets across multiple lines.",
    )
    html2text_unicode_snob: bool = Field(
        default=False,
        validation_alias=AliasChoices("HTML2TEXT_UNICODE_SNOB"),
        description="Prefer unicode characters for typographical symbols when converting HTML.",
    )

    # Vector store -------------------------------------------------------
    vector_backend: str = Field(default="qdrant", validation_alias=AliasChoices("VECTOR_BACKEND"))
    vector_embed_model: str = Field(
        default="intfloat/multilingual-e5-small",
        validation_alias=AliasChoices("VECTOR_EMBED_MODEL", "EMBED_MODEL"),
    )
    vector_embed_dimension: int = Field(
        default=384,
        validation_alias=AliasChoices("VECTOR_EMBED_DIMENSION", "EMBED_DIMENSION"),
    )
    embed_batch_size: int = Field(
        default=64,
        validation_alias=AliasChoices("EMBED_BATCH_SIZE", "VECTOR_EMBED_BATCH_SIZE"),
    )
    vector_e5_prefix: bool = Field(
        default=False,
        validation_alias=AliasChoices("VECTOR_E5_PREFIX"),
        description=(
            "Prepend e5 'query:'/'passage:' prefixes at encode time. Only affects "
            "e5-family models (no-op otherwise) and REQUIRES a reindex so passages "
            "are embedded with the 'passage:' prefix."
        ),
    )
    qdrant_url: str = Field(
        default="",
        validation_alias=AliasChoices("QDRANT_URL"),
        description="HTTP endpoint for an external Qdrant instance. Leave blank to use the embedded store.",
    )
    qdrant_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_PATH", "QDRANT_STORAGE_PATH"),
        description="Filesystem directory used for embedded Qdrant storage.",
    )
    qdrant_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_API_KEY")
    )
    qdrant_collection: str = Field(
        default="kb_chunks", validation_alias=AliasChoices("QDRANT_COLLECTION")
    )

    # Memory -------------------------------------------------------------
    chat_memory_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("CHAT_MEMORY_ENABLED", "MEMORY_ENABLED"),
    )
    memory_db_path: Path | None = Field(
        default=None, validation_alias=AliasChoices("MEMORY_DB_PATH")
    )
    chat_memory_ttl_days: int = Field(
        default=90,
        validation_alias=AliasChoices("CHAT_MEMORY_TTL_DAYS", "MEMORY_TTL_DAYS"),
    )
    chat_memory_max_tokens: int = Field(
        default=2000,
        validation_alias=AliasChoices("CHAT_MEMORY_MAXTOK", "MEMORY_MAX_TOKENS"),
    )

    # LLM ----------------------------------------------------------------
    use_lora: bool = Field(
        default=True,
        validation_alias=AliasChoices("USE_LORA", "ENABLE_LORA"),
    )
    llm_provider: str = Field(default="llama-cpp", validation_alias=AliasChoices("LLM_PROVIDER"))
    llm_model_name: str = Field(
        default="kb-llama",
        validation_alias=AliasChoices("LLM_MODEL_NAME", "GEN_MODEL"),
    )
    llm_model_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_MODEL_VERSION", "MODEL_VERSION"),
    )
    llm_model_path: Path = Field(
        default=Path("./models/model.gguf"),
        validation_alias=AliasChoices("LLM_MODEL_PATH", "LLAMA_MODEL_PATH"),
    )
    llm_ctx: int = Field(default=4096, validation_alias=AliasChoices("LLM_CTX", "LLAMA_CTX"))
    llm_threads: int = Field(default=4, validation_alias=AliasChoices("LLM_THREADS"))
    llm_gpu_layers: int = Field(default=0, validation_alias=AliasChoices("LLM_GPU_LAYERS"))
    llm_temperature: float = Field(default=0.7, validation_alias=AliasChoices("LLM_TEMPERATURE"))
    llm_top_p: float = Field(default=0.95, validation_alias=AliasChoices("LLM_TOP_P"))
    llm_top_k: int = Field(default=40, validation_alias=AliasChoices("LLM_TOP_K"))
    llm_max_tokens: int = Field(
        default=1024,
        validation_alias=AliasChoices("LLM_MAX_TOKENS", "MAX_GENERATION_TOKENS"),
    )
    llm_api_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_BASE_URL", "OPENAI_BASE_URL"),
    )
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_api_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_API_MODEL", "OPENAI_MODEL"),
    )
    llm_api_timeout_sec: float = Field(
        default=60.0,
        validation_alias=AliasChoices("LLM_API_TIMEOUT_SEC"),
    )
    llm_api_retries: int = Field(
        default=2,
        validation_alias=AliasChoices("LLM_API_RETRIES"),
    )
    llm_api_backoff_sec: float = Field(
        default=0.5,
        validation_alias=AliasChoices("LLM_API_BACKOFF_SEC"),
    )

    # LangChain integration ---------------------------------------------
    langchain_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("LANGCHAIN_ENABLED")
    )
    langchain_mode: str = Field(default="legacy", validation_alias=AliasChoices("LANGCHAIN_MODE"))
    langchain_use_history_aware: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGCHAIN_USE_HISTORY_AWARE"),
    )
    langchain_return_source_docs: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGCHAIN_RETURN_SOURCE_DOCS"),
    )
    langchain_tracing: bool = Field(
        default=False, validation_alias=AliasChoices("LANGCHAIN_TRACING")
    )
    langchain_project: str = Field(
        default="kb-ai", validation_alias=AliasChoices("LANGCHAIN_PROJECT")
    )
    redis_url: str | None = Field(default=None, validation_alias=AliasChoices("REDIS_URL"))

    llm_lora_adapter: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_LORA_ADAPTER", "LLM_ADAPTER", "OLLAMA_ADAPTER"),
    )
    lora_default_adapter: str | None = Field(
        default="none",
        validation_alias=AliasChoices("LORA_DEFAULT_ADAPTER", "DEFAULT_LORA_ADAPTER"),
    )
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    )
    lora_adapter_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LORA_ADAPTER_PATH"),
    )
    lora_scaling: float = Field(default=1.0, validation_alias=AliasChoices("LORA_SCALING"))
    lora_adapter_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LORA_ADAPTER_VERSION"),
    )
    lora_registry_dir: Path = Field(
        default=Path("./data/lora/registry"),
        validation_alias=AliasChoices("LORA_REGISTRY_DIR"),
    )
    lora_train_output_dir: Path = Field(
        default=Path("./data/lora/runs"),
        validation_alias=AliasChoices("LORA_TRAIN_OUTPUT_DIR"),
    )
    lora_train_base_model: str = Field(
        default="meta-llama/Llama-3-8b-Instruct",
        validation_alias=AliasChoices("LORA_TRAIN_BASE_MODEL"),
    )
    lora_train_max_seq_len: int = Field(
        default=4096,
        validation_alias=AliasChoices("LORA_TRAIN_MAX_SEQ_LEN"),
    )
    lora_train_fp16: bool = Field(default=True, validation_alias=AliasChoices("LORA_FP16"))
    lora_train_bf16: bool = Field(default=False, validation_alias=AliasChoices("LORA_BF16"))
    lora_use_qlora: bool = Field(default=True, validation_alias=AliasChoices("LORA_USE_QLORA"))

    # Security -----------------------------------------------------------
    secret_key: str = Field(default="change-me", validation_alias=AliasChoices("SECRET_KEY"))
    jwt_algorithm: str = Field(default="HS256", validation_alias=AliasChoices("JWT_ALGORITHM"))
    access_token_expire_minutes: int = Field(
        default=30,
        validation_alias=AliasChoices("ACCESS_TOKEN_EXPIRE_MINUTES"),
    )
    api_key_hash_salt: str = Field(
        default="kb-ai-salt", validation_alias=AliasChoices("API_KEY_HASH_SALT")
    )

    # Billing ------------------------------------------------------------
    billing_enabled: bool = Field(default=False, validation_alias=AliasChoices("BILLING_ENABLED"))
    billing_provider: str = Field(default="none", validation_alias=AliasChoices("BILLING_PROVIDER"))

    # Audit log ----------------------------------------------------------
    audit_log_retention_days: int = Field(
        default=0,
        validation_alias=AliasChoices("AUDIT_LOG_RETENTION_DAYS"),
        description="Days of audit_log history to retain. Zero (default) disables purging.",
    )

    @field_validator("audit_log_retention_days", mode="before")
    @classmethod
    def _validate_audit_retention(cls, value: object) -> int:
        if value in {None, "", Ellipsis}:
            return 0
        if not isinstance(value, (str, int, float)):
            raise ValueError("audit_log_retention_days must be numeric")
        candidate = int(value)
        if candidate < 0:
            raise ValueError("audit_log_retention_days cannot be negative")
        return candidate

    # Validators ---------------------------------------------------------
    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _normalise_origins(cls, value: object) -> list[str]:
        if value is None or value == "" or value is Ellipsis:
            return ["*"]
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
            return items or ["*"]
        if isinstance(value, Sequence):
            items = [str(item).strip() for item in value if str(item).strip()]
            return items or ["*"]
        raise ValueError("CORS origins must be a string or iterable")

    @field_validator("cors_allow_origins", mode="after")
    @classmethod
    def _ensure_origins(cls, value: list[str]) -> list[str]:
        return value or ["*"]

    @field_validator(
        "chat_memory_enabled",
        "rerank_enabled",
        "langchain_enabled",
        "langchain_use_history_aware",
        "langchain_return_source_docs",
        "langchain_tracing",
        "billing_enabled",
        mode="before",
    )
    @classmethod
    def _parse_bool(cls, value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @field_validator("rate_burst", mode="before")
    @classmethod
    def _ensure_rate_burst(cls, value: object) -> int:
        if value in {None, "", Ellipsis}:
            return 0
        return int(value)

    @field_validator(
        "data_dir",
        "chat_db_path",
        "memory_db_path",
        "qdrant_path",
        "llm_model_path",
        "lora_adapter_path",
        "lora_registry_dir",
        "lora_train_output_dir",
        mode="before",
    )
    @classmethod
    def _expand_paths(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            return Path(value).expanduser()
        return value

    @field_validator("data_dir", mode="after")
    @classmethod
    def _ensure_data_dir(cls, value: Path) -> Path:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return Path("./var/data").expanduser()
            value = Path(value)
        return value.expanduser()

    @field_validator("chat_db_path", "memory_db_path", "qdrant_path", mode="after")
    @classmethod
    def _resolve_optional_path(cls, value: Path | None) -> Path | None:
        if value in {None, "", Ellipsis}:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            value = Path(stripped)
        return value.expanduser()

    @field_validator("lora_registry_dir", "lora_train_output_dir", mode="after")
    @classmethod
    def _ensure_directory(cls, value: Path) -> Path:
        resolved = Path(value).expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @model_validator(mode="after")
    def _ensure_precision_flags(self) -> "Settings":
        if self.lora_train_fp16 and self.lora_train_bf16:
            raise ValueError("Only one of LORA_FP16 or LORA_BF16 can be enabled")
        return self

    @field_validator("rerank_topk", mode="before")
    @classmethod
    def _coerce_optional_int(cls, value: object) -> int | None:
        if value in {None, "", Ellipsis}:
            return None
        return int(value)

    @field_validator("langchain_mode", mode="before")
    @classmethod
    def _validate_langchain_mode(cls, value: object) -> str:
        mode = str(value or "legacy").strip().lower()
        return mode if mode in {"legacy", "lcel", "agent"} else "legacy"

    @field_validator("rate_limit_backend", mode="before")
    @classmethod
    def _validate_rate_limit_backend(cls, value: object) -> str:
        backend = str(value or "memory").strip().lower()
        return backend if backend in {"memory", "redis"} else "memory"

    @field_validator("billing_provider", mode="before")
    @classmethod
    def _validate_billing_provider(cls, value: object) -> str:
        provider = str(value or "none").strip().lower()
        return provider if provider else "none"

    # Computed helpers ---------------------------------------------------
    @computed_field
    @property
    def files_dir(self) -> Path:
        return self.data_dir / self.files_subdir

    @computed_field
    @property
    def chat_db_path_resolved(self) -> Path:
        base = self.chat_db_path or (self.data_dir / "db" / "chat_history.sqlite")
        return Path(base)

    @computed_field
    @property
    def memory_db_path_resolved(self) -> Path:
        base = self.memory_db_path or (self.data_dir / "db" / "memory.sqlite")
        return Path(base)

    @computed_field
    @property
    def qdrant_path_resolved(self) -> Path:
        base = self.qdrant_path or (self.data_dir / "qdrant")
        return Path(base)

    @computed_field
    @property
    def rerank_limit(self) -> int:
        candidate = self.rerank_topk or self.retrieve_topk
        candidate = max(1, int(candidate))
        return min(self.retrieve_topk, candidate)

    @computed_field
    @property
    def citations_bounds(self) -> tuple[int, int]:
        minimum = max(1, int(self.chat_min_citations))
        maximum = max(minimum, int(self.chat_max_citations))
        return minimum, maximum

    @computed_field
    @property
    def gen_model(self) -> str:
        return self.llm_model_name

    @computed_field
    @property
    def lora_registry_path(self) -> Path:
        return Path(self.lora_registry_dir)

    @computed_field
    @property
    def lora_runs_path(self) -> Path:
        return Path(self.lora_train_output_dir)

    def iter_secret_fields(self) -> Iterable[str]:
        """Return names of settings that contain sensitive values."""

        return ("secret_key", "qdrant_api_key", "chat_db_dsn", "llm_api_key")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""

    return Settings()


def get_version_info(settings: Settings | None = None) -> dict[str, Any]:
    """Return structured version information for API responses."""

    resolved = settings or get_settings()
    app_version = (resolved.app_version or "0.0.0").strip() or "0.0.0"

    def _clean(value: str | None, *, fallback: str = "unknown") -> str:
        if value is None:
            return fallback
        cleaned = str(value).strip()
        return cleaned or fallback

    model_version = _clean(resolved.llm_model_version)
    adapter_version = _clean(resolved.lora_adapter_version)
    payload: dict[str, Any] = {
        "app": {"version": app_version},
        "model": {
            "name": resolved.llm_model_name,
            "version": model_version,
        },
        "lora": {
            "adapter": resolved.llm_lora_adapter,
            "version": adapter_version,
            "enabled": bool(resolved.llm_lora_adapter or resolved.lora_adapter_path),
        },
    }
    return payload


__all__ = ["Settings", "get_settings", "get_version_info"]
