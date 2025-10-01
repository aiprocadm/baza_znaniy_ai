"""Application configuration models and helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import AliasChoices, Field

try:  # pragma: no cover - optional features may be unavailable in tests
    from pydantic import BaseModel, computed_field, field_validator
except ImportError:  # pragma: no cover - provide light-weight fallbacks
    from pydantic import BaseModel  # type: ignore[assignment]

    def computed_field(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return property(func)

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

    def field_validator(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

try:  # pragma: no cover - ``pydantic-settings`` is optional
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - minimal shim for tests
    import os

    class SettingsConfigDict(dict):  # type: ignore[override]
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    class BaseSettings(BaseModel):  # type: ignore[misc]
        model_config = SettingsConfigDict()

        def __init__(self, **data: object) -> None:
            values: dict[str, object] = {}
            for name in getattr(self, "__annotations__", {}):
                env_name = name.upper()
                env_value = os.getenv(env_name)
                if env_value is not None:
                    values[name] = env_value
            values.update(data)
            super().__init__(**values)


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
    rate_burst: int = Field(default=0, validation_alias=AliasChoices("RATE_BURST"))

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
    max_upload_mb: int = Field(
        default=25,
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
    chat_db_schema: str | None = Field(default=None, validation_alias=AliasChoices("CHAT_DB_SCHEMA"))
    chat_history_limit: int = Field(default=12, validation_alias=AliasChoices("CHAT_HISTORY_LIMIT"))
    chat_summary_trigger: int = Field(
        default=10,
        validation_alias=AliasChoices("CHAT_SUMMARY_TRIGGER", "MEMORY_SUMMARY_TRIGGER"),
    )
    chat_min_citations: int = Field(default=3, validation_alias=AliasChoices("CHAT_MIN_CITATIONS"))
    chat_max_citations: int = Field(default=5, validation_alias=AliasChoices("CHAT_MAX_CITATIONS"))

    # Retrieval ----------------------------------------------------------
    retrieve_topk: int = Field(default=10, validation_alias=AliasChoices("RETRIEVE_TOPK"))
    rerank_enabled: bool = Field(default=False, validation_alias=AliasChoices("RERANK_ENABLED"))
    rerank_topk: int | None = Field(
        default=50,
        validation_alias=AliasChoices("RERANK_TOPK", "RERANK_TOP_K"),
    )
    rag_tokenizer_name: str = Field(default="cl100k_base", validation_alias=AliasChoices("RAG_TOKENIZER_NAME"))
    rag_chunk: int = Field(default=900, validation_alias=AliasChoices("RAG_CHUNK"))
    rag_overlap: int = Field(default=140, validation_alias=AliasChoices("RAG_OVERLAP"))
    ingest_max_retries: int = Field(default=3, validation_alias=AliasChoices("INGEST_MAX_RETRIES"))
    ingest_backoff_seconds: float = Field(
        default=1.0,
        validation_alias=AliasChoices("INGEST_BACKOFF_SECONDS", "INGEST_BACKOFF_BASE"),
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
    qdrant_api_key: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_API_KEY"))
    qdrant_collection: str = Field(default="kb_chunks", validation_alias=AliasChoices("QDRANT_COLLECTION"))

    # Memory -------------------------------------------------------------
    chat_memory_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("CHAT_MEMORY_ENABLED", "MEMORY_ENABLED"),
    )
    memory_db_path: Path | None = Field(default=None, validation_alias=AliasChoices("MEMORY_DB_PATH"))
    chat_memory_ttl_days: int = Field(
        default=90,
        validation_alias=AliasChoices("CHAT_MEMORY_TTL_DAYS", "MEMORY_TTL_DAYS"),
    )
    chat_memory_max_tokens: int = Field(
        default=2000,
        validation_alias=AliasChoices("CHAT_MEMORY_MAXTOK", "MEMORY_MAX_TOKENS"),
    )

    # LLM ----------------------------------------------------------------
    llm_provider: str = Field(default="llama-cpp", validation_alias=AliasChoices("LLM_PROVIDER"))
    llm_model_name: str = Field(
        default="kb-llama",
        validation_alias=AliasChoices("LLM_MODEL_NAME", "GEN_MODEL"),
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
    lora_adapter_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LORA_ADAPTER_PATH"),
    )
    lora_scaling: float = Field(default=1.0, validation_alias=AliasChoices("LORA_SCALING"))

    # Security -----------------------------------------------------------
    secret_key: str = Field(default="change-me", validation_alias=AliasChoices("SECRET_KEY"))
    jwt_algorithm: str = Field(default="HS256", validation_alias=AliasChoices("JWT_ALGORITHM"))
    access_token_expire_minutes: int = Field(
        default=30,
        validation_alias=AliasChoices("ACCESS_TOKEN_EXPIRE_MINUTES"),
    )

    # Validators ---------------------------------------------------------
    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _normalise_origins(cls, value: object) -> list[str]:
        if value in {None, "", Ellipsis}:
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
        mode="before",
    )
    @classmethod
    def _expand_paths(cls, value: object) -> object:
        if isinstance(value, str) and value:
            return Path(value).expanduser()
        return value

    @field_validator("data_dir", mode="after")
    @classmethod
    def _ensure_data_dir(cls, value: Path) -> Path:
        return value.expanduser()

    @field_validator("chat_db_path", "memory_db_path", "qdrant_path", mode="after")
    @classmethod
    def _resolve_optional_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser()

    @field_validator("rerank_topk", mode="before")
    @classmethod
    def _coerce_optional_int(cls, value: object) -> int | None:
        if value in {None, "", Ellipsis}:
            return None
        return int(value)

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

    def iter_secret_fields(self) -> Iterable[str]:
        """Return names of settings that contain sensitive values."""

        return ("secret_key", "qdrant_api_key", "chat_db_dsn")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
