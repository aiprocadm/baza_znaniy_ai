"""Application configuration models and helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from pydantic import AliasChoices, BaseModel, Field

try:  # pragma: no cover - support for environments without computed_field
    from pydantic import computed_field
except ImportError:  # pragma: no cover - test stubs

    def computed_field(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return property(func)

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator

try:  # pragma: no cover - support for environments without field_validator
    from pydantic import field_validator
except ImportError:  # pragma: no cover - test stubs

    def field_validator(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        if args and callable(args[0]):
            return decorator(args[0])
        return decorator
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - test stubs
    import os
    from dataclasses import dataclass

    @dataclass
    class SettingsConfigDict(dict):
        env_file: tuple[str, ...] | None = None
        env_file_encoding: str | None = None
        extra: str | None = None

    class BaseSettings(BaseModel):  # type: ignore[misc]
        model_config = SettingsConfigDict()

        def __init__(self, **data: object) -> None:
            values: dict[str, object] = {}
            for name in getattr(self, "__annotations__", {}):
                field = getattr(self.__class__, name, None)
                aliases: list[str] = []
                metadata = getattr(field, "metadata", None)
                if metadata:
                    alias_spec = metadata.get("validation_alias")
                    if isinstance(alias_spec, AliasChoices):
                        aliases.extend(alias_spec)
                    elif isinstance(alias_spec, str):
                        aliases.append(alias_spec)
                if not aliases:
                    aliases.append(name.upper())
                for alias in aliases:
                    env_value = os.getenv(alias)
                    if env_value is not None:
                        values[name] = env_value
                        break
            values.update(data)
            super().__init__(**values)


class Settings(BaseSettings):
    """Configuration loaded from environment variables and ``.env`` files."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(
        default=Path("/opt/knowlab/data/files"),
        validation_alias=AliasChoices("DATA_DIR", "FILES_ROOT"),
    )
    chat_db_backend: str = Field(
        default="sqlite",
        validation_alias=AliasChoices("CHAT_DB_BACKEND"),
    )
    chat_db_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("CHAT_DB_PATH"),
    )
    chat_db_dsn: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CHAT_DB_DSN", "CHAT_DB_URL", "DATABASE_URL"),
    )
    chat_db_schema: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CHAT_DB_SCHEMA"),
    )
    chat_history_limit: int = Field(
        default=12,
        validation_alias=AliasChoices("CHAT_HISTORY_LIMIT"),
    )
    chat_summary_trigger: int = Field(
        default=10,
        validation_alias=AliasChoices("CHAT_SUMMARY_TRIGGER", "MEMORY_SUMMARY_TRIGGER"),
    )
    chat_min_citations: int = Field(
        default=3,
        validation_alias=AliasChoices("CHAT_MIN_CITATIONS"),
    )
    chat_max_citations: int = Field(
        default=5,
        validation_alias=AliasChoices("CHAT_MAX_CITATIONS"),
    )
    retrieve_topk: int = Field(
        default=10,
        validation_alias=AliasChoices("RETRIEVE_TOPK"),
    )
    rerank_topk: int | None = Field(
        default=None,
        validation_alias=AliasChoices("RERANK_TOPK"),
    )
    chat_memory_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("CHAT_MEMORY_ENABLED", "MEMORY_ENABLED"),
    )
    memory_db_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("MEMORY_DB_PATH"),
    )
    chat_memory_ttl_days: int = Field(
        default=90,
        validation_alias=AliasChoices("CHAT_MEMORY_TTL_DAYS", "MEMORY_TTL_DAYS"),
    )
    chat_memory_max_tokens: int = Field(
        default=2000,
        validation_alias=AliasChoices("CHAT_MEMORY_MAXTOK", "MEMORY_MAX_TOKENS"),
    )
    rag_tokenizer_name: str = Field(
        default="cl100k_base",
        validation_alias=AliasChoices("RAG_TOKENIZER_NAME"),
    )
    rag_chunk: int = Field(
        default=900,
        validation_alias=AliasChoices("RAG_CHUNK"),
    )
    rag_overlap: int = Field(
        default=140,
        validation_alias=AliasChoices("RAG_OVERLAP"),
    )
    vector_backend: str = Field(
        default="qdrant",
        validation_alias=AliasChoices("VECTOR_BACKEND"),
    )
    qdrant_url: str = Field(
        default="http://qdrant:6333",
        validation_alias=AliasChoices("QDRANT_URL"),
    )
    qdrant_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QDRANT_API_KEY"),
    )
    qdrant_collection: str = Field(
        default="kb_chunks",
        validation_alias=AliasChoices("QDRANT_COLLECTION"),
    )
    vector_embed_model: str = Field(
        default="intfloat/multilingual-e5-small",
        validation_alias=AliasChoices("VECTOR_EMBED_MODEL", "EMBED_MODEL"),
    )
    vector_embed_dimension: int = Field(
        default=384,
        validation_alias=AliasChoices("VECTOR_EMBED_DIMENSION", "EMBED_DIMENSION"),
    )
    llm_provider: str = Field(
        default="ollama",
        validation_alias=AliasChoices("LLM_PROVIDER"),
    )
    llm_model_name: str = Field(
        default="qwen2.5:3b-instruct",
        validation_alias=AliasChoices("LLM_MODEL_NAME", "GEN_MODEL", "OLLAMA_MODEL"),
    )
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    )
    max_context_tokens: int = Field(
        default=4096,
        validation_alias=AliasChoices("MAX_CONTEXT_TOKENS"),
    )
    secret_key: str = Field(
        default="change-me",
        validation_alias=AliasChoices("SECRET_KEY"),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALGORITHM"),
    )
    access_token_expire_minutes: int = Field(
        default=30,
        validation_alias=AliasChoices("ACCESS_TOKEN_EXPIRE_MINUTES"),
    )

    @field_validator(
        "chat_history_limit",
        "chat_summary_trigger",
        "chat_min_citations",
        "chat_max_citations",
        "retrieve_topk",
        "rag_chunk",
        "rag_overlap",
        "vector_embed_dimension",
        "chat_memory_ttl_days",
        "chat_memory_max_tokens",
        "access_token_expire_minutes",
        mode="before",
    )
    @classmethod
    def _ensure_int(cls, value: object) -> int:
        return int(value) if value not in {None, ""} else 0

    @field_validator(
        "chat_memory_enabled",
        mode="before",
    )
    @classmethod
    def _normalise_bool(cls, value: object) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @field_validator("ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

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
    def rerank_limit(self) -> int:
        candidate = self.rerank_topk or self.retrieve_topk
        candidate = max(1, candidate)
        return min(self.retrieve_topk, candidate)

    @computed_field
    @property
    def citations_bounds(self) -> tuple[int, int]:
        minimum = max(1, self.chat_min_citations)
        maximum = max(minimum, self.chat_max_citations)
        return minimum, maximum

    @field_validator("chat_db_backend")
    @classmethod
    def _normalise_backend(cls, value: str) -> str:
        return (value or "sqlite").strip().lower()

    @field_validator("vector_backend")
    @classmethod
    def _normalise_vector_backend(cls, value: str) -> str:
        return (value or "qdrant").strip().lower()

    @field_validator("llm_provider")
    @classmethod
    def _normalise_provider(cls, value: str) -> str:
        return (value or "ollama").strip().lower()

    @field_validator("data_dir", "chat_db_path", "memory_db_path", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> object:
        if isinstance(value, str) and value:
            return Path(value).expanduser()
        return value

    @field_validator("chat_db_path", "memory_db_path", mode="after")
    @classmethod
    def _absolute_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser().resolve()

    @field_validator("data_dir", mode="after")
    @classmethod
    def _ensure_dir(cls, value: Path) -> Path:
        return value.expanduser()

    def iter_secret_fields(self) -> Iterable[str]:
        """Return names of settings that contain secrets."""

        return ("secret_key", "qdrant_api_key", "chat_db_dsn")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
