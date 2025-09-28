"""Application settings and helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Type

from pydantic import AliasChoices, BaseModel, Field


class Settings(BaseModel):
    """Runtime configuration loaded from the environment."""

    # General service configuration
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_reload: bool = Field(default=False, alias="APP_RELOAD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: Path = Field(default=Path("/data/storage"), alias="DATA_DIR")
    app_secret: str = Field(default="", alias="APP_SECRET")
    basic_user: str = Field(default="admin", alias="BASIC_USER")
    rate_limit: str = Field(default="30r/m", alias="RATE_LIMIT")
    rate_burst: int = Field(default=20, alias="RATE_BURST")

    # Retrieval settings
    rag_chunk: int = Field(default=900, alias="RAG_CHUNK")
    rag_overlap: int = Field(default=140, alias="RAG_OVERLAP")
    rag_tokenizer_name: str = Field(default="cl100k_base", alias="RAG_TOKENIZER_NAME")
    retrieve_topk: int = Field(default=10, alias="RETRIEVE_TOPK")
    rerank_topk: int = Field(default=10, alias="RERANK_TOPK")

    # Qdrant configuration
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="kb_chunks", alias="QDRANT_COLLECTION")
    embed_model: str = Field(default="intfloat/multilingual-e5-small", alias="EMBED_MODEL")
    embed_dimension: int = Field(
        default=384, alias=AliasChoices("EMBED_DIMENSION", "VECTOR_SIZE")
    )

    # Ollama / generation
    ollama_base_url: str = Field(default="http://ollama:11434", alias="OLLAMA_BASE_URL")
    gen_model: str = Field(default="qwen2.5:3b-instruct", alias="GEN_MODEL")

    # Chat behaviour
    chat_history_limit: int = Field(default=12, alias="CHAT_HISTORY_LIMIT")
    chat_summary_trigger: int = Field(default=10, alias="CHAT_SUMMARY_TRIGGER")
    chat_min_citations: int = Field(default=3, alias="CHAT_MIN_CITATIONS")
    chat_max_citations: int = Field(default=5, alias="CHAT_MAX_CITATIONS")
    chat_db_path: Path | None = Field(default=None, alias="CHAT_DB_PATH")

    # Long term memory
    chat_memory_enabled: bool = Field(default=False, alias="CHAT_MEMORY_ENABLED")
    chat_memory_db_path: Path | None = Field(default=None, alias="CHAT_MEMORY_DB_PATH")
    chat_memory_ttl_days: int = Field(default=90, alias="CHAT_MEMORY_TTL_DAYS")
    chat_memory_max_tokens: int = Field(default=2000, alias="CHAT_MEMORY_MAXTOK")

    class Config:
        populate_by_name = True

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def chat_database(self) -> Path:
        if self.chat_db_path:
            return self.chat_db_path
        return self.data_dir / "db" / "chat.sqlite"

    @property
    def memory_database(self) -> Path:
        if self.chat_memory_db_path:
            return self.chat_memory_db_path
        return self.data_dir / "db" / "memory.sqlite"


def _normalise_aliases(alias_value: Any, fallback: str) -> Tuple[str, ...]:
    if alias_value is None:
        return (fallback,)
    if isinstance(alias_value, (tuple, list, set)):
        return tuple(str(item) for item in alias_value)
    return (str(alias_value),)


def _build_field_metadata() -> Dict[str, Dict[str, Any]]:
    """Return settings field metadata keyed by attribute name."""

    metadata: Dict[str, Dict[str, Any]] = {}
    fields = getattr(Settings, "model_fields", None)
    if fields:
        for name, field in fields.items():
            alias_value = getattr(field, "alias", None)
            aliases = _normalise_aliases(alias_value, name.upper())
            annotation = getattr(field, "annotation", Any)
            metadata[name] = {"aliases": aliases, "annotation": annotation}
        return metadata

    legacy_fields = getattr(Settings, "__fields__", {})
    for name, field in legacy_fields.items():
        alias_value = getattr(field, "alias", None)
        aliases = _normalise_aliases(alias_value, name.upper())
        metadata[name] = {"aliases": aliases, "annotation": getattr(field, "type_", Any)}
    if metadata:
        return metadata

    annotations = getattr(Settings, "__annotations__", {})
    for name, annotation in annotations.items():
        field = getattr(Settings, name, ...)
        alias_value = getattr(field, "alias", None) if field is not ... else None
        aliases = _normalise_aliases(alias_value, name.upper())
        metadata[name] = {"aliases": aliases, "annotation": annotation}
    return metadata


_FIELD_METADATA = _build_field_metadata()


def _coerce_value(value: str, annotation: Type[Any]) -> Any:
    """Convert environment strings to the expected field types."""

    if annotation is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    if annotation is int:
        return int(value)
    if annotation is Path:
        return Path(value)
    return value


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    overrides: Dict[str, Any] = {}
    for name, info in _FIELD_METADATA.items():
        aliases: Iterable[str] = info["aliases"]
        for alias in aliases:
            if alias in os.environ:
                overrides[name] = _coerce_value(os.environ[alias], info["annotation"])
                break
    return Settings(**overrides)
