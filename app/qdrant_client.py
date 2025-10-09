"""Backward compatible shims for Qdrant helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Iterator, Protocol, runtime_checkable, cast

from app.core.config import get_settings


@runtime_checkable
class QdrantSettings(Protocol):
    """Protocol describing the configuration required by this module."""

    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    qdrant_path_resolved: str
    vector_embed_model: str
    vector_embed_dimension: int

try:  # pragma: no cover - optional dependency surface during import
    from app.retriever import VectorStore, get_vector_store as _get_vector_store_factory
except ImportError as exc:  # pragma: no cover - import guard executed in tests
    _VECTOR_STORE_IMPORT_ERROR = exc
    _get_vector_store_factory = None  # type: ignore[assignment]
else:
    _VECTOR_STORE_IMPORT_ERROR = None

_SETTING_EXPORTS = {
    "QDRANT_URL": "qdrant_url",
    "QDRANT_API_KEY": "qdrant_api_key",
    "QDRANT_COLLECTION": "qdrant_collection",
    "QDRANT_PATH": "qdrant_path_resolved",
    "EMBED_MODEL": "vector_embed_model",
    "EMBED_DIMENSION": "vector_embed_dimension",
}


@lru_cache(maxsize=1)
def _cached_settings() -> QdrantSettings:
    settings = get_settings()
    if not isinstance(settings, QdrantSettings):
        # ``SimpleNamespace`` instances used in tests satisfy the protocol but do
        # not register automatically; cast to appease type-checkers.
        return cast(QdrantSettings, settings)
    return settings


@lru_cache(maxsize=1)
def _cached_vector_store() -> VectorStore:
    if _get_vector_store_factory is None:
        raise RuntimeError(
            "app.retriever.get_vector_store is unavailable; ensure the retriever "
            "package is installed correctly",
        ) from _VECTOR_STORE_IMPORT_ERROR

    store = _get_vector_store_factory(_cached_settings())
    if store is None:  # pragma: no cover - defensive guard against misconfigured factories
        raise RuntimeError("Vector store factory returned None")
    return store


def _resolve_vector_store() -> VectorStore:
    return _cached_vector_store()


def ensure_collection() -> None:
    _resolve_vector_store().ensure_ready()


def upsert_chunks(chunks: Iterable[dict[str, object]]) -> None:
    _resolve_vector_store().upsert(chunks)


def search_chunks(query: str, top_k: int = 10) -> list[dict[str, object]]:
    return _resolve_vector_store().search(query, top_k)


def reset_collection() -> None:
    store = _resolve_vector_store()
    if hasattr(store, "reset_collection"):
        store.reset_collection()
        return
    raise NotImplementedError("Active vector store does not support resetting the index")


def export_payloads(batch_size: int = 256) -> Iterator[dict[str, object]]:
    store = _resolve_vector_store()
    if hasattr(store, "export_payloads"):
        return store.export_payloads(batch_size=batch_size)
    raise NotImplementedError("Active vector store does not support exporting payloads")


def import_payloads(payloads: Iterable[dict[str, object]]) -> None:
    store = _resolve_vector_store()
    if hasattr(store, "import_payloads"):
        store.import_payloads(payloads)
        return
    raise NotImplementedError("Active vector store does not support importing payloads")


def __getattr__(name: str):  # pragma: no cover - exercised indirectly
    setting_name = _SETTING_EXPORTS.get(name)
    if setting_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    settings = _cached_settings()
    value = getattr(settings, setting_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - developer ergonomics
    return sorted(set(globals()) | set(_SETTING_EXPORTS))


def _clear_cache() -> None:
    """Helper used in tests to clear cached settings/vector store."""

    _cached_settings.cache_clear()
    _cached_vector_store.cache_clear()

__all__ = (
    *tuple(_SETTING_EXPORTS.keys()),
    "ensure_collection",
    "upsert_chunks",
    "search_chunks",
    "reset_collection",
    "export_payloads",
    "import_payloads",
)
