"""Common vector store interfaces and factory helpers."""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from app.core.config import Settings, get_settings


@runtime_checkable
class VectorStore(Protocol):
    """Protocol describing the behaviour of vector store backends."""

    def ensure_ready(self) -> None:
        """Ensure the underlying resources exist and are initialised."""

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        """Insert or update the provided chunks in the index."""

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Run a similarity search returning up to *top_k* results."""


_cached_store: VectorStore | None = None


def _build_backend(settings: Settings) -> VectorStore:
    backend = settings.vector_backend.lower()
    if backend == "faiss":
        from .faiss import FaissVectorStore

        return FaissVectorStore(settings)
    if backend == "qdrant":
        from .qdrant import QdrantVectorStore

        return QdrantVectorStore(settings)
    raise RuntimeError(f"Unsupported vector backend: {backend}")


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return a cached vector store instance configured via *settings*."""

    global _cached_store
    if settings is not None:
        _cached_store = _build_backend(settings)
        return _cached_store
    if _cached_store is None:
        _cached_store = _build_backend(get_settings())
    return _cached_store


def _clear_cache() -> None:
    global _cached_store
    _cached_store = None


get_vector_store.cache_clear = _clear_cache  # type: ignore[attr-defined]


# Backwards compatibility alias for older tests/imports
_build_store = _build_backend


__all__ = ["VectorStore", "get_vector_store", "_build_backend"]
