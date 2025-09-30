"""Vector store protocol and factory helpers."""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from app.core.config import Settings, get_settings
from .faiss import FaissVectorStore
from .qdrant import QdrantVectorStore


@runtime_checkable
class VectorStore(Protocol):
    """Protocol implemented by vector store backends."""

    def ensure_ready(self) -> None:
        """Ensure the underlying store is ready for use."""

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        """Insert or update the provided chunks."""

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Return up to ``top_k`` most relevant chunks for ``query``."""


def _build_backend(settings: Settings) -> VectorStore:
    """Instantiate the configured vector store backend."""

    backend = (settings.vector_backend or "qdrant").strip().lower()
    if backend == "qdrant":
        return QdrantVectorStore(settings=settings)
    if backend == "faiss":
        return FaissVectorStore(settings=settings)
    raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")


_DEFAULT_CACHE: dict[str, VectorStore] = {}


def _cached_backend(settings: Settings) -> VectorStore:
    backend = (settings.vector_backend or "qdrant").strip().lower()
    store = _DEFAULT_CACHE.get(backend)
    if store is None:
        store = _build_backend(settings)
        _DEFAULT_CACHE[backend] = store
    return store


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the configured vector store implementation."""

    resolved = settings or get_settings()
    if settings is None:
        return _cached_backend(resolved)
    return _build_backend(resolved)


def _cache_clear() -> None:
    _DEFAULT_CACHE.clear()


setattr(get_vector_store, "cache_clear", _cache_clear)


__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "QdrantVectorStore",
    "get_vector_store",
]
