"""Common vector store interfaces and factory helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Protocol, runtime_checkable

from app.core.config import Settings, get_settings


@runtime_checkable
class VectorStore(Protocol):
    """Minimal protocol implemented by vector store backends."""

    def ensure_ready(self) -> None:
        """Ensure the underlying store is ready for use."""

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:
        """Insert or update the provided chunks in the store."""

    def search(self, query: str, top_k: int) -> list[dict[str, object]]:
        """Return the ``top_k`` most relevant chunks for ``query``."""


def _build_backend(settings: Settings) -> VectorStore:
    """Instantiate the configured vector store implementation."""

    from .faiss import FaissVectorStore
    from .qdrant import QdrantVectorStore

    backend = settings.vector_backend
    if backend == "qdrant":
        return QdrantVectorStore(settings=settings)
    if backend == "faiss":
        return FaissVectorStore(settings=settings)
    raise ValueError(f"Unsupported vector backend: {backend}")


@lru_cache(maxsize=1)
def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the cached vector store instance for the given settings."""

    resolved_settings = settings or get_settings()
    return _build_backend(resolved_settings)


__all__ = ["VectorStore", "get_vector_store"]
