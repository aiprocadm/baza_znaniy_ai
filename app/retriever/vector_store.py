"""Common vector store interfaces and factory helpers."""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from app.core.config import Settings, get_settings


@runtime_checkable
class VectorStore(Protocol):
    """Protocol describing the interface exposed by vector stores."""

    def ensure_ready(self) -> None:  # pragma: no cover - protocol definition
        """Ensure that the backing index exists and is configured."""

    def upsert(self, chunks: Iterable[dict[str, object]]) -> None:  # pragma: no cover - protocol definition
        """Persist or update a batch of document chunks."""

    def search(
        self,
        query: str,
        top_k: int,
        *,
        owner: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, object]]:  # pragma: no cover - protocol definition
        """Return the top matching chunks for the supplied query."""



def _load_backends():
    from .faiss import FaissVectorStore
    from .qdrant import QdrantVectorStore

    return {
        "faiss": FaissVectorStore,
        "qdrant": QdrantVectorStore,
    }


def _build_backend(settings: Settings) -> VectorStore:
    """Instantiate the backend requested by :class:`Settings`."""

    backend = (settings.vector_backend or "").lower()
    backends = _load_backends()
    factory = backends.get(backend)
    if factory is None:
        raise ValueError(f"Unsupported vector backend: {settings.vector_backend!r}")
    return factory(settings=settings)


_DEFAULT_STORE: VectorStore | None = None


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the cached vector store instance for the active settings."""

    global _DEFAULT_STORE
    if settings is None:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = _build_backend(get_settings())
        return _DEFAULT_STORE
    return _build_backend(settings)


def _clear_cache() -> None:
    global _DEFAULT_STORE
    _DEFAULT_STORE = None


setattr(get_vector_store, "cache_clear", _clear_cache)


__all__ = ["VectorStore", "get_vector_store", "_build_backend"]
