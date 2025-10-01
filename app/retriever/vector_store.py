        codex/refactor-upload-and-ingest-apis-to-use-ingestservice
        # codex/implement-vector-store-interface-and-refactor-qdrant-logic
"""Vector store abstractions and concrete implementations."""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, Iterator, List, MutableMapping, Protocol, Sequence, runtime_checkable

import faiss  # type: ignore[import-untyped]
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

        codex/clean-up-code-and-run-tests
"""Vector store protocol and factory helpers."""
        main

"""Common vector store interfaces and factory helpers."""
        main

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable
        codex/refactor-upload-and-ingest-apis-to-use-ingestservice
        # main

        main

from app.core.config import Settings, get_settings
        codex/clean-up-code-and-run-tests
from .faiss import FaissVectorStore
from .qdrant import QdrantVectorStore

        codex/refactor-upload-and-ingest-apis-to-use-ingestservice

        # codex/implement-vector-store-interface-and-refactor-qdrant-logic
logger = logging.getLogger(__name__)

        main
        main


@runtime_checkable
class VectorStore(Protocol):
        codex/clean-up-code-and-run-tests
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


        codex/clean-up-retriever-module-exports
def _clear_cache() -> None:
    global _cached_store
    _cached_store = None


get_vector_store.cache_clear = _clear_cache  # type: ignore[attr-defined]


# Backwards compatibility alias for older tests/imports
_build_store = _build_backend


__all__ = ["VectorStore", "get_vector_store", "_build_backend"]

__all__ = ["VectorStore", "get_vector_store"]
          codex/refactor-upload-and-ingest-apis-to-use-ingestservice
        # main

          main
          main
        main
