"""Convenience exports for retriever components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .vector_store import VectorStore, get_vector_store

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .faiss import FaissVectorStore
    from .qdrant import QdrantVectorStore
    from .rerank import (
        CrossEncoderReranker,
        apply_rerank,
        get_rerank_top_k,
        get_reranker,
        is_rerank_enabled,
    )


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    if name == "FaissVectorStore":
        from .faiss import FaissVectorStore

        return FaissVectorStore
    if name == "QdrantVectorStore":
        from .qdrant import QdrantVectorStore

        return QdrantVectorStore
    if name in {
        "CrossEncoderReranker",
        "apply_rerank",
        "get_rerank_top_k",
        "get_reranker",
        "is_rerank_enabled",
    }:
        from . import rerank

        return getattr(rerank, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "QdrantVectorStore",
    "CrossEncoderReranker",
    "apply_rerank",
    "get_rerank_top_k",
    "get_reranker",
    "is_rerank_enabled",
    "get_vector_store",
]
