"""Convenience imports for retriever components."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "QdrantVectorStore",
    "CrossEncoderReranker",
    "apply_rerank",
    "get_reranker",
    "get_rerank_top_k",
    "is_rerank_enabled",
    "get_vector_store",
]

_EXPORTS = {
    "VectorStore": (".vector_store", "VectorStore"),
    "get_vector_store": (".vector_store", "get_vector_store"),
    "FaissVectorStore": (".faiss", "FaissVectorStore"),
    "QdrantVectorStore": (".qdrant", "QdrantVectorStore"),
    "CrossEncoderReranker": (".rerank", "CrossEncoderReranker"),
    "apply_rerank": (".rerank", "apply_rerank"),
    "get_reranker": (".rerank", "get_reranker"),
    "get_rerank_top_k": (".rerank", "get_rerank_top_k"),
    "is_rerank_enabled": (".rerank", "is_rerank_enabled"),
}


def __getattr__(name: str) -> Any:
    """Dynamically import retriever components on first access."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the module exports for interactive helpers."""

    return sorted(__all__)
