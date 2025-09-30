"""Retriever package exposing vector store implementations and helpers."""
        codex/create-vectorstore-package-and-implementations

from app.core.config import Settings



from __future__ import annotations

import importlib
from typing import Any

from app.core.config import Settings
        codex/implement-reranking-functionality-and-tests

from .qdrant import QdrantVectorStore, get_vector_store
from .rerank import CrossEncoderReranker

__all__ = ["QdrantVectorStore", "CrossEncoderReranker", "get_vector_store", "Settings"]


      codex/create-vectorstore-package-and-implementations-p7jgtz
        main
from .faiss import FaissVectorStore
from .qdrant import QdrantVectorStore
from .vector_store import VectorStore, get_vector_store

__all__ = [
    "FaissVectorStore",
    "QdrantVectorStore",
    "VectorStore",
    "get_vector_store",
    "Settings",
]
        codex/create-vectorstore-package-and-implementations


__all__ = [
    "CrossEncoderReranker",
    "QdrantVectorStore",
    "apply_rerank",
    "get_rerank_top_k",
    "get_reranker",
    "get_vector_store",
    "is_rerank_enabled",
    "Settings",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial passthrough
    if name in {"QdrantVectorStore", "get_vector_store"}:
        module = importlib.import_module(".qdrant", __name__)
        return getattr(module, name)
    if name in {
        "CrossEncoderReranker",
        "apply_rerank",
        "get_rerank_top_k",
        "get_reranker",
        "is_rerank_enabled",
    }:
        module = importlib.import_module(".rerank", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial passthrough
    return sorted(set(globals()) | set(__all__))
        main
        main
