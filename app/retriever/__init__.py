"""Retriever package exposing vector store and reranking utilities."""

from __future__ import annotations

from app.core.config import Settings
from .faiss import FaissVectorStore
from .qdrant import QdrantVectorStore, get_vector_store
from .rerank import (
    CrossEncoderReranker,
    apply_rerank,
    get_rerank_top_k,
    get_reranker,
    is_rerank_enabled,
)
from .vector_store import VectorStore

__all__ = [
    "VectorStore",
    "QdrantVectorStore",
    "FaissVectorStore",
    "get_vector_store",
    "CrossEncoderReranker",
    "apply_rerank",
    "get_rerank_top_k",
    "get_reranker",
    "is_rerank_enabled",
    "Settings",
]
