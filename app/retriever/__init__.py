"""Retriever package exposing the configured vector store."""

from app.core.config import Settings
from .vector_store import (
    FaissVectorStore,
    QdrantVectorStore,
    VectorStore,
    get_vector_store,
)

__all__ = [
    "VectorStore",
    "QdrantVectorStore",
    "FaissVectorStore",
    "get_vector_store",
    "Settings",
]
