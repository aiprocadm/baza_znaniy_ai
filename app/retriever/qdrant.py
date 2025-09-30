"""Compatibility wrapper exposing Qdrant vector store symbols."""

from __future__ import annotations

from .vector_store import QdrantVectorStore, get_vector_store

__all__ = ["QdrantVectorStore", "get_vector_store"]

