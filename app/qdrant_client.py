"""Backward compatible shims for Qdrant helpers."""

from __future__ import annotations

from typing import Iterable, Iterator

from app.core.config import get_settings
from app.retriever import get_vector_store

_settings = get_settings()
_vector_store = get_vector_store(_settings)

QDRANT_URL = _settings.qdrant_url
QDRANT_API_KEY = _settings.qdrant_api_key
QDRANT_COLLECTION = _settings.qdrant_collection
EMBED_MODEL = _settings.vector_embed_model
EMBED_DIMENSION = _settings.vector_embed_dimension


def ensure_collection() -> None:
    _vector_store.ensure_ready()


def upsert_chunks(chunks: Iterable[dict[str, object]]) -> None:
    _vector_store.upsert(chunks)


def search_chunks(query: str, top_k: int = 10) -> list[dict[str, object]]:
    return _vector_store.search(query, top_k)


def reset_collection() -> None:
    _vector_store.reset_collection()


def export_payloads(batch_size: int = 256) -> Iterator[dict[str, object]]:
    return _vector_store.export_payloads(batch_size=batch_size)


def import_payloads(payloads: Iterable[dict[str, object]]) -> None:
    _vector_store.import_payloads(payloads)


__all__ = [
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "EMBED_MODEL",
    "EMBED_DIMENSION",
    "ensure_collection",
    "upsert_chunks",
    "search_chunks",
    "reset_collection",
    "export_payloads",
    "import_payloads",
]
