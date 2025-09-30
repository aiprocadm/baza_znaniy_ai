"""Helpers for indexing and searching document chunks."""

from __future__ import annotations

import logging
from typing import Iterable, List

from app.qdrant_client import ensure_collection, search_chunks as qdrant_search, upsert_chunks

LOGGER = logging.getLogger(__name__)

_FALLBACK_INDEX: List[dict[str, object]] = []


def index_chunks(chunks: Iterable[dict[str, object]]) -> int:
    """Store *chunks* in the primary vector store or the in-memory fallback."""

    items = list(chunks)
    if not items:
        return 0

    try:
        ensure_collection()
        upsert_chunks(items)
        return len(items)
    except Exception:  # pragma: no cover - gracefully degrade when Qdrant is unavailable
        LOGGER.exception("Falling back to in-memory index")
        _FALLBACK_INDEX.extend(items)
        return len(items)


def _search_fallback(query: str, top_k: int) -> List[dict[str, object]]:
    """Very small substring-based search over the fallback index."""

    if not query:
        return []
    hits: List[tuple[float, dict[str, object]]] = []
    needle = query.lower()
    for chunk in _FALLBACK_INDEX:
        text = str(chunk.get("text", ""))
        haystack = text.lower()
        if not haystack:
            continue
        if needle in haystack:
            score = haystack.count(needle) / (len(text) or 1)
            hits.append((score, chunk))
    hits.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _score, chunk in hits[:top_k]]


def search(query: str, top_k: int = 10) -> List[dict[str, object]]:
    """Run a similarity search returning at most *top_k* hits."""

    try:
        ensure_collection()
        return qdrant_search(query, top_k=top_k)
    except Exception:  # pragma: no cover - fallback path used in tests
        LOGGER.exception("Falling back to in-memory search")
        return _search_fallback(query, top_k)


def clear_fallback() -> None:
    """Reset the in-memory fallback index (used in tests)."""

    _FALLBACK_INDEX.clear()


__all__ = ["index_chunks", "search", "clear_fallback"]
