"""Helpers for indexing and searching document chunks."""

from __future__ import annotations

import logging
import time
from typing import Iterable, List

from app.core.config import get_settings
from app.retriever import get_vector_store
from app.observability.metrics import record_index_operation, record_search_operation

LOGGER = logging.getLogger(__name__)

_FALLBACK_INDEX: List[dict[str, object]] = []
_VECTOR_STORE = get_vector_store(get_settings())


def index_chunks(chunks: Iterable[dict[str, object]]) -> int:
    """Store *chunks* in the primary vector store or the in-memory fallback."""

    items = list(chunks)
    if not items:
        return 0

    start = time.perf_counter()

    try:
        _VECTOR_STORE.ensure_ready()
        _VECTOR_STORE.upsert(items)
    except Exception:  # pragma: no cover - gracefully degrade when Qdrant is unavailable
        duration = time.perf_counter() - start
        record_index_operation("error", "vector", len(items), duration)
        LOGGER.exception("Falling back to in-memory index")
        fallback_start = time.perf_counter()
        _FALLBACK_INDEX.extend(items)
        record_index_operation(
            "success", "fallback", len(items), time.perf_counter() - fallback_start
        )
        return len(items)

    record_index_operation("success", "vector", len(items), time.perf_counter() - start)
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

    start = time.perf_counter()

    try:
        _VECTOR_STORE.ensure_ready()
        hits = _VECTOR_STORE.search(query, top_k=top_k)
    except Exception:  # pragma: no cover - fallback path used in tests
        duration = time.perf_counter() - start
        record_search_operation("vector", "error", duration, 0)
        LOGGER.exception("Falling back to in-memory search")
        fallback_start = time.perf_counter()
        fallback_hits = _search_fallback(query, top_k)
        record_search_operation(
            "fallback",
            "success",
            time.perf_counter() - fallback_start,
            len(fallback_hits),
        )
        return fallback_hits

    record_search_operation("vector", "success", time.perf_counter() - start, len(hits))
    return hits


def clear_fallback() -> None:
    """Reset the in-memory fallback index (used in tests)."""

    _FALLBACK_INDEX.clear()


__all__ = ["index_chunks", "search", "clear_fallback"]
