"""Helpers for indexing and searching document chunks."""

from __future__ import annotations

import logging
import time
from typing import Iterable, List

from app.core.config import get_settings
from app.retriever.vector_store import get_vector_store
from app.observability.metrics import record_index_operation, record_search_operation

LOGGER = logging.getLogger(__name__)

_DEFAULT_FALLBACK: List[dict[str, object]] = []
_FALLBACK_STORAGE: List[dict[str, object]] | None = _DEFAULT_FALLBACK
_VECTOR_STORE: object | None = None


def set_fallback_storage(storage: List[dict[str, object]] | None) -> None:
    """Configure the container used for the in-memory fallback index."""

    global _FALLBACK_STORAGE
    if storage is None:
        storage = []
    _FALLBACK_STORAGE = storage


def get_fallback_storage() -> List[dict[str, object]]:
    """Return the active fallback container, creating one when needed."""

    global _FALLBACK_STORAGE
    if _FALLBACK_STORAGE is None:
        _FALLBACK_STORAGE = []
    return _FALLBACK_STORAGE


def _resolve_vector_store():
    """Lazily instantiate and cache the primary vector store."""

    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = get_vector_store(get_settings())
    return _VECTOR_STORE


def index_chunks(chunks: Iterable[dict[str, object]]) -> int:
    """Store *chunks* in the primary vector store or the in-memory fallback."""

    items = list(chunks)
    if not items:
        return 0

    start = time.perf_counter()

    try:
        store = _resolve_vector_store()
    except Exception:  # pragma: no cover - backend unavailable or optional deps missing
        duration = time.perf_counter() - start
        record_index_operation("error", "vector", len(items), duration)
        LOGGER.exception("Falling back to in-memory index")
        fallback_start = time.perf_counter()
        get_fallback_storage().extend(items)
        record_index_operation(
            "success", "fallback", len(items), time.perf_counter() - fallback_start
        )
        return len(items)

    try:
        store.ensure_ready()
        store.upsert(items)
    except Exception:  # pragma: no cover - gracefully degrade when Qdrant is unavailable
        duration = time.perf_counter() - start
        record_index_operation("error", "vector", len(items), duration)
        LOGGER.exception("Falling back to in-memory index")
        fallback_start = time.perf_counter()
        get_fallback_storage().extend(items)
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
    for chunk in get_fallback_storage():
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
        store = _resolve_vector_store()
        store.ensure_ready()
        hits = store.search(query, top_k=top_k)
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

    get_fallback_storage().clear()


__all__ = [
    "index_chunks",
    "search",
    "clear_fallback",
    "set_fallback_storage",
    "get_fallback_storage",
]
