"""Helpers for indexing and searching document chunks."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Iterable, MutableSequence
from threading import Lock
from typing import List

try:  # pragma: no cover - optional dependency for deployments with Qdrant
    from qdrant_client.http.exceptions import (
        ApiException as QdrantApiException,
        ResponseHandlingException as QdrantResponseHandlingException,
        UnexpectedResponse as QdrantUnexpectedResponse,
    )
except Exception:  # pragma: no cover - fallback when qdrant-client not installed
    QdrantApiException = QdrantResponseHandlingException = QdrantUnexpectedResponse = None

from app.core.config import get_settings
from app.retriever.vector_store import get_vector_store
from app.observability.metrics import record_index_operation, record_search_operation

LOGGER = logging.getLogger(__name__)

FallbackStorage = MutableSequence[dict[str, object]] | deque[dict[str, object]]

_DEFAULT_FALLBACK: FallbackStorage = deque()
_FALLBACK_STORAGE: FallbackStorage | None = _DEFAULT_FALLBACK
_FALLBACK_LOCK = Lock()
_VECTOR_STORE: object | None = None

_VECTOR_ERRORS: tuple[type[Exception], ...] = tuple(
    filter(
        None,
        (
            RuntimeError,
            ValueError,
            ConnectionError,
            TimeoutError,
            ImportError,
            QdrantApiException,
            QdrantResponseHandlingException,
            QdrantUnexpectedResponse,
        ),
    )
)


def set_fallback_storage(storage: FallbackStorage | None) -> None:
    """Configure the container used for the in-memory fallback index."""

    global _FALLBACK_STORAGE
    with _FALLBACK_LOCK:
        if storage is None:
            storage = deque()
        _FALLBACK_STORAGE = storage


def get_fallback_storage() -> FallbackStorage:
    """Return the active fallback container, creating one when needed."""

    global _FALLBACK_STORAGE
    if _FALLBACK_STORAGE is None:
        with _FALLBACK_LOCK:
            if _FALLBACK_STORAGE is None:
                _FALLBACK_STORAGE = deque()
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

    def _store_in_fallback() -> int:
        fallback_start = time.perf_counter()
        with _FALLBACK_LOCK:
            get_fallback_storage().extend(items)
        record_index_operation(
            "success", "fallback", len(items), time.perf_counter() - fallback_start
        )
        return len(items)

    try:
        store = _resolve_vector_store()
    except _VECTOR_ERRORS as exc:  # pragma: no cover - backend unavailable
        duration = time.perf_counter() - start
        record_index_operation("error", "vector", len(items), duration)
        LOGGER.exception("Falling back to in-memory index", exc_info=exc)
        return _store_in_fallback()

    try:
        store.ensure_ready()
        store.upsert(items)
    except _VECTOR_ERRORS as exc:  # pragma: no cover - gracefully degrade when Qdrant is unavailable
        duration = time.perf_counter() - start
        record_index_operation("error", "vector", len(items), duration)
        LOGGER.exception("Falling back to in-memory index", exc_info=exc)
        return _store_in_fallback()

    record_index_operation("success", "vector", len(items), time.perf_counter() - start)
    return len(items)


def _search_fallback(
    query: str,
    top_k: int,
    *,
    owner: str | None = None,
    tags: list[str] | None = None,
    act_type: str | None = None,
    issuer: str | None = None,
    reg_number: str | None = None,
    is_active: bool | None = None,
    revision_mode: str = "current",
) -> List[dict[str, object]]:
    """Very small substring-based search over the fallback index."""

    if not query:
        return []
    hits: List[tuple[float, dict[str, object]]] = []
    needle = query.lower()
    with _FALLBACK_LOCK:
        snapshot = list(get_fallback_storage())

    normalized_tags = {tag.strip().lower() for tag in (tags or []) if tag and tag.strip()}
    normalized_owner = (owner or "").strip().lower()

    for chunk in snapshot:
        if normalized_owner:
            chunk_owner = str(chunk.get("owner", "")).strip().lower()
            if chunk_owner != normalized_owner:
                continue
        if normalized_tags:
            chunk_tags = chunk.get("tags")
            if not isinstance(chunk_tags, list):
                continue
            chunk_tag_set = {
                str(tag).strip().lower() for tag in chunk_tags if str(tag).strip()
            }
            if not normalized_tags.issubset(chunk_tag_set):
                continue

        meta = chunk.get("meta") if isinstance(chunk.get("meta"), dict) else {}
        if act_type and str(meta.get("act_type", "")).strip().lower() != act_type.strip().lower():
            continue
        if issuer and issuer.strip().lower() not in str(meta.get("issuer", "")).strip().lower():
            continue
        if reg_number and reg_number.strip().lower() != str(meta.get("reg_number", "")).strip().lower():
            continue
        if is_active is not None and bool(meta.get("is_active", True)) is not is_active:
            continue
        if revision_mode == "current" and meta.get("is_active") is False:
            continue
        if revision_mode == "historical" and meta.get("is_active") is True:
            continue

        text = str(chunk.get("text", ""))
        haystack = text.lower()
        if not haystack:
            continue
        if needle in haystack:
            score = haystack.count(needle) / (len(text) or 1)
            if bool(meta.get("is_active", True)):
                score += 0.2
            score += min(0.2, float(bool(reg_number and meta.get("reg_number"))))
            hits.append((score, chunk))
    hits.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _score, chunk in hits[:top_k]]


def search(
    query: str,
    top_k: int = 10,
    *,
    owner: str | None = None,
    tags: list[str] | None = None,
    act_type: str | None = None,
    issuer: str | None = None,
    reg_number: str | None = None,
    is_active: bool | None = None,
    revision_mode: str = "current",
) -> List[dict[str, object]]:
    """Run a similarity search returning at most *top_k* hits."""

    start = time.perf_counter()

    try:
        store = _resolve_vector_store()
        store.ensure_ready()
        hits = store.search(query, top_k=top_k, owner=owner, tags=tags)
    except _VECTOR_ERRORS as exc:  # pragma: no cover - fallback path used in tests
        duration = time.perf_counter() - start
        record_search_operation("vector", "error", duration, 0)
        LOGGER.exception("Falling back to in-memory search", exc_info=exc)
        fallback_start = time.perf_counter()
        fallback_hits = _search_fallback(query, top_k, owner=owner, tags=tags, act_type=act_type, issuer=issuer, reg_number=reg_number, is_active=is_active, revision_mode=revision_mode)
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

    with _FALLBACK_LOCK:
        get_fallback_storage().clear()


__all__ = [
    "index_chunks",
    "search",
    "clear_fallback",
    "set_fallback_storage",
    "get_fallback_storage",
]
