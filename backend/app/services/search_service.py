from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from app.retriever.rerank import apply_rerank, get_rerank_top_k, get_reranker, is_rerank_enabled
from app.retriever.vector_store import get_vector_store

from backend.app.schemas.knowledge_base import SearchRequest, SearchResponse, SearchResultItem, ServiceHealth


class SearchService:
    def __init__(self) -> None:
        self._store = get_vector_store()

    def index_document(self, *, file_id: str, file_name: str, text: str, owner: str | None = None, tags: list[str] | None = None) -> None:
        self._store.ensure_ready()
        self._store.upsert([
            {
                "id": file_id,
                "sha256": file_id,
                "file": file_name,
                "text": text,
                "owner": owner,
                "tags": tags or [],
                "page": 1,
            }
        ])

    def search(self, payload: SearchRequest) -> SearchResponse:
        self._store.ensure_ready()
        hits: list[dict[str, Any]] = self._store.search(payload.query, max(payload.top_k, get_rerank_top_k()), owner=payload.owner, tags=payload.tags)
        rerank_enabled = is_rerank_enabled(default=False)
        reranker = get_reranker() if rerank_enabled else None
        ordered = apply_rerank(payload.query, hits, payload.top_k, rerank_enabled, reranker)
        results = [
            SearchResultItem(
                id=f"r_{str(hit.get('id', idx))}",
                title=str(hit.get("file") or "document"),
                snippet=str(hit.get("text") or "")[:280],
                score=float(hit.get("score") or 0.0),
                source=str(hit.get("sha256") or hit.get("id") or ""),
                updated_at=datetime.now(timezone.utc),
            )
            for idx, hit in enumerate(ordered)
        ]
        return SearchResponse(results=results, total=len(results))

    def health(self) -> ServiceHealth:
        try:
            self._store.ensure_ready()
            return ServiceHealth(name="vector_store", status="healthy", latency_ms=0)
        except Exception as exc:  # noqa: BLE001
            return ServiceHealth(name="vector_store", status="offline", latency_ms=0, last_error=str(exc))


search_service = SearchService()
