"""Similarity search endpoint (protected)."""

from __future__ import annotations
from fastapi import Request
from .common import protected, _hit_to_out, _store_for
from .rag import _retrieve_with_rerank
from .schemas import SearchRequest, SearchResponse


@protected.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest, request: Request) -> SearchResponse:
    """Run a similarity search, optionally followed by cross-encoder rerank."""

    store = _store_for(request)
    hits, rerank_info = _retrieve_with_rerank(store, payload.query, payload.top_k)
    return SearchResponse(
        query=payload.query,
        hits=[_hit_to_out(hit) for hit in hits],
        rerank=rerank_info,
    )
