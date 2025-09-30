"""Search endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core.deps import get_tenant
from app.models import SearchHit, SearchResponse
from app.services.vectorstore import search

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search_endpoint(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    tenant: str = Depends(get_tenant),
) -> SearchResponse:
    """Perform a similarity search without invoking the LLM."""

    hits = search(query, top_k=top_k)
    models = [
        SearchHit(
            file=item.get("file"),
            page=item.get("page"),
            score=float(item.get("score", 0.0)),
            text=item.get("text", ""),
        )
        for item in hits
    ]
    return SearchResponse(query=query, hits=models)
