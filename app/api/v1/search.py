"""Search endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.core.auth import ensure_tenant_access, get_current_active_user
from app.models.user import UserRecord
from app.models import SearchHit, SearchResponse
from app.retriever.vector_store import SearchFilters
from app.services.vectorstore import search

RevisionMode = str

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse)
def search_endpoint(
    user: UserRecord = Depends(get_current_active_user),
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    owner: str | None = Query(None, min_length=1),
    tags: list[str] | None = Query(default=None),
    act_type: str | None = Query(None),
    issuer: str | None = Query(None),
    reg_number: str | None = Query(None),
    is_active: bool | None = Query(None),
    revision_mode: str = Query("current", pattern="^(current|historical)$"),
    tenant: str = Depends(ensure_tenant_access),
) -> SearchResponse:
    """Perform a similarity search without invoking the LLM."""

    effective_tenant = tenant.strip() if isinstance(tenant, str) else ""
    if not effective_tenant:
        raise ValueError("tenant context is required")
    filters = SearchFilters.from_input(
        tenant_id=effective_tenant,
        owner=owner,
        tags=tags,
        act_type=act_type,
        issuer=issuer,
        reg_number=reg_number,
        is_active=is_active,
        revision_mode=revision_mode,
    )
    hits = search(
        query,
        top_k=top_k,
        owner=filters.owner,
        tags=list(filters.tags) or None,
        act_type=filters.act_type,
        issuer=filters.issuer,
        reg_number=filters.reg_number,
        is_active=filters.is_active,
        revision_mode=filters.revision_mode,
        tenant_id=filters.tenant_id,
    )
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
