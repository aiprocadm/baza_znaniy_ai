from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.app.api.deps import get_tenant_context
from backend.app.schemas.knowledge_base import SearchRequest, SearchResponse
from backend.app.services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["rag"])


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=10, ge=1, le=50)
    retriever_mode: Literal["vector", "hybrid"] = "vector"
    tags: list[str] | None = None
    include_debug: bool = False


class RagSourceItem(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    source: str


class RagSearchDebug(BaseModel):
    rewritten_query: str
    top_k: int
    retriever_mode: str


class RagSearchResponse(BaseModel):
    answer: str
    sources: list[RagSourceItem]
    debug: RagSearchDebug | None = None


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=50)


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[RagSourceItem]


class ChatCompatRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=50)


class ChatCompatResponse(BaseModel):
    answer: str
    sources: list[RagSourceItem]


@router.post("/search", response_model=RagSearchResponse)
def rag_search(payload: RagSearchRequest, tenant_ctx: tuple[str, str] = Depends(get_tenant_context)) -> RagSearchResponse:
    _, tenant_slug = tenant_ctx
    return rag_service.search(payload, tenant_slug=tenant_slug)


@router.post("/query", response_model=RagQueryResponse)
def rag_query(payload: RagQueryRequest, tenant_ctx: tuple[str, str] = Depends(get_tenant_context)) -> RagQueryResponse:
    _, tenant_slug = tenant_ctx
    return rag_service.query(payload, tenant_slug=tenant_slug)


@router.post("/query/stream")
async def rag_query_stream(payload: RagQueryRequest, tenant_ctx: tuple[str, str] = Depends(get_tenant_context)) -> EventSourceResponse:
    _, tenant_slug = tenant_ctx

    async def _event_gen() -> AsyncIterator[dict[str, str]]:
        try:
            async for event_name, data in rag_service.stream_query(payload, tenant_slug=tenant_slug):
                yield {"event": event_name, "data": json.dumps(data, ensure_ascii=False)}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(_event_gen())


@router.post("/compat/search", response_model=SearchResponse)
def compat_search(payload: SearchRequest, tenant_ctx: tuple[str, str] = Depends(get_tenant_context)) -> SearchResponse:
    _, tenant_slug = tenant_ctx
    return rag_service.compat_search(payload, tenant_slug=tenant_slug)


@router.post("/compat/chat", response_model=ChatCompatResponse)
def compat_chat(payload: ChatCompatRequest, tenant_ctx: tuple[str, str] = Depends(get_tenant_context)) -> ChatCompatResponse:
    _, tenant_slug = tenant_ctx
    return rag_service.compat_chat(payload, tenant_slug=tenant_slug)
