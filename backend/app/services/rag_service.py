from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

from backend.app.schemas.knowledge_base import SearchRequest, SearchResponse
from backend.app.services.search_service import search_service


class RagService:
    def search(self, payload, *, tenant_slug: str):
        search_response = search_service.search(
            SearchRequest(query=payload.query, top_k=payload.top_k, tags=payload.tags),
            tenant_slug=tenant_slug,
        )
        sources = [
            {
                "id": item.id,
                "title": item.title,
                "snippet": item.snippet,
                "score": item.score,
                "source": item.source,
            }
            for item in search_response.results
        ]
        debug = None
        if payload.include_debug:
            debug = {
                "rewritten_query": payload.query,
                "top_k": payload.top_k,
                "retriever_mode": payload.retriever_mode,
            }
        return {"answer": "", "sources": sources, "debug": debug}

    def query(self, payload, *, tenant_slug: str):
        result = self.search(payload, tenant_slug=tenant_slug)
        answer = f"Found {len(result['sources'])} sources for: {payload.query}"
        return {"answer": answer, "sources": result["sources"]}

    async def stream_query(self, payload, *, tenant_slug: str) -> AsyncIterator[tuple[str, dict]]:
        response = self.query(payload, tenant_slug=tenant_slug)
        for token in response["answer"].split():
            yield "token", {"token": token}
        for source in response["sources"]:
            yield "source", source
        yield "done", {"finished_at": datetime.now(timezone.utc).isoformat()}

    def compat_search(self, payload: SearchRequest, *, tenant_slug: str) -> SearchResponse:
        return search_service.search(payload, tenant_slug=tenant_slug)

    def compat_chat(self, payload, *, tenant_slug: str):
        query_payload = type("Tmp", (), {"query": payload.query, "top_k": payload.top_k, "tags": None, "include_debug": False, "retriever_mode": "vector"})
        return self.query(query_payload, tenant_slug=tenant_slug)


rag_service = RagService()
