"""LangChain-compatible retriever adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.retriever.vector_store import SearchFilters

try:  # pragma: no cover - optional dependency
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    class Document:  # type: ignore[override]
        def __init__(self, page_content: str, metadata: dict[str, object] | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}


@dataclass
class TenantFilteredQdrantRetriever:
    """Retriever enforcing tenant isolation for Qdrant-backed searches."""

    store: Any
    tenant_id: str
    k: int = 10

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        if not tenant:
            raise ValueError("tenant_id is required")
        self.tenant_id = tenant

    def get_relevant_documents(self, query: str, **kwargs: object) -> list[Document]:
        metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else {}
        filters = SearchFilters.from_input(
            tenant_id=self.tenant_id,
            owner=metadata.get("owner"),  # type: ignore[arg-type]
            tags=metadata.get("tags"),  # type: ignore[arg-type]
            act_type=metadata.get("act_type"),  # type: ignore[arg-type]
            issuer=metadata.get("issuer"),  # type: ignore[arg-type]
            reg_number=metadata.get("reg_number"),  # type: ignore[arg-type]
            is_active=metadata.get("is_active"),  # type: ignore[arg-type]
            revision_mode=metadata.get("revision", "current"),  # type: ignore[arg-type]
        )
        hits = self.store.search(query=query, top_k=int(kwargs.get("k", self.k)), filters=filters)
        return self.store.hits_to_documents(hits)

    def invoke(self, query: str, **kwargs: object) -> list[Document]:
        return self.get_relevant_documents(query, **kwargs)

