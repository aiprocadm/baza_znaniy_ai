from __future__ import annotations

from app.langchain.retrievers import TenantFilteredQdrantRetriever
from app.services import vectorstore as vectorstore_service


class _StoreStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.settings = type("S", (), {"qdrant_collection": "kb"})()

    def search(self, *, query: str, top_k: int, filters):  # noqa: ANN001
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [{"text": "hello", "tenant_id": filters.tenant_id, "owner": filters.owner, "tags": list(filters.tags)}]

    def hits_to_documents(self, hits):  # noqa: ANN001
        docs = []
        for hit in hits:
            payload = dict(hit)
            text = str(payload.pop("text", ""))
            docs.append(type("Doc", (), {"page_content": text, "metadata": payload})())
        return docs

    # reindex API
    def resolve_collection_name(self, alias: str) -> str:
        return f"{alias}__v1"

    def create_collection_like(self, target: str, source: str) -> None:
        self.created = (target, source)

    def export_payloads_from_collection(self, source: str):  # noqa: ANN001
        yield {"document_id": "doc-1", "text": "a", "vector": [0.1], "id": "1"}

    def import_payloads_to_collection(self, target: str, payloads):  # noqa: ANN001
        self.imported = (target, list(payloads))

    def validate_collection_not_empty(self, target: str) -> None:
        self.validated = target

    def switch_alias(self, alias: str, target: str) -> None:
        self.switched = (alias, target)

    def delete_collection_safe(self, target: str) -> None:
        self.deleted = target

    def ensure_ready(self) -> None:
        raise AssertionError("ensure_ready should not be called during query path")


def test_tenant_isolation_guard_and_metadata_filters_preserved() -> None:
    store = _StoreStub()
    retriever = TenantFilteredQdrantRetriever(store=store, tenant_id="tenant-a", k=4)
    docs = retriever.invoke(
        "q",
        metadata={
            "owner": "tenant-a",
            "tags": ["prod"],
            "act_type": "law",
            "issuer": "MoJ",
            "reg_number": "123",
            "is_active": True,
            "revision": "historical",
        },
    )
    assert docs and docs[0].metadata["tenant_id"] == "tenant-a"
    call = store.calls[-1]
    filters = call["filters"]
    assert filters.tenant_id == "tenant-a"
    assert filters.owner == "tenant-a"
    assert filters.tags == ("prod",)
    assert filters.act_type == "law"
    assert filters.issuer == "MoJ"
    assert filters.reg_number == "123"
    assert filters.is_active is True
    assert filters.revision_mode == "historical"


def test_query_path_does_not_force_reindex(monkeypatch) -> None:  # noqa: ANN001
    class _SearchStore:
        def __init__(self):
            self.ensure_calls = 0

        def ensure_ready(self):
            self.ensure_calls += 1

        def search(self, query, top_k, *, filters):  # noqa: ANN001
            return [{"text": query, "tenant_id": filters.tenant_id}]

    store = _SearchStore()
    monkeypatch.setattr(vectorstore_service, "_resolve_vector_store", lambda: store)
    hits = vectorstore_service.search("alpha", tenant_id="tenant-a")
    assert hits
    assert store.ensure_calls == 1


def test_atomic_alias_switch_pipeline(monkeypatch) -> None:  # noqa: ANN001
    store = _StoreStub()
    monkeypatch.setattr(vectorstore_service, "_resolve_vector_store", lambda: store)
    result = vectorstore_service.reindex_alias_atomic(document_id="doc-1")
    assert result["status"] == "completed"
    assert store.switched[0] == "kb"
    assert store.created[1] == "kb__v1"


def test_tenant_id_required() -> None:
    store = _StoreStub()
    try:
        TenantFilteredQdrantRetriever(store=store, tenant_id="   ")
    except ValueError as exc:
        assert "tenant_id is required" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ValueError expected")
