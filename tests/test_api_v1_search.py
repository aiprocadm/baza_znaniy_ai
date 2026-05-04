from __future__ import annotations

import app.api.v1.search as search_api


def test_search_endpoint_passes_owner_and_tags_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_search(
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
        tenant_id: str | None = None,
    ) -> list[dict[str, object]]:
        captured["query"] = query
        captured["top_k"] = top_k
        captured["owner"] = owner
        captured["tags"] = tags
        captured["tenant_id"] = tenant_id
        captured["act_type"] = act_type
        captured["issuer"] = issuer
        captured["reg_number"] = reg_number
        captured["is_active"] = is_active
        captured["revision_mode"] = revision_mode
        return [{"file": "doc.md", "page": 1, "score": 0.9, "text": "result"}]

    monkeypatch.setattr(search_api, "search", _fake_search)

    response = search_api.search_endpoint(
        query="replication",
        top_k=3,
        owner="alice@kb.ai",
        tags=["prod", "runbook"],
        tenant="test-tenant",
    )
    assert captured == {
        "query": "replication",
        "top_k": 3,
        "owner": "alice@kb.ai",
        "tags": ["prod", "runbook"],
        "tenant_id": "test-tenant",
        "act_type": None,
        "issuer": None,
        "reg_number": None,
        "is_active": None,
        "revision_mode": "current",
    }
    assert response.hits[0].file == "doc.md"


def test_search_endpoint_normalizes_empty_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_search(
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
        tenant_id: str | None = None,
    ) -> list[dict[str, object]]:
        captured["owner"] = owner
        captured["tags"] = tags
        captured["tenant_id"] = tenant_id
        captured["act_type"] = act_type
        captured["issuer"] = issuer
        captured["reg_number"] = reg_number
        captured["is_active"] = is_active
        captured["revision_mode"] = revision_mode
        return []

    monkeypatch.setattr(search_api, "search", _fake_search)

    response = search_api.search_endpoint(
        query="replication",
        owner="   ",
        tags=["", "  "],
        tenant="test-tenant",
    )
    assert response.hits == []
    assert captured == {"owner": None, "tags": None, "tenant_id": "test-tenant", "act_type": None, "issuer": None, "reg_number": None, "is_active": None, "revision_mode": "current"}


def test_search_endpoint_passes_all_npa_filters(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_search(
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
        tenant_id: str | None = None,
    ) -> list[dict[str, object]]:
        captured.update(
            {
                "query": query,
                "top_k": top_k,
                "owner": owner,
                "tags": tags,
                "act_type": act_type,
                "issuer": issuer,
                "reg_number": reg_number,
                "is_active": is_active,
                "revision_mode": revision_mode,
                "tenant_id": tenant_id,
            }
        )
        return []

    monkeypatch.setattr(search_api, "search", _fake_search)

    search_api.search_endpoint(
        query="law 123",
        top_k=7,
        owner="alice@kb.ai",
        tags=["prod"],
        act_type="law",
        issuer="Минюст",
        reg_number="123-ФЗ",
        is_active=False,
        revision_mode="historical",
        tenant="test-tenant",
    )
    assert captured == {
        "query": "law 123",
        "top_k": 7,
        "owner": "alice@kb.ai",
        "tags": ["prod"],
        "act_type": "law",
        "issuer": "Минюст",
        "reg_number": "123-ФЗ",
        "is_active": False,
        "revision_mode": "historical",
        "tenant_id": "test-tenant",
    }



def test_search_endpoint_requires_tenant_context(monkeypatch) -> None:
    def _fake_search(*args, **kwargs):
        raise AssertionError("search should not be called")

    monkeypatch.setattr(search_api, "search", _fake_search)

    import pytest

    with pytest.raises(ValueError, match="tenant context is required"):
        search_api.search_endpoint(query="replication", tenant="   ")
