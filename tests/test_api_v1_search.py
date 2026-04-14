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
    ) -> list[dict[str, object]]:
        captured["query"] = query
        captured["top_k"] = top_k
        captured["owner"] = owner
        captured["tags"] = tags
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
    ) -> list[dict[str, object]]:
        captured["owner"] = owner
        captured["tags"] = tags
        return []

    monkeypatch.setattr(search_api, "search", _fake_search)

    response = search_api.search_endpoint(
        query="replication",
        owner="   ",
        tags=["", "  "],
        tenant="test-tenant",
    )
    assert response.hits == []
    assert captured == {"owner": None, "tags": None}
