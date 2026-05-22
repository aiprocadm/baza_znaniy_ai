"""Test that page and has_original propagate through search/ask responses."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services import kb_llm
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeBaseStore:
    return KnowledgeBaseStore(tmp_path / "test.sqlite")


@pytest.fixture
def app_with_store(store: KnowledgeBaseStore, monkeypatch):
    """Build a minimal FastAPI app with the MVP router and pinned store."""
    from fastapi import FastAPI
    from app.api.kb_mvp import router

    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    # Force extractive answer to avoid hitting any real LLM
    monkeypatch.setattr(kb_llm, "select_provider", lambda: None)
    return fastapi_app


def test_search_response_has_page_and_has_original(app_with_store, store):
    store.add_document(
        "doc1",
        pages=[(1, "alpha beta " * 50)],
        source="file",
        filename="x.pdf",
    )
    store.update_file_metadata(1, file_relpath="kb_files/1.pdf")

    client = TestClient(app_with_store)
    resp = client.post("/api/kb/search", json={"query": "alpha"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["hits"], "expected hits"
    for hit in data["hits"]:
        assert hit["page"] == 1
        assert hit["has_original"] is True


def test_ask_response_has_page_and_has_original(app_with_store, store):
    store.add_document(
        "doc1",
        pages=[(1, "alpha beta " * 50), (2, "gamma delta " * 50)],
        source="file",
        filename="x.pdf",
    )
    store.update_file_metadata(1, file_relpath="kb_files/1.pdf")

    client = TestClient(app_with_store)
    resp = client.post("/api/kb/ask", json={"question": "alpha"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sources"], "expected sources"
    # At least one source should match page 1
    pages = {s["page"] for s in data["sources"]}
    assert 1 in pages or 2 in pages, f"no page info in sources: {data['sources']}"
    assert all(s["has_original"] is True for s in data["sources"])


def test_legacy_text_document_has_null_page(app_with_store, store):
    """Documents added via legacy text= path should have page=null."""
    store.add_document("legacy", text="alpha beta " * 50)

    client = TestClient(app_with_store)
    resp = client.post("/api/kb/search", json={"query": "alpha"})
    data = resp.json()
    for hit in data["hits"]:
        assert hit["page"] is None
        assert hit["has_original"] is False
