"""Test that page and has_original propagate through search/ask responses."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services import kb_llm
from app.services.kb_embeddings import HashingEmbedder
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeBaseStore:
    # Hashing embedder produces non-zero vectors → search returns results in tests
    return KnowledgeBaseStore(tmp_path / "test.sqlite", embedder=HashingEmbedder())


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
    assert any(
        s["page"] is not None for s in data["sources"]
    ), f"no page info in sources: {data['sources']}"
    assert all(s["has_original"] is True for s in data["sources"])


def test_legacy_text_document_has_null_page(app_with_store, store):
    """Documents added via legacy text= path should have page=null."""
    store.add_document("legacy", text="alpha beta " * 50)

    client = TestClient(app_with_store)
    resp = client.post("/api/kb/search", json={"query": "alpha"})
    assert resp.status_code == 200
    data = resp.json()
    for hit in data["hits"]:
        assert hit["page"] is None
        assert hit["has_original"] is False


def test_ask_stream_meta_has_page_and_has_original(app_with_store, store):
    """The SSE meta event's sources include page and has_original."""
    import json

    store.add_document(
        "doc1",
        pages=[(1, "alpha beta " * 50)],
        source="file",
        filename="x.pdf",
    )
    store.update_file_metadata(1, file_relpath="kb_files/1.pdf")

    client = TestClient(app_with_store)
    with client.stream("POST", "/api/kb/ask/stream", json={"question": "alpha"}) as resp:
        assert resp.status_code == 200
        # Read events; the meta event is the first one we care about
        body = b""
        for chunk in resp.iter_bytes():
            body += chunk
        text = body.decode("utf-8")

    # Parse SSE: find the line `event: meta` then the following `data: ...` line
    lines = text.splitlines()
    meta_data = None
    for i, line in enumerate(lines):
        if line.strip() == "event: meta" and i + 1 < len(lines):
            data_line = lines[i + 1]
            if data_line.startswith("data: "):
                meta_data = json.loads(data_line[len("data: ") :])
                break
    assert meta_data is not None, f"meta event not found in SSE stream:\n{text}"

    sources = meta_data.get("sources", [])
    assert sources, "expected sources in meta event"
    for src in sources:
        assert "page" in src, f"page missing from source: {src}"
        assert "has_original" in src, f"has_original missing from source: {src}"
    assert all(src["has_original"] is True for src in sources)
    assert any(src["page"] == 1 for src in sources)
