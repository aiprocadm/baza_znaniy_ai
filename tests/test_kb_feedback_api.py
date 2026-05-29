"""Endpoint tests for /api/kb/messages/{id}/feedback and /api/kb/feedback/export."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_with_store(tmp_path, monkeypatch):
    """Build a FastAPI app whose KnowledgeBaseStore points at tmp_path."""

    monkeypatch.setenv("KB_API_KEY", "test-key")
    monkeypatch.setenv("KB_MVP_DB_PATH", str(tmp_path / "kb.sqlite"))

    # Reset the cached default store so the new env var is picked up.
    import app.services.kb_store as kb_store_mod

    kb_store_mod._DEFAULT_STORE = None

    from app.services.kb_store import get_store

    store = get_store()
    conv = store.create_conversation(title="test")
    store.add_message(conversation_id=conv.id, role="user", content="hello?")
    asst = store.add_message(conversation_id=conv.id, role="assistant", content="hi back")

    from app.core.app import create_app

    app = create_app()
    client = TestClient(app)
    return client, conv.id, asst.id


def test_post_feedback_persists_and_returns_id(app_with_store) -> None:
    client, _conv, asst = app_with_store
    r = client.post(
        f"/api/kb/messages/{asst}/feedback",
        json={"rating": 1, "comment": "ok"},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert "id" in body
    assert "created_at" in body


def test_post_feedback_rejects_invalid_rating(app_with_store) -> None:
    client, _conv, asst = app_with_store
    r = client.post(
        f"/api/kb/messages/{asst}/feedback",
        json={"rating": 5},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 400, r.text


def test_post_feedback_requires_auth(app_with_store) -> None:
    client, _conv, asst = app_with_store
    r = client.post(f"/api/kb/messages/{asst}/feedback", json={"rating": 1})
    assert r.status_code == 401


def test_post_feedback_unknown_message_returns_404(app_with_store) -> None:
    client, _conv, _asst = app_with_store
    r = client.post(
        "/api/kb/messages/99999/feedback",
        json={"rating": 1},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 404


def test_export_returns_ndjson(app_with_store) -> None:
    client, _conv, asst = app_with_store
    r = client.post(
        f"/api/kb/messages/{asst}/feedback",
        json={"rating": -1, "alternative_answer": "лучше"},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 201

    r = client.get(
        "/api/kb/feedback/export",
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert "application/x-ndjson" in r.headers["content-type"]
    lines = [line for line in r.text.splitlines() if line.strip()]
    assert len(lines) == 1
    import json as _json

    pair = _json.loads(lines[0])
    assert pair["chosen"] == "лучше"


def test_export_empty_returns_200_with_zero_lines(app_with_store) -> None:
    client, _conv, _asst = app_with_store
    r = client.get(
        "/api/kb/feedback/export",
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert r.text.strip() == ""
