from __future__ import annotations

import importlib


def test_docling_backend_falls_back_when_disabled(monkeypatch):
    monkeypatch.setenv("DOCUMENT_PARSER_BACKEND", "docling")
    monkeypatch.setenv("DOCLING_ENABLED", "false")
    from app.ingest import chunking

    importlib.reload(chunking)
    assert chunking._resolve_parser_backend() == "legacy"


def test_docling_backend_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("DOCUMENT_PARSER_BACKEND", "unexpected")
    from app.ingest import chunking

    importlib.reload(chunking)
    assert chunking._resolve_parser_backend() == "legacy"
