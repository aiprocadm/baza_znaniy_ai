"""Tests for app.services.rag_dataset — pure-logic RAG dataset builder."""

from __future__ import annotations


def test_module_imports() -> None:
    """Module imports without side effects."""
    from app.services import rag_dataset

    assert rag_dataset.__name__ == "app.services.rag_dataset"
