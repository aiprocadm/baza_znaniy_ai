"""Test per-page chunking and Document/SearchHit page_number propagation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeBaseStore:
    return KnowledgeBaseStore(tmp_path / "test.sqlite")


def test_legacy_text_path_still_works(store: KnowledgeBaseStore) -> None:
    """add_document(title, text=...) keeps old behaviour — page_number is NULL."""
    doc = store.add_document("Title", text="hello world " * 130)
    assert doc.id > 0
    assert doc.has_original_file is False
    assert doc.file_relpath is None
    # search → SearchHit.page is None for legacy chunks
    hits = store.search("hello")
    assert hits
    assert all(hit.page is None for hit in hits)
    assert all(hit.has_original is False for hit in hits)


def test_pages_path_assigns_page_per_chunk(store: KnowledgeBaseStore) -> None:
    """add_document(pages=[...]) chunks per page and preserves page_number."""
    pages = [
        (1, "page one " * 200),  # large enough to produce 1+ chunks
        (2, "page two " * 200),
        (3, "page three " * 200),
    ]
    doc = store.add_document("Pages", pages=pages, source="file", filename="x.pdf")
    assert doc.id > 0
    # Force search returns hits per page
    hits = store.search("page one", top_k=10)
    assert hits
    pages_found = {hit.page for hit in hits if "page one" in hit.text}
    assert pages_found == {1}, f"expected page 1 only, got {pages_found}"


def test_update_file_metadata_marks_has_original(store: KnowledgeBaseStore) -> None:
    """update_file_metadata flips has_original_file and stores relpath."""
    doc = store.add_document("doc", text="hi")
    store.update_file_metadata(doc.id, file_relpath="kb_files/1.pdf")
    refreshed = store.get_document(doc.id)
    assert refreshed is not None
    assert refreshed.has_original_file is True
    assert refreshed.file_relpath == "kb_files/1.pdf"


def test_search_includes_has_original_flag(store: KnowledgeBaseStore) -> None:
    """Searched chunks of a doc with original file get has_original=True."""
    doc = store.add_document(
        "doc",
        pages=[(1, "alpha beta " * 50)],
        source="file",
        filename="x.pdf",
    )
    store.update_file_metadata(doc.id, file_relpath="kb_files/1.pdf")
    hits = store.search("alpha", top_k=5)
    assert hits
    assert all(hit.has_original for hit in hits)
    assert all(hit.page == 1 for hit in hits)


def test_add_document_pages_and_text_raise(store: KnowledgeBaseStore) -> None:
    """Passing both text and pages is ambiguous → raise."""
    with pytest.raises(ValueError):
        store.add_document("x", text="foo", pages=[(1, "bar")])


def test_add_document_empty_pages_raises(store: KnowledgeBaseStore) -> None:
    """All-empty pages → ValueError(Text is empty)."""
    with pytest.raises(ValueError):
        store.add_document("x", pages=[(1, ""), (2, "  ")])


def test_add_document_neither_text_nor_pages_raises(store: KnowledgeBaseStore) -> None:
    """Passing neither text= nor pages= → ValueError with explicit message."""
    with pytest.raises(ValueError, match="Pass either text= or pages="):
        store.add_document("x")
