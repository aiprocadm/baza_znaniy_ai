"""Embedder signature guard: hard-stop on index/embedder mismatch."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.services.embedder_signature import (
    EmbedderMismatchError,
    signature_for,
    verify_or_store,
)
from app.services.kb_embeddings import HashingEmbedder
from app.services.kb_store import KnowledgeBaseStore


class _FakeEmbedder:
    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dimension = dim

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dimension


def test_signature_format():
    assert signature_for(_FakeEmbedder("e5-small", 384)) == "e5-small:384"


def test_fresh_index_stores_and_passes():
    store: dict[str, str] = {}
    verify_or_store(
        _FakeEmbedder("e5-small", 384), load=store.get, save=lambda s: store.__setitem__("sig", s)
    )
    assert store["sig"] == "e5-small:384"


def test_matching_signature_passes():
    store = {"sig": "e5-small:384"}
    verify_or_store(
        _FakeEmbedder("e5-small", 384), load=store.get, save=lambda s: store.__setitem__("sig", s)
    )  # no raise


def test_mismatch_raises_with_instructions():
    store = {"sig": "hash:256"}
    with pytest.raises(EmbedderMismatchError) as exc:
        verify_or_store(_FakeEmbedder("e5-small", 384), load=store.get, save=lambda s: None)
    msg = str(exc.value)
    assert "reindex" in msg.lower()
    assert "hash:256" in msg and "e5-small:384" in msg


# ---------------------------------------------------------------------------
# Store-level upgrade-path tests (real SQLite, tmp file DB)
# ---------------------------------------------------------------------------


def _drop_sig_row(db_path: Path) -> None:
    """Delete the kv_meta sig row to simulate a pre-kv_meta DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM kv_meta WHERE key = 'sig'")
        conn.commit()
    finally:
        conn.close()


def test_fresh_empty_db_stores_active_sig(tmp_path: Path) -> None:
    """Opening an empty DB (no chunks) stores the active embedder sig and passes."""
    db_path = tmp_path / "fresh.sqlite"
    embedder = HashingEmbedder()
    store = KnowledgeBaseStore(db_path, embedder=embedder)
    # Sig must have been written
    sig = store._kv_load("sig")
    assert sig == f"{embedder.name}:{embedder.dimension}"


def test_same_embedder_reopened_passes(tmp_path: Path) -> None:
    """Reopening the same DB with the same embedder succeeds without error."""
    db_path = tmp_path / "same.sqlite"
    embedder = HashingEmbedder()
    store = KnowledgeBaseStore(db_path, embedder=embedder)
    store.add_document("doc", text="hello world")
    # Second open with identical embedder — must not raise
    KnowledgeBaseStore(db_path, embedder=HashingEmbedder())


def test_upgrade_path_populated_db_no_sig_different_embedder_raises(tmp_path: Path) -> None:
    """Populated pre-kv_meta DB opened with a different-dimension embedder raises.

    Simulates the protection gap: a DB built with hash:256 has its sig row
    deleted (pre-kv_meta upgrade scenario), then is opened with an e5-style
    embedder (dim=384).  The guard must detect the mismatch and raise.
    """
    db_path = tmp_path / "legacy_populated.sqlite"
    hash_embedder = HashingEmbedder()  # name="hash", dimension=256

    # Step 1: build and populate the DB with the hashing embedder
    store = KnowledgeBaseStore(db_path, embedder=hash_embedder)
    store.add_document("doc", text="hello world, this is test content for the index")
    assert store._kv_load("sig") == "hash:256"

    # Step 2: drop the sig row to simulate a pre-kv_meta DB
    _drop_sig_row(db_path)

    # Verify the sig is gone but chunks remain
    conn = sqlite3.connect(str(db_path))
    try:
        sig_row = conn.execute("SELECT value FROM kv_meta WHERE key='sig'").fetchone()
        chunk_count = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
    finally:
        conn.close()
    assert sig_row is None, "sig row should be absent after deletion"
    assert chunk_count > 0, "chunks must exist to exercise the upgrade path"

    # Step 3: reopen with a different-dimension embedder — must raise EmbedderMismatchError
    different_embedder = _FakeEmbedder("e5-small", 384)
    with pytest.raises(EmbedderMismatchError) as exc:
        KnowledgeBaseStore(db_path, embedder=different_embedder)

    msg = str(exc.value)
    assert "hash:256" in msg, f"expected legacy sig in error message, got: {msg}"
    assert "e5-small:384" in msg, f"expected active sig in error message, got: {msg}"
    assert "reindex" in msg.lower()


def test_upgrade_path_populated_db_no_sig_same_embedder_passes(tmp_path: Path) -> None:
    """Populated pre-kv_meta DB opened with the SAME embedder type passes.

    If the stored chunks were built with hash:256 and we reopen with hash:256,
    the backfilled sig matches the active sig — no error.
    """
    db_path = tmp_path / "legacy_same.sqlite"
    hash_embedder = HashingEmbedder()  # name="hash", dimension=256

    store = KnowledgeBaseStore(db_path, embedder=hash_embedder)
    store.add_document("doc", text="content to populate the index")

    # Drop the sig to simulate pre-kv_meta
    _drop_sig_row(db_path)

    # Reopen with the same embedder — must not raise
    store2 = KnowledgeBaseStore(db_path, embedder=HashingEmbedder())
    # Sig should be backfilled and match
    sig = store2._kv_load("sig")
    assert sig == "hash:256"
