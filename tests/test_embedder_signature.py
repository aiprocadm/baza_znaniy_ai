"""Embedder signature guard: hard-stop on index/embedder mismatch."""

from __future__ import annotations

import pytest

from app.services.embedder_signature import (
    EmbedderMismatchError,
    signature_for,
    verify_or_store,
)


class _FakeEmbedder:
    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dimension = dim


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
