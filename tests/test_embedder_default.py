"""Default embedder is the real ST/e5 model, not hashing, when available."""
from __future__ import annotations

import app.services.kb_embeddings as emb
from app.retriever.e5 import e5_prefix


def setup_function(_):
    emb.reset_embedder()


def teardown_function(_):
    emb.reset_embedder()


def test_default_is_st_when_available(monkeypatch):
    sentinel = emb.HashingEmbedder()  # any Embedder; identity is what we assert
    monkeypatch.setattr(emb, "_try_build_st_embedder", lambda env: sentinel, raising=False)
    chosen = emb._build_from_env(env={})
    assert chosen is sentinel


def test_falls_back_to_hash_when_st_unavailable(monkeypatch):
    monkeypatch.setattr(emb, "_try_build_st_embedder", lambda env: None, raising=False)
    chosen = emb._build_from_env(env={})
    assert isinstance(chosen, emb.HashingEmbedder)


def test_explicit_hash_skips_st(monkeypatch):
    called = {"st": False}

    def _spy(env):
        called["st"] = True
        return object()

    monkeypatch.setattr(emb, "_try_build_st_embedder", _spy, raising=False)
    chosen = emb._build_from_env(env={"KB_EMBEDDINGS_BACKEND": "hash"})
    assert isinstance(chosen, emb.HashingEmbedder)
    assert called["st"] is False


def test_e5_prefix_query_vs_passage():
    assert e5_prefix("foo", role="query", model="multilingual-e5-small", enabled=True) == "query: foo"
    assert e5_prefix("foo", role="passage", model="multilingual-e5-small", enabled=True) == "passage: foo"
    # "bge-m3" does not contain "e5", so prefix is not applied even with enabled=True
    assert e5_prefix("foo", role="query", model="BAAI/bge-m3", enabled=True) == "foo"
