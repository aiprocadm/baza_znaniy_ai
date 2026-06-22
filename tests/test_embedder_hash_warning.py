"""Characterization test: the hashing-embedder fallback warns loudly when a
production-like config (KB_API_KEY set, no real embedder backend) would silently
degrade semantic search to near-random results. Pins existing behaviour in
app/services/kb_embeddings.py:_build_from_env — do not let it regress silently.

The sentence_transformers stub in tests/stubs/ accepts any model path without
raising, so we cannot force the ST probe to fail via an env path trick. Instead
we patch _try_build_st_embedder to return None, which is what the real function
does when the dependency or weights are missing — the exact condition the warning
is designed to catch."""

from __future__ import annotations

import logging
from unittest.mock import patch

import app.services.kb_embeddings as emb
from app.services.kb_embeddings import _build_from_env

_PATCH = "app.services.kb_embeddings._try_build_st_embedder"


def setup_function(_):
    emb.reset_embedder()


def teardown_function(_):
    emb.reset_embedder()


def test_hash_fallback_warns_when_api_key_set(caplog):
    env = {"KB_API_KEY": "secret", "KB_EMBEDDINGS_BACKEND": ""}
    with patch(_PATCH, return_value=None):
        with caplog.at_level(logging.WARNING, logger="app.services.kb_embeddings"):
            embedder = _build_from_env(env)
    assert embedder.name == "hash"
    assert embedder.dimension == 256
    assert any(
        "hashing embedder" in rec.getMessage() and "near-random" in rec.getMessage()
        for rec in caplog.records
    ), "expected a loud WARNING about the hashing fallback"


def test_hash_fallback_silent_without_api_key(caplog):
    env = {"KB_EMBEDDINGS_BACKEND": ""}
    with patch(_PATCH, return_value=None):
        with caplog.at_level(logging.WARNING, logger="app.services.kb_embeddings"):
            embedder = _build_from_env(env)
    assert embedder.name == "hash"
    assert not any(
        "hashing embedder" in rec.getMessage() for rec in caplog.records
    ), "no KB_API_KEY → no warning noise"
