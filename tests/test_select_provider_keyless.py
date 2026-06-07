"""Keyless auto-selection: local GGUF is the DEFAULT fallback (opt-out, not opt-in)."""
from __future__ import annotations

import app.services.kb_llm as kb_llm


def test_gguf_used_by_default_when_no_keys(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: sentinel)
    assert kb_llm.select_provider(env={}) is sentinel  # no env at all → GGUF


def test_gguf_disabled_explicitly(monkeypatch):
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: object())
    assert kb_llm.select_provider(env={"KB_LLM_LOCAL_FALLBACK": "0"}) is None


def test_none_when_no_keys_and_no_model(monkeypatch):
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: None)
    assert kb_llm.select_provider(env={}) is None


def test_external_key_wins_over_gguf(monkeypatch):
    called = {"gguf": False}

    def _spy(env=None):
        called["gguf"] = True
        return object()

    monkeypatch.setattr(kb_llm, "_build_gguf_provider", _spy)
    provider = kb_llm.select_provider(env={"DEEPSEEK_API_KEY": "sk-test"})
    assert provider is not None and provider.name == "deepseek"
    assert called["gguf"] is False


def test_explicit_gguf_still_works(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(kb_llm, "_build_gguf_provider", lambda env=None: sentinel)
    assert kb_llm.select_provider(env={"KB_LLM_PROVIDER": "gguf"}) is sentinel
