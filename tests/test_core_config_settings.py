"""Tests for new configuration fields and helpers."""

from __future__ import annotations

from typing import Iterable

import pytest

from app.core.config import Settings


def _clear_env(monkeypatch: pytest.MonkeyPatch, names: Iterable[str]) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_settings_new_fields_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "EMBED_BATCH_SIZE",
        "RERANK_ENABLED",
        "RERANK_TOPK",
        "RETRIEVE_TOPK",
        "OLLAMA_MODEL",
        "LLM_MODEL_NAME",
        "MAX_CONTEXT_TOKENS",
    ]
    _clear_env(monkeypatch, keys)

    settings = Settings()

    assert settings.embed_batch_size == 32
    assert settings.rerank_enabled is True
    assert settings.rerank_limit == settings.retrieve_topk
    assert settings.should_rerank is True
    assert settings.max_context_tokens == 3000
    assert settings.llm_model == settings.llm_model_name


def test_settings_parses_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_BATCH_SIZE", "0")
    monkeypatch.setenv("RETRIEVE_TOPK", "6")
    monkeypatch.setenv("RERANK_TOPK", "3")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("OLLAMA_MODEL", "  mistral ")
    monkeypatch.setenv("MAX_CONTEXT_TOKENS", "-10")

    settings = Settings()

    assert settings.embed_batch_size == 1
    assert settings.rerank_enabled is False
    assert settings.rerank_limit == 6
    assert settings.should_rerank is False
    assert settings.llm_model == "mistral"
    assert settings.max_context_tokens == 1


def test_settings_falls_back_to_llm_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "   ")
    monkeypatch.setenv("LLM_MODEL_NAME", "granite-8b")

    settings = Settings()

    assert settings.ollama_model is None
    assert settings.llm_model == "granite-8b"


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    yield
    for name in [
        "EMBED_BATCH_SIZE",
        "RERANK_ENABLED",
        "RERANK_TOPK",
        "RETRIEVE_TOPK",
        "OLLAMA_MODEL",
        "LLM_MODEL_NAME",
        "MAX_CONTEXT_TOKENS",
    ]:
        monkeypatch.delenv(name, raising=False)
