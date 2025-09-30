"""Tests for selecting and caching LLM providers."""

from __future__ import annotations

from types import SimpleNamespace

from app.llm import (
    OllamaProvider,
    StubProvider,
    get_llm_provider,
    reset_llm_provider_cache,
)


def _settings(provider: str, base_url: str = "http://ollama", model: str = "model") -> SimpleNamespace:
    return SimpleNamespace(
        llm_provider=provider,
        ollama_base_url=base_url,
        llm_model_name=model,
        max_context_tokens=2048,
    )


def test_get_llm_provider_returns_stub() -> None:
    reset_llm_provider_cache()
    settings = _settings("stub")

    provider = get_llm_provider(settings)

    assert isinstance(provider, StubProvider)
    assert get_llm_provider(settings) is provider  # cache reuse


def test_get_llm_provider_returns_ollama() -> None:
    reset_llm_provider_cache()
    settings = _settings("ollama", base_url="http://ollama.local", model="llama")

    provider = get_llm_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://ollama.local"
    assert provider.model_name == "llama"
