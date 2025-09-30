"""Tests for the LLM provider factory and helpers."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.llm import get_cached_provider, reset_provider_cache
from app.llm.providers import LLMProvider, OllamaProvider, StubProvider, get_llm_provider


def test_get_llm_provider_selects_stub() -> None:
    settings = Settings(llm_provider="stub")
    provider = get_llm_provider(settings)
    assert isinstance(provider, StubProvider)


def test_get_llm_provider_selects_ollama() -> None:
    settings = Settings(llm_provider="ollama")
    provider = get_llm_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_stub_provider_appends_citations() -> None:
    provider = StubProvider(Settings(llm_provider="stub"))
    prompt = "Привет"
    citations = [{"file": "doc.md", "page": 2}, {"file": "info.txt"}]
    answer = provider.generate(prompt, context={"citations": citations})
    assert "Привет!" in answer
    assert "Источники" in answer
    assert "[1] doc.md — страница 2" in answer
    assert "[2] info.txt" in answer


def test_get_cached_provider_caches_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_provider_cache()
    created: list[LLMProvider] = []
    settings = Settings(llm_provider="stub")

    def factory(_settings: Settings) -> LLMProvider:
        provider = StubProvider(_settings)
        created.append(provider)
        return provider

    monkeypatch.setattr("app.llm.get_llm_provider", factory)

    first = get_cached_provider(settings)
    second = get_cached_provider()

    assert first is second
    assert len(created) == 1


def test_get_cached_provider_updates_when_settings_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_provider_cache()
    created: list[Any] = []

    def factory(settings: Settings) -> LLMProvider:
        provider = StubProvider(settings)
        created.append((settings.llm_model_name, provider))
        return provider

    monkeypatch.setattr("app.llm.get_llm_provider", factory)

    first_settings = Settings(llm_provider="stub", llm_model_name="model-a")
    second_settings = Settings(llm_provider="stub", llm_model_name="model-b")

    first = get_cached_provider(first_settings)
    second = get_cached_provider(second_settings)

    assert first is not second
    assert created == [
        ("model-a", first),
        ("model-b", second),
    ]
