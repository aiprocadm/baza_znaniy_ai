"""Tests for the LLM provider factory and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from app.core.config import Settings
from app.llm import get_cached_provider, reset_provider_cache
from app.llm.llama_cpp_provider import LlamaCppProvider
from app.llm.providers import LLMProvider, get_llm_provider


def test_get_llm_provider_selects_llama_cpp(tmp_path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    settings = Settings(llm_provider="llama-cpp", llm_model_path=str(model_path))
    provider = get_llm_provider(settings)
    assert isinstance(provider, LlamaCppProvider)


def test_get_llm_provider_accepts_alias(tmp_path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    settings = Settings(llm_provider="llama_cpp", llm_model_path=str(model_path))
    provider = get_llm_provider(settings)
    assert isinstance(provider, LlamaCppProvider)


@dataclass
class _DummyProvider:
    name: str
    settings: Settings

    def ensure_model(self) -> None:  # pragma: no cover - behaviour not needed
        return None

    def ensure_ready(self) -> None:  # pragma: no cover - behaviour not needed
        return None

    def ensure_adapter(self) -> None:  # pragma: no cover - behaviour not needed
        return None

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        return f"dummy:{prompt}"


def test_get_cached_provider_caches_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_provider_cache()
    created: list[LLMProvider] = []
    settings = Settings(llm_provider="llama-cpp")

    def factory(_settings: Settings) -> LLMProvider:
        provider = _DummyProvider("dummy", _settings)
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
        provider = _DummyProvider(f"dummy-{settings.llm_model_name}", settings)
        created.append((settings.llm_model_name, provider))
        return provider

    monkeypatch.setattr("app.llm.get_llm_provider", factory)

    first_settings = Settings(llm_provider="llama-cpp", llm_model_name="model-a")
    second_settings = Settings(llm_provider="llama-cpp", llm_model_name="model-b")

    first = get_cached_provider(first_settings)
    second = get_cached_provider(second_settings)

    assert first is not second
    assert created == [
        ("model-a", first),
        ("model-b", second),
    ]
