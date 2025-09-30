"""Backward compatible wrappers for the configured LLM provider."""

from __future__ import annotations

from app.llm import LLMProviderProtocol, get_llm_provider

_provider: LLMProviderProtocol = get_llm_provider()
OLLAMA_BASE_URL = getattr(_provider, "base_url", "")
MODEL_NAME = getattr(_provider, "model_name", "")


def ensure_model() -> None:
    _provider.ensure_model()


def generate(prompt: str, context: str | None = None) -> str:
    return _provider.generate(prompt, context=context)


__all__ = ["ensure_model", "generate", "OLLAMA_BASE_URL", "MODEL_NAME"]
