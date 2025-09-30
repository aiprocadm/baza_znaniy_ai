"""Backward compatible wrappers for the Ollama provider."""

from __future__ import annotations

from app.core.config import get_settings
from app.llm.providers import OllamaProvider

_client = OllamaProvider(get_settings())
OLLAMA_BASE_URL = _client.base_url
MODEL_NAME = _client.model_name


def ensure_model() -> None:
    _client.ensure_model()


def generate(prompt: str) -> str:
    return _client.generate(prompt)


__all__ = ["ensure_model", "generate", "OLLAMA_BASE_URL", "MODEL_NAME"]
