"""Backward compatible wrappers for the Ollama client."""

from __future__ import annotations

from app.llm import get_llm_client

_client = get_llm_client()
OLLAMA_BASE_URL = _client.base_url
MODEL_NAME = _client.model_name


def ensure_model() -> None:
    _client.ensure_model()


def generate(prompt: str) -> str:
    return _client.generate(prompt)


__all__ = ["ensure_model", "generate", "OLLAMA_BASE_URL", "MODEL_NAME"]
