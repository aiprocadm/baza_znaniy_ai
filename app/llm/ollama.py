"""Backward compatible wrappers for the Ollama LLM provider."""

from __future__ import annotations

from app.core.config import Settings
from app.llm.providers import OllamaProvider


class OllamaClient(OllamaProvider):
    """Compatibility alias for legacy imports."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings=settings)


def get_llm_client(settings: Settings | None = None) -> OllamaClient:
    """Return an :class:`OllamaClient` instance."""

    return OllamaClient(settings=settings)


__all__ = ["OllamaClient", "get_llm_client"]

