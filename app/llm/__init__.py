"""Helpers for working with language model providers."""

from __future__ import annotations

from typing import Optional

from app.core.config import Settings, get_settings

from .exceptions import (
    LLMProviderError,
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from .llama_cpp_provider import LlamaCppProvider
from .providers import LLMProvider, get_llm_provider

__all__ = [
    "LLMProvider",
    "LlamaCppProvider",
    "get_llm_provider",
    "get_cached_provider",
    "reset_provider_cache",
    "LLMProviderError",
    "ModelNotFoundError",
    "ModelNotReadyError",
    "LoRAAdapterNotFoundError",
]

_cached_provider: Optional[LLMProvider] = None


def get_cached_provider(settings: Settings | None = None) -> LLMProvider:
    """Return a cached provider instance.

    When *settings* are supplied a fresh provider is constructed and cached.
    When omitted the previously cached provider (or a lazily created default)
    is returned.
    """

    global _cached_provider
    if settings is not None:
        _cached_provider = get_llm_provider(settings)
        return _cached_provider
    if _cached_provider is None:
        _cached_provider = get_llm_provider(get_settings())
    return _cached_provider


def reset_provider_cache() -> None:
    """Clear the cached provider (useful for tests)."""

    global _cached_provider
    _cached_provider = None


# Backwards compatible alias used by legacy modules/tests.
get_llm_client = get_cached_provider

__all__.append("get_llm_client")
