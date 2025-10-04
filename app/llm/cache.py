"""Helpers for caching and reusing LLM provider instances."""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from app.core.config import Settings as SettingsType
else:
    SettingsType = Any

from .exceptions import (
    LLMProviderError,
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from .llama_cpp_provider import LlamaCppProvider
from .providers import LLMProvider, get_llm_provider as _providers_get_llm_provider

__all__ = [
    "LLMProvider",
    "LlamaCppProvider",
    "LLMProviderError",
    "LoRAAdapterNotFoundError",
    "ModelNotFoundError",
    "ModelNotReadyError",
    "get_cached_provider",
    "get_llm_client",
    "get_llm_provider",
    "reset_provider_cache",
]

_cached_provider: Optional[LLMProvider] = None


def _get_settings() -> SettingsType:
    """Return configured settings, deferring the import until runtime."""

    from app.core.config import get_settings as _get_settings  # local import for flexibility

    settings = _get_settings()
    return cast(SettingsType, settings)


def _resolve_factory() -> Callable[[SettingsType], LLMProvider]:
    """Return the currently active ``get_llm_provider`` implementation."""

    package = sys.modules.get("app.llm")
    candidate = getattr(package, "get_llm_provider", None) if package else None
    if callable(candidate) and candidate is not get_llm_provider:
        return candidate  # type: ignore[return-value]
    return _providers_get_llm_provider


def get_llm_provider(settings: SettingsType) -> LLMProvider:
    """Expose the default provider factory used by the cache module."""

    return _providers_get_llm_provider(settings)


def get_cached_provider(settings: SettingsType | None = None) -> LLMProvider:
    """Return a cached provider instance, creating one on demand."""

    global _cached_provider
    if settings is not None:
        _cached_provider = _resolve_factory()(settings)
        return _cached_provider
    if _cached_provider is None:
        _cached_provider = _resolve_factory()(_get_settings())
    return _cached_provider


def reset_provider_cache() -> None:
    """Clear the cached provider instance (useful for tests)."""

    global _cached_provider
    _cached_provider = None


# Backwards-compatible alias expected by parts of the application.
get_llm_client = get_cached_provider
