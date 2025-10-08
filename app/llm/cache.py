"""Helpers for caching and reusing LLM provider instances."""

from __future__ import annotations

import sys
from importlib import import_module
from typing import TYPE_CHECKING, Callable, Optional, cast

from app.core.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from app.core.config import Settings as SettingsType
else:  # pragma: no cover - fallback type used at runtime
    from typing import Any as SettingsType

from .exceptions import (
    LLMProviderError,
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from .llama_cpp_provider import LlamaCppProvider
from .providers import LLMProvider

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
    "set_provider_factory_override",
]

_cached_provider: Optional[LLMProvider] = None
_factory_override: Optional[Callable[[SettingsType], LLMProvider]] = None


def _get_settings() -> SettingsType:
    """Return configured settings, deferring the import until runtime."""

    settings = get_settings()
    return cast(SettingsType, settings)


def set_provider_factory_override(
    factory: Callable[[SettingsType], LLMProvider] | None,
) -> None:
    """Set or clear an explicit factory override (primarily for tests)."""

    global _factory_override
    _factory_override = factory


def _resolve_factory() -> Callable[[SettingsType], LLMProvider]:
    """Return the currently active ``get_llm_provider`` implementation."""

    if _factory_override is not None:
        return _factory_override

    providers_module = sys.modules.get("app.llm.providers")
    if providers_module is None:
        providers_module = import_module("app.llm.providers")
    provider_factory = getattr(providers_module, "get_llm_provider", None)

    package = sys.modules.get("app.llm")
    if package is None:
        package = import_module("app.llm")
    candidate = getattr(package, "get_llm_provider", None) if package else None
    candidate_module = getattr(candidate, "__module__", "")
    provider_module_name = getattr(providers_module, "__name__", "")
    if callable(candidate):
        stale_from_cache = candidate_module == __name__ and candidate is get_llm_provider
        stale_from_provider = (
            candidate_module == provider_module_name and candidate is not provider_factory
        )
        if not stale_from_cache and not stale_from_provider:
            return candidate  # type: ignore[return-value]

    if getattr(providers_module, "__file__", None) is None:
        if callable(provider_factory):
            return provider_factory  # type: ignore[return-value]

    if not callable(provider_factory):  # pragma: no cover - defensive guard
        raise RuntimeError("LLM provider factory is not callable")
    return provider_factory  # type: ignore[return-value]


def get_llm_provider(settings: SettingsType) -> LLMProvider:
    """Expose the default provider factory used by the cache module."""

    return _resolve_factory()(settings)


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
    global _factory_override
    _cached_provider = None
    _factory_override = None


# Backwards-compatible alias expected by parts of the application.
get_llm_client = get_cached_provider
