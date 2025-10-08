"""Caching helpers for language model providers."""
from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from app.core.config import Settings as SettingsType
else:  # pragma: no cover - runtime fallback when ``Settings`` is absent
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
_external_factory: Callable[[SettingsType], LLMProvider] | None = None
_DEFAULT_MARKER = "_llm_cache_default"


def _get_settings() -> SettingsType:
    """Return configured settings, deferring the import until runtime."""

    from app.core.config import get_settings as _config_get_settings

    return cast(SettingsType, _config_get_settings())


def _resolve_factory() -> Callable[[SettingsType], LLMProvider]:
    """Return the currently active ``get_llm_provider`` implementation."""

    if _external_factory is not None:
        return _external_factory

    package = sys.modules.get("app.llm")
    candidate = getattr(package, "get_llm_provider", None) if package else None
    if _is_external_factory(candidate):
        _sync_external_factory(candidate)
        return candidate  # type: ignore[return-value]

    try:
        module = importlib.import_module("app.llm")
    except ModuleNotFoundError:  # pragma: no cover - defensive guard
        module = package  # type: ignore[assignment]

    _register_llm_module(module)

    candidate = getattr(module, "get_llm_provider", None) if module else None
    if _is_external_factory(candidate):
        return candidate  # type: ignore[return-value]
    return _providers_get_llm_provider


def get_llm_provider(settings: SettingsType) -> LLMProvider:
    """Expose the default provider factory used by the cache module."""

    return _providers_get_llm_provider(settings)


setattr(get_llm_provider, _DEFAULT_MARKER, True)


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

    global _cached_provider, _external_factory
    _cached_provider = None
    _external_factory = None
    try:  # pragma: no cover - defensive import to restore package state
        module = importlib.import_module("app.llm")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        _register_llm_module(None)
    else:
        _register_llm_module(module)


# Backwards-compatible alias expected by parts of the application.
get_llm_client = get_cached_provider


def _is_external_factory(candidate: object) -> bool:
    return callable(candidate) and not getattr(candidate, _DEFAULT_MARKER, False) and getattr(candidate, "__module__", None) != __name__


def _sync_external_factory(candidate: object) -> None:
    """Update the cached external factory based on *candidate*."""

    global _external_factory
    if _is_external_factory(candidate):
        _external_factory = candidate  # type: ignore[assignment]
    else:
        _external_factory = None


def _register_llm_module(module: object | None) -> None:
    """Record overrides exported via the ``app.llm`` package."""

    if module is None:
        _sync_external_factory(None)
    else:
        _sync_external_factory(getattr(module, "get_llm_provider", None))
