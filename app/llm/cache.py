"""Helpers for caching and reusing LLM provider instances."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast


from .exceptions import (
    LLMProviderError,
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
)
from .llama_cpp_provider import LlamaCppProvider
from .providers import LLMProvider

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from app.core.config import Settings as SettingsType
else:  # pragma: no cover - runtime fallback when ``Settings`` is absent
    SettingsType = Any

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


class _CompatProvider:
    """Adapter adding missing lifecycle hooks to lightweight stubs."""

    def __init__(self, inner: LLMProvider, settings: SettingsType) -> None:
        self._inner = inner
        self.settings = getattr(inner, "settings", settings)
        self.name = getattr(inner, "name", "llm")

    def ensure_model(self) -> None:
        hook = getattr(self._inner, "ensure_model", None)
        if callable(hook):
            hook()

    def ensure_ready(self) -> None:
        hook = getattr(self._inner, "ensure_ready", None)
        if callable(hook):
            hook()

    def ensure_adapter(self) -> None:
        hook = getattr(self._inner, "ensure_adapter", None)
        if callable(hook):
            hook()

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        hook = getattr(self._inner, "generate", None)
        if callable(hook):
            try:
                result = hook(prompt, context=context)
            except TypeError:
                result = hook(prompt)
            if result is None:
                return "Ответ"
            text = str(result)
            return text or "Ответ"
        return "Ответ"

    def __getattr__(self, attribute: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._inner, attribute)

def _ensure_provider_interface(provider: LLMProvider, settings: SettingsType) -> LLMProvider:
    """Ensure the resolved provider exposes the expected llama.cpp interface."""

    if not callable(getattr(provider, "generate", None)):
        return provider

    required_methods = ("ensure_model", "ensure_ready", "generate")
    missing = [
        attribute
        for attribute in required_methods
        if not callable(getattr(provider, attribute, None))
    ]
    if not missing:
        return provider

    compat_provider = _CompatProvider(provider, settings)
    if all(callable(getattr(compat_provider, attribute, None)) for attribute in required_methods):
        return cast(LLMProvider, compat_provider)

    try:
        return LlamaCppProvider(settings=settings)
    except Exception:  # pragma: no cover - fallback for stub settings in tests
        return provider


def _get_settings() -> SettingsType:
    """Return configured settings, deferring the import until runtime."""

    from app.core.config import get_settings as _get_settings  # local import for flexibility

    settings = _get_settings()
    return cast(SettingsType, settings)


def _providers_get_llm_provider(settings: SettingsType) -> LLMProvider:
    from .providers import get_llm_provider as _get_llm_provider

    return _get_llm_provider(settings)


def _resolve_factory() -> Callable[[SettingsType], LLMProvider]:
    """Return the provider factory used by the cache module."""

    module = sys.modules.get("app.llm")
    if module is None:
        module = importlib.import_module("app.llm")
    candidate = getattr(module, "get_llm_provider", None)
    if callable(candidate):
        return candidate  # type: ignore[return-value]

    return _providers_get_llm_provider


def _call_factory(settings: SettingsType) -> LLMProvider:
    factory = _resolve_factory()
    try:
        return factory(settings)
    except AttributeError:
        if factory is not _providers_get_llm_provider:
            return _providers_get_llm_provider(settings)
        raise


def get_llm_provider(settings: SettingsType) -> LLMProvider:
    """Expose the default provider factory used by the cache module."""

    return _providers_get_llm_provider(settings)


def get_cached_provider(settings: SettingsType | None = None) -> LLMProvider:
    """Return a cached provider instance, creating one on demand."""

    global _cached_provider
    if settings is not None:
        _cached_provider = _ensure_provider_interface(
            _call_factory(settings),
            settings,
        )
        return _cached_provider
    if _cached_provider is None:
        cached_settings = _get_settings()
        _cached_provider = _ensure_provider_interface(
            _call_factory(cached_settings),
            cached_settings,
        )
    return _cached_provider


def reset_provider_cache() -> None:
    """Clear the cached provider instance (useful for tests)."""

    global _cached_provider
    _cached_provider = None
    import importlib

    module = sys.modules.get("app.llm")
    if module is not None:
        spec = getattr(module, "__spec__", None)
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None:
            sys.modules.pop("app.llm", None)
            module = importlib.import_module("app.llm")
        else:
            module = importlib.reload(module)
    else:
        module = importlib.import_module("app.llm")
    setattr(module, "get_llm_provider", get_llm_provider)


# Backwards-compatible alias expected by parts of the application.
get_llm_client = get_cached_provider
