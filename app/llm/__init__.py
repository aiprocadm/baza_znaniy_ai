"""Public exports for LLM provider management."""

from __future__ import annotations

import sys
import types

from .cache import (
    LLMProvider,
    LLMProviderError,
    LlamaCppProvider,
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    get_cached_provider,
    get_llm_client,
    get_llm_provider,
    reset_provider_cache,
    set_provider_factory_override,
)

__all__ = [
    "LLMProvider",
    "LlamaCppProvider",
    "get_llm_provider",
    "get_cached_provider",
    "reset_provider_cache",
    "set_provider_factory_override",
    "LLMProviderError",
    "ModelNotFoundError",
    "ModelNotReadyError",
    "LoRAAdapterNotFoundError",
    "get_llm_client",
]


class _LLMModule(types.ModuleType):
    """Module proxy that tracks overrides for ``get_llm_provider``."""

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover - simple setter
        if name == "get_llm_provider":
            if callable(value) and value is not get_llm_provider:
                set_provider_factory_override(value)
            else:
                set_provider_factory_override(None)
        super().__setattr__(name, value)


_current_module = sys.modules.get(__name__)
if _current_module is not None and not isinstance(_current_module, _LLMModule):
    _current_module.__class__ = _LLMModule  # type: ignore[misc]
