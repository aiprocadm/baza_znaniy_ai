"""Public exports for LLM provider management."""

from __future__ import annotations

import sys
from types import ModuleType

from .cache import (
    LLMProvider,
    LLMProviderError,
    LlamaCppProvider,
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
    _register_llm_module,
    _sync_external_factory,
    get_cached_provider,
    get_llm_client,
    get_llm_provider,
    reset_provider_cache,
)

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
    "get_llm_client",
]


class _LlmModule(ModuleType):
    """Custom module type capturing external factory overrides."""

    def __setattr__(self, name: str, value: object) -> None:
        if name == "get_llm_provider":
            _sync_external_factory(value)
        super().__setattr__(name, value)


_module = sys.modules[__name__]
if not isinstance(_module, _LlmModule):
    _module.__class__ = _LlmModule  # type: ignore[misc]
    _register_llm_module(_module)
