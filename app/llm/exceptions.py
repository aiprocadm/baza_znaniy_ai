"""Custom exception hierarchy for LLM providers."""

from __future__ import annotations

from pathlib import Path


class LLMProviderError(RuntimeError):
    """Base class for provider-specific errors."""


class ModelNotReadyError(LLMProviderError):
    """Raised when the underlying model has not been initialised."""


class ModelNotFoundError(LLMProviderError):
    """Raised when a configured model file cannot be located."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(f"LLM model not found: {path}")
        self.path = Path(path)


class LoRAAdapterNotFoundError(LLMProviderError):
    """Raised when a requested LoRA adapter file does not exist."""

    def __init__(self, path: Path | str) -> None:
        super().__init__(f"LoRA adapter not found: {path}")
        self.path = Path(path)


__all__ = [
    "LLMProviderError",
    "ModelNotReadyError",
    "ModelNotFoundError",
    "LoRAAdapterNotFoundError",
]
