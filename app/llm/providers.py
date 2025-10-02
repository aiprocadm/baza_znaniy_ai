"""Language model provider implementations."""

from __future__ import annotations

from typing import (
    Any,
    Mapping,
    Protocol,
    runtime_checkable,
)

from app.core.config import Settings, get_settings

from .llama_cpp_provider import LlamaCppProvider


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by all language model providers."""

    name: str

    def ensure_model(self) -> None:
        """Ensure the underlying model (if any) is ready for use."""
    def ensure_ready(self) -> None:
        """Perform provider specific readiness checks."""

    def ensure_adapter(self) -> None:
        """Ensure optional adapters (such as LoRA) are available."""

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        """Generate a completion for *prompt*."""


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Factory returning an LLM provider according to *settings*."""

    resolved_settings = settings or get_settings()
    provider_name = (resolved_settings.llm_provider or "llama-cpp").lower()
    if provider_name in {"llama", "llama-cpp", "llama_cpp", "llamacpp"}:
        return LlamaCppProvider(resolved_settings)
    raise ValueError(f"Unsupported LLM provider: {resolved_settings.llm_provider!r}")


__all__ = [
    "LLMProvider",
    "LlamaCppProvider",
    "get_llm_provider",
]
