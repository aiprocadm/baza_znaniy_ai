"""Language model provider implementations."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from app.core.config import Settings, get_settings

from .llama_cpp_provider import LlamaCppProvider


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by all language model providers."""

    name: str

    def ensure_model(self) -> None:
        """Ensure the underlying model (if any) is ready for use."""

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        """Generate a completion for *prompt*."""


class StubProvider:
    """Deterministic provider used in tests and offline environments."""

    name = "stub"
    handles_citations = True

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def ensure_model(self) -> None:  # pragma: no cover - nothing to ensure
        return None

    def _base_response(self, prompt: str) -> str:
        prompt_normalized = prompt.strip().lower()
        if not prompt_normalized:
            return "Я пока не знаю, что ответить."
        if "привет" in prompt_normalized:
            return "Привет! Я тестовый помощник."
        if prompt_normalized.endswith("?"):
            return "Это тестовый ответ-заглушка."
        return f"Заглушечный ответ: {prompt.strip()}"

    def _format_citations(self, citations: list[dict[str, Any]]) -> str:
        if not citations:
            return ""
        lines: list[str] = []
        for index, citation in enumerate(citations, start=1):
            file_name = citation.get("file", "неизвестный источник")
            page = citation.get("page")
            if page is None:
                lines.append(f"[{index}] {file_name}")
            else:
                lines.append(f"[{index}] {file_name} — страница {page}")
        return "Источники:\n" + "\n".join(lines)

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        base = self._base_response(prompt)
        citations = []
        if context:
            citations = list((context.get("citations") or []))  # type: ignore[arg-type]
        formatted_citations = self._format_citations(citations)
        if formatted_citations:
            return "\n\n".join([base, formatted_citations])
        return base


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Factory returning an LLM provider according to *settings*."""

    resolved_settings = settings or get_settings()
    provider_name = (resolved_settings.llm_provider or "llama-cpp").lower()
    if provider_name == "stub":
        return StubProvider(resolved_settings)
    if provider_name in {"llama", "llama-cpp", "llama_cpp", "llamacpp"}:
        return LlamaCppProvider(resolved_settings)
    raise ValueError(f"Unsupported LLM provider: {resolved_settings.llm_provider!r}")


__all__ = [
    "LLMProvider",
    "LlamaCppProvider",
    "StubProvider",
    "get_llm_provider",
]
