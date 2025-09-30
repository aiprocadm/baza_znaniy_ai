"""Language model provider implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from app.core.config import Settings, get_settings


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by all language model providers."""

    name: str

    def ensure_model(self) -> None:
        """Ensure the underlying model (if any) is ready for use."""

    def generate(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        """Generate a completion for *prompt*."""


@dataclass(slots=True)
class OllamaProvider:
    """HTTP client wrapper for interacting with an Ollama deployment."""

    settings: Settings
    _model_ensured: bool = False

    name: str = "ollama"

    @property
    def base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        return self.settings.llm_model_name

    @property
    def max_context_tokens(self) -> int:
        return self.settings.max_context_tokens

    def ensure_model(self) -> None:
        """Ensure the configured model is available locally."""

        if self._model_ensured:
            return
        try:
            with httpx.Client(timeout=60) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models: list[dict[str, Any]] = response.json().get("models", [])
                names = {item.get("name") for item in models}
            if self.model_name not in names:
                with httpx.Client(timeout=None) as client:
                    client.post(
                        f"{self.base_url}/api/pull", json={"name": self.model_name}
                    )
            self._model_ensured = True
        except Exception:  # pragma: no cover - service may be offline
            return

    def generate(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        options: dict[str, Any] = {}
        if self.max_context_tokens:
            options["num_ctx"] = self.max_context_tokens
        if options:
            payload["options"] = options
        if context:
            payload.update({key: value for key, value in context.items() if key not in payload})
        with httpx.Client(timeout=None) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")


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

    def generate(self, prompt: str, *, context: dict[str, Any] | None = None) -> str:
        base = self._base_response(prompt)
        citations = []
        if context:
            citations = list(context.get("citations", []) or [])
        formatted_citations = self._format_citations(citations)
        if formatted_citations:
            return "\n\n".join([base, formatted_citations])
        return base


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Factory returning an LLM provider according to *settings*."""

    resolved_settings = settings or get_settings()
    provider_name = (resolved_settings.llm_provider or "ollama").lower()
    if provider_name == "stub":
        return StubProvider(resolved_settings)
    if provider_name in {"ollama", "ollama-provider"}:
        return OllamaProvider(resolved_settings)
    raise ValueError(f"Unsupported LLM provider: {resolved_settings.llm_provider!r}")


__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "StubProvider",
    "get_llm_provider",
]
