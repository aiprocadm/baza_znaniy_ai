"""Language model provider implementations and factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import httpx

from app.core.config import Settings, get_settings

CitationInput = Any


class LLMProviderProtocol(Protocol):
    """Abstract interface implemented by language model providers."""

    formats_citations: bool

    def ensure_model(self) -> None:  # pragma: no cover - interface declaration
        """Ensure the underlying model assets are available."""

    def generate(
        self,
        message: str,
        context: str | None = None,
        citations: Sequence[CitationInput] | None = None,
    ) -> str:  # pragma: no cover - interface declaration
        """Generate a response for *message* optionally using *context*."""


@dataclass(slots=True)
class OllamaProvider(LLMProviderProtocol):
    """Provider backed by an Ollama deployment."""

    settings: Settings
    formats_citations: bool = False
    _model_ready: bool = False

    def __init__(self, settings: Settings | None = None) -> None:
        object.__setattr__(self, "settings", settings or get_settings())
        object.__setattr__(self, "_model_ready", False)

    @property
    def base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        return self.settings.llm_model_name

    @property
    def max_context_tokens(self) -> int | None:
        value = getattr(self.settings, "max_context_tokens", None)
        if value in {None, 0}:
            return None
        return int(value)

    def ensure_model(self) -> None:
        if self._model_ready:
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
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name},
                    )
            self._model_ready = True
        except Exception:  # pragma: no cover - remote service may be unavailable in tests
            pass

    def generate(
        self,
        message: str,
        context: str | None = None,
        citations: Sequence[CitationInput] | None = None,
    ) -> str:
        prompt_parts = []
        if context:
            prompt_parts.append(context.strip())
        prompt_parts.append(message.strip())
        prompt = "\n\n".join(part for part in prompt_parts if part)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if self.max_context_tokens:
            payload["options"] = {"num_ctx": self.max_context_tokens}

        with httpx.Client(timeout=None) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("response", "")


class StubProvider(LLMProviderProtocol):
    """Deterministic provider used for tests and offline development."""

    formats_citations = True

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.last_call: tuple[str, str | None, Sequence[CitationInput] | None] | None = None

    def ensure_model(self) -> None:  # pragma: no cover - no-op
        return None

    def _normalise_citations(
        self, citations: Sequence[CitationInput] | None
    ) -> list[tuple[str | None, int | None]]:
        if not citations:
            return []

        normalised: list[tuple[str | None, int | None]] = []
        for item in citations:
            file_name: str | None = None
            page: int | None = None
            if isinstance(item, Mapping):
                file_name = item.get("file")  # type: ignore[arg-type]
                page = item.get("page")  # type: ignore[arg-type]
            else:
                file_name = getattr(item, "file", None)
                page = getattr(item, "page", None)
            normalised.append((file_name, page))
        return normalised

    def generate(
        self,
        message: str,
        context: str | None = None,
        citations: Sequence[CitationInput] | None = None,
    ) -> str:
        self.last_call = (message, context, citations)

        text = (message or "").strip()
        context_text = (context or "").strip()

        if not text:
            reply = "Это заглушка. Сообщение не было предоставлено."
        elif text.endswith("?"):
            reply = "Это заглушка. Вопрос распознан и обработан."
        else:
            reply = f"Это заглушка. Получено сообщение: {text}"

        if context_text:
            preview = context_text.splitlines()[0][:160]
            reply = f"{reply}\n\nКонтекст: {preview}"

        normalised = self._normalise_citations(citations)
        if normalised:
            lines = []
            for idx, (file_name, page) in enumerate(normalised, start=1):
                location = f" — страница {page}" if page is not None else ""
                title = file_name or "неизвестный источник"
                lines.append(f"[{idx}] {title}{location}")
            reply = "\n".join([reply, "", "Источники:", *lines])

        return reply


_PROVIDER_CACHE: dict[tuple[str, str, str, int | None], LLMProviderProtocol] = {}


def _provider_cache_key(settings: Settings) -> tuple[str, str, str, int | None]:
    return (
        settings.llm_provider,
        settings.ollama_base_url.rstrip("/"),
        settings.llm_model_name,
        getattr(settings, "max_context_tokens", None),
    )


def get_llm_provider(settings: Settings | None = None) -> LLMProviderProtocol:
    """Return (and cache) the configured language model provider."""

    active_settings = settings or get_settings()
    key = _provider_cache_key(active_settings)
    if key in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[key]

    provider_name = active_settings.llm_provider
    if provider_name == "ollama":
        provider = OllamaProvider(active_settings)
    elif provider_name == "stub":
        provider = StubProvider(active_settings)
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported LLM provider: {provider_name}")

    _PROVIDER_CACHE[key] = provider
    return provider


def reset_llm_provider_cache() -> None:
    """Clear the cached provider instances (useful for tests)."""

    _PROVIDER_CACHE.clear()


__all__ = [
    "CitationInput",
    "LLMProviderProtocol",
    "OllamaProvider",
    "StubProvider",
    "get_llm_provider",
    "reset_llm_provider_cache",
]

