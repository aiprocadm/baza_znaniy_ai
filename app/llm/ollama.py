"""Client wrapper around an Ollama deployment."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings, get_settings


class OllamaClient:
    """HTTP client for interacting with an Ollama server."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def base_url(self) -> str:
        return self.settings.ollama_base_url.rstrip("/")

    @property
    def model_name(self) -> str:
        settings = self.settings
        if hasattr(settings, "llm_model"):
            return getattr(settings, "llm_model")
        return getattr(settings, "llm_model_name")

    def ensure_model(self) -> None:
        """Ensure the configured model is available locally."""

        try:
            with httpx.Client(timeout=60) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models: list[dict[str, Any]] = response.json().get("models", [])
                names = {item.get("name") for item in models}
            if self.model_name in names:
                return
            with httpx.Client(timeout=None) as client:
                client.post(f"{self.base_url}/api/pull", json={"name": self.model_name})
        except Exception:  # pragma: no cover - service may be offline in tests
            pass

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=None) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json().get("response", "")


def get_llm_client(settings: Settings | None = None) -> OllamaClient:
    """Return an :class:`OllamaClient` configured from :class:`Settings`."""

    return OllamaClient(settings=settings)


__all__ = ["OllamaClient", "get_llm_client"]
