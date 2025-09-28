"""Minimal Ollama client used for text generation."""

from __future__ import annotations

import httpx

from .config import get_settings

_ENSURED_MODEL = False


def _base_url() -> str:
    return get_settings().ollama_base_url.rstrip("/")


def ensure_model() -> None:
    """Ensure the configured model is available locally."""

    global _ENSURED_MODEL
    if _ENSURED_MODEL:
        return

    settings = get_settings()
    base_url = _base_url()
    try:
        with httpx.Client(timeout=60) as client:
            response = client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {item.get("name") for item in models}
        if settings.gen_model in names:
            _ENSURED_MODEL = True
            return
        with httpx.Client(timeout=None) as client:
            client.post(f"{base_url}/api/pull", json={"name": settings.gen_model})
    except Exception:  # pragma: no cover - service may be offline in tests
        return
    _ENSURED_MODEL = True


def generate(prompt: str) -> str:
    settings = get_settings()
    base_url = _base_url()
    with httpx.Client(timeout=None) as client:
        response = client.post(
            f"{base_url}/api/generate",
            json={"model": settings.gen_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json().get("response", "")


__all__ = ["ensure_model", "generate"]
