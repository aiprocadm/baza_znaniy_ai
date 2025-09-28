"""Minimal Ollama client used for text generation."""

from __future__ import annotations

import os
from typing import Any

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")).rstrip("/")
MODEL_NAME = os.getenv("GEN_MODEL", "qwen2.5:3b-instruct")


def ensure_model() -> None:
    """Ensure the configured model is available locally."""

    try:
        with httpx.Client(timeout=60) as client:
            response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            models: list[dict[str, Any]] = response.json().get("models", [])
            names = {item.get("name") for item in models}
        if MODEL_NAME in names:
            return
        with httpx.Client(timeout=None) as client:
            client.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": MODEL_NAME})
    except Exception:  # pragma: no cover - service may be offline in tests
        pass


def generate(prompt: str) -> str:
    with httpx.Client(timeout=None) as client:
        response = client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json().get("response", "")


__all__ = ["ensure_model", "generate"]
