"""Pluggable embedding backends: Ollama, OpenAI-compat API, hashing fallback.

The backend is picked from env at first ``get_embedder()`` call —
explicit ``KB_EMBEDDINGS_BACKEND`` wins, otherwise the first configured
provider in the precedence list (Ollama, API, hash). See the README's
«Embedding-модели» section for full configuration.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

try:  # pragma: no cover
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from app.observability.metrics import record_embedder_backend
from app.services._envutil import env as _env
from app.services.kb_store import EMBEDDING_DIM, embed as hashing_embed

LOGGER = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, text: str) -> list[float]:  # pragma: no cover - protocol
        ...


@dataclass
class HashingEmbedder:
    """Dependency-free fallback embedder used when no API is configured."""

    name: str = "hash"
    dimension: int = EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        return hashing_embed(text, dim=self.dimension)


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class OpenAICompatibleEmbedder:
    """Embedder that calls a remote OpenAI-style ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        dimension: Optional[int] = None,
        name: str = "openai-compatible",
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = max(1.0, timeout)
        self.name = name
        self._dimension = dimension
        self._probed = False

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe with a tiny payload on first access. Keeps construction
            # cheap and avoids surprising failures during app startup.
            self.embed("dim-probe")
        return int(self._dimension or EMBEDDING_DIM)

    def embed(self, text: str) -> list[float]:
        if httpx is None:
            raise RuntimeError("httpx is required for API embeddings")
        if not text:
            return [0.0] * (self._dimension or EMBEDDING_DIM)

        url = f"{self.api_base}/embeddings"
        payload = {"model": self.model, "input": text}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        vec = _extract_first_embedding(data)
        if not vec:
            raise RuntimeError(f"empty embedding from {self.name}")

        if self._dimension is None:
            self._dimension = len(vec)
        return _normalise(vec)


class OllamaEmbedder:
    """Embedder backed by a local Ollama server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        dimension: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = max(1.0, timeout)
        self.name = "ollama"
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.embed("dim-probe")
        return int(self._dimension or EMBEDDING_DIM)

    def embed(self, text: str) -> list[float]:
        if httpx is None:
            raise RuntimeError("httpx is required for Ollama embeddings")
        if not text:
            return [0.0] * (self._dimension or EMBEDDING_DIM)

        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        response = httpx.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        vec = data.get("embedding") if isinstance(data, dict) else None
        if not isinstance(vec, list) or not vec:
            raise RuntimeError("Ollama returned no embedding")

        if self._dimension is None:
            self._dimension = len(vec)
        return _normalise([float(v) for v in vec])


def _extract_first_embedding(data: object) -> list[float]:
    if not isinstance(data, dict):
        return []
    items = data.get("data")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            vec = first.get("embedding")
            if isinstance(vec, list):
                return [float(v) for v in vec]
    vec = data.get("embedding")
    if isinstance(vec, list):
        return [float(v) for v in vec]
    return []


def _build_from_env(env: Mapping[str, str] | None = None) -> Embedder:
    explicit = (_env("KB_EMBEDDINGS_BACKEND", env) or "").lower()
    if explicit == "hash" or explicit == "":
        pass  # decide below
    elif explicit not in {"ollama", "api", "hash"}:
        LOGGER.warning("Unknown KB_EMBEDDINGS_BACKEND=%r; falling back", explicit)

    ollama_base = _env("OLLAMA_BASE_URL", env) or "http://localhost:11434"
    ollama_model = _env("OLLAMA_EMBED_MODEL", env)

    api_base = _env("EMBEDDINGS_API_BASE_URL", env)
    api_model = _env("EMBEDDINGS_API_MODEL", env) or "text-embedding-3-small"
    api_key = _env("EMBEDDINGS_API_KEY", env)

    if explicit == "ollama" or (not explicit and ollama_model and httpx is not None):
        if ollama_model:
            record_embedder_backend("ollama")
            return OllamaEmbedder(base_url=ollama_base, model=ollama_model)
        if explicit == "ollama":
            LOGGER.warning("Ollama backend requested but OLLAMA_EMBED_MODEL missing")

    if explicit == "api" or (not explicit and api_base and httpx is not None):
        if api_base:
            record_embedder_backend("api")
            return OpenAICompatibleEmbedder(
                api_base=api_base,
                model=api_model,
                api_key=api_key,
                name="openai-compatible",
            )
        if explicit == "api":
            LOGGER.warning("API backend requested but EMBEDDINGS_API_BASE_URL missing")

    # Implicit hashing fallback in a production-like config (KB_API_KEY set)
    # is almost always an unintended silent failure: semantic search returns
    # near-random results while the LLM still answers confidently. Surface it.
    if not explicit and _env("KB_API_KEY", env):
        LOGGER.warning(
            "Falling back to hashing embedder while KB_API_KEY is set — "
            "semantic search will return near-random results. Set "
            "KB_EMBEDDINGS_BACKEND=ollama (+ OLLAMA_EMBED_MODEL) or "
            "KB_EMBEDDINGS_BACKEND=api (+ EMBEDDINGS_API_BASE_URL) for "
            "real embeddings; set KB_EMBEDDINGS_BACKEND=hash to silence this."
        )

    record_embedder_backend("hash")
    return HashingEmbedder()


_DEFAULT_EMBEDDER: Optional[Embedder] = None


def get_embedder(env: Mapping[str, str] | None = None) -> Embedder:
    """Return the cached embedder selected for the current environment."""

    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is not None and env is None:
        return _DEFAULT_EMBEDDER
    embedder = _build_from_env(env=env)
    if env is None:
        _DEFAULT_EMBEDDER = embedder
    return embedder


def reset_embedder() -> None:
    """Clear the cached embedder (used in tests)."""

    global _DEFAULT_EMBEDDER
    _DEFAULT_EMBEDDER = None


def embedder_status(env: Mapping[str, str] | None = None) -> dict[str, object]:
    """Diagnostic snapshot for ``/api/kb/health``.

    Returns the embedder name and its dimension when known. For remote
    embedders the dimension is reported as ``None`` until the first call
    has been made (we deliberately do NOT probe here — health checks
    must not issue paid HTTP calls).
    """

    embedder = get_embedder(env=env)
    dimension: object
    if isinstance(embedder, HashingEmbedder):
        dimension = embedder.dimension
    else:
        # Remote embedders only know their dim after the first probe.
        dimension = getattr(embedder, "_dimension", None)
    return {"name": embedder.name, "dimension": dimension}


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OllamaEmbedder",
    "OpenAICompatibleEmbedder",
    "embedder_status",
    "get_embedder",
    "reset_embedder",
]
