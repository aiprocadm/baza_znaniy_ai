"""Pluggable LLM providers for the MVP knowledge base.

All supported providers (DeepSeek, Groq, OpenRouter, OpenAI, Ollama,
custom) speak the same OpenAI ``POST /v1/chat/completions`` protocol,
so one transport (:class:`OpenAICompatibleProvider`) covers them all.
Auto-selection by env key is documented in the README's «Подключение
LLM-провайдеров» section.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

try:  # pragma: no cover - tests run without httpx in some matrices
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from app.services._envutil import env as _env

LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.2


class LLMUnavailable(RuntimeError):
    """Raised when a provider is selected but cannot be used."""


class LLMTransportError(RuntimeError):
    """Raised when an HTTP call to a provider fails."""


@dataclass(frozen=True)
class LLMResponse:
    """Successful response from a chat-completions provider."""

    text: str
    provider: str
    model: str
    elapsed_ms: float
    raw_usage: Optional[Mapping[str, Any]] = None


@dataclass
class LLMConfig:
    """Connection parameters for an OpenAI-compatible chat endpoint."""

    provider: str
    api_base: str
    model: str
    api_key: Optional[str] = None
    timeout: float = _DEFAULT_TIMEOUT
    max_tokens: int = _DEFAULT_MAX_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE
    extra_headers: Mapping[str, str] = field(default_factory=dict)


# Per-provider presets. ``key_env`` is the variable used for both
# auto-detection and per-provider key resolution. ``model_env`` allows
# overriding the default model without changing the codebase.
KNOWN_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "api_base": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "needs_key": True,
    },
    "groq": {
        "api_base": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "needs_key": True,
    },
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-chat",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "needs_key": True,
        "extra_headers_env": {
            "HTTP-Referer": "OPENROUTER_REFERER",
            "X-Title": "OPENROUTER_TITLE",
        },
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "needs_key": True,
    },
    "ollama": {
        "api_base": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "key_env": None,
        "model_env": "OLLAMA_MODEL",
        "base_env": "OLLAMA_BASE_URL",
        "needs_key": False,
    },
    "custom": {
        "api_base": None,
        "default_model": "gpt-4o-mini",
        "key_env": "LLM_API_KEY",
        "model_env": "LLM_API_MODEL",
        "base_env": "LLM_API_BASE_URL",
        "needs_key": False,
    },
}

# Order of auto-detection — cheapest/fastest first.
_AUTO_ORDER = ("deepseek", "groq", "openrouter", "openai", "custom")


def _build_config(provider: str, env: Mapping[str, str] | None = None) -> LLMConfig:
    preset = KNOWN_PRESETS.get(provider)
    if preset is None:
        raise LLMUnavailable(f"Unknown LLM provider: {provider}")

    base_env = preset.get("base_env")
    api_base = _env(base_env, env) if base_env else None
    if not api_base:
        api_base = preset.get("api_base")
    if not api_base:
        raise LLMUnavailable(
            f"Provider {provider!r} requires a base URL (set {base_env or 'api_base'})"
        )
    if api_base.endswith("/"):
        api_base = api_base.rstrip("/")

    model_env = preset.get("model_env")
    model = _env(model_env, env) if model_env else None
    if not model:
        model = preset["default_model"]

    api_key = None
    key_env = preset.get("key_env")
    if key_env:
        api_key = _env(key_env, env)
    if preset.get("needs_key") and not api_key:
        raise LLMUnavailable(f"Provider {provider!r} requires an API key in {key_env}")

    extra_headers: dict[str, str] = {}
    extra_env = preset.get("extra_headers_env", {}) or {}
    for header_name, env_name in extra_env.items():
        value = _env(env_name, env)
        if value:
            extra_headers[header_name] = value

    timeout = _DEFAULT_TIMEOUT
    raw_timeout = _env("KB_LLM_TIMEOUT", env)
    if raw_timeout:
        try:
            timeout = max(1.0, float(raw_timeout))
        except ValueError:
            pass

    max_tokens = _DEFAULT_MAX_TOKENS
    raw_max = _env("KB_LLM_MAX_TOKENS", env)
    if raw_max:
        try:
            max_tokens = max(64, int(raw_max))
        except ValueError:
            pass

    temperature = _DEFAULT_TEMPERATURE
    raw_temp = _env("KB_LLM_TEMPERATURE", env)
    if raw_temp:
        try:
            temperature = max(0.0, min(2.0, float(raw_temp)))
        except ValueError:
            pass

    return LLMConfig(
        provider=provider,
        api_base=api_base,
        model=model,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_headers=extra_headers,
    )


class OpenAICompatibleProvider:
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        if httpx is None:
            return False
        if KNOWN_PRESETS.get(self.config.provider, {}).get("needs_key"):
            return bool(self.config.api_key)
        return bool(self.config.api_base)

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        if httpx is None:
            raise LLMUnavailable("httpx is required to call LLM providers")
        if not prompt.strip():
            raise ValueError("prompt is empty")

        url = f"{self.config.api_base}/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": False,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for header_name, value in self.config.extra_headers.items():
            headers[header_name] = value

        started = time.perf_counter()
        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout,
            )
        except httpx.RequestError as exc:  # pragma: no cover - network failure
            raise LLMTransportError(f"network error: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code >= 400:
            detail = _safe_error_message(response)
            raise LLMTransportError(
                f"{self.config.provider} returned HTTP {response.status_code}: {detail}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMTransportError("provider returned non-JSON body") from exc

        text = _extract_text(data)
        if not text:
            raise LLMTransportError(f"{self.config.provider} returned an empty completion")

        usage = data.get("usage") if isinstance(data, dict) else None

        return LLMResponse(
            text=text.strip(),
            provider=self.config.provider,
            model=self.config.model,
            elapsed_ms=round(elapsed_ms, 2),
            raw_usage=usage,
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """Yield assistant text chunks via OpenAI-compatible SSE streaming.

        All supported providers (DeepSeek, Groq, OpenRouter, OpenAI,
        Ollama) accept ``"stream": true`` and emit:

            data: {"choices":[{"delta":{"content":"Hi"}}]}\\n\\n
            ...
            data: [DONE]\\n\\n

        We parse line-by-line, extract ``delta.content`` and yield it.
        ``[DONE]`` terminates the stream cleanly.
        """

        if httpx is None:
            raise LLMUnavailable("httpx is required to call LLM providers")
        if not prompt.strip():
            raise ValueError("prompt is empty")

        url = f"{self.config.api_base}/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "stream": True,
        }

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for header_name, value in self.config.extra_headers.items():
            headers[header_name] = value

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        detail = (body.decode("utf-8", errors="replace") or "")[:300]
                        raise LLMTransportError(
                            f"{self.config.provider} returned HTTP {response.status_code}: {detail}"
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith(":"):
                            continue  # SSE comment / keep-alive
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        if not data:
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            LOGGER.debug("Ignoring malformed SSE chunk: %s", data[:80])
                            continue
                        delta = _extract_delta_text(chunk)
                        if delta:
                            yield delta
        except httpx.RequestError as exc:
            raise LLMTransportError(f"network error: {exc}") from exc


def _extract_delta_text(data: Any) -> str:
    """Extract assistant token from a streaming OpenAI-compatible chunk."""

    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    # Some providers (notably Ollama) embed the partial in 'message' instead
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _safe_error_message(response: "httpx.Response") -> str:
    try:
        data = response.json()
    except Exception:
        return (response.text or "")[:200]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and "message" in err:
            return str(err["message"])[:300]
        if "message" in data:
            return str(data["message"])[:300]
    return json.dumps(data)[:300]


def _extract_text(data: Any) -> str:
    """Return the assistant text from an OpenAI-compatible payload."""

    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    return str(text) if isinstance(text, str) else ""


def build_provider(provider: str, env: Mapping[str, str] | None = None) -> OpenAICompatibleProvider:
    """Construct a provider by short name, raising :class:`LLMUnavailable`.

    ``provider`` is a key of :data:`KNOWN_PRESETS` (``"deepseek"``, …).
    Missing keys/base URLs result in :class:`LLMUnavailable`.
    """

    config = _build_config(provider, env=env)
    provider_obj = OpenAICompatibleProvider(config)
    if not provider_obj.is_available():
        raise LLMUnavailable(f"Provider {provider!r} is configured but not usable")
    return provider_obj


def select_provider(
    env: Mapping[str, str] | None = None,
) -> Optional[OpenAICompatibleProvider]:
    """Pick the most appropriate provider for the current environment."""

    explicit = _env("KB_LLM_PROVIDER", env)
    if explicit:
        try:
            return build_provider(explicit, env=env)
        except LLMUnavailable as exc:
            LOGGER.warning("Configured LLM provider unusable: %s", exc)
            return None

    for name in _AUTO_ORDER:
        preset = KNOWN_PRESETS.get(name) or {}
        key_env = preset.get("key_env")
        base_env = preset.get("base_env")
        if key_env and _env(key_env, env):
            try:
                return build_provider(name, env=env)
            except LLMUnavailable:
                continue
        elif name == "custom" and base_env and _env(base_env, env):
            try:
                return build_provider(name, env=env)
            except LLMUnavailable:
                continue

    return None


def provider_status(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Diagnostic snapshot of which providers are configured.

    Returned by ``GET /api/kb/health`` so operators can verify that the
    expected provider was picked up after editing ``.env``.
    """

    info: dict[str, Any] = {"providers": []}
    for name, preset in KNOWN_PRESETS.items():
        key_env = preset.get("key_env")
        base_env = preset.get("base_env")
        configured = False
        if key_env and _env(key_env, env):
            configured = True
        elif base_env and _env(base_env, env):
            configured = True
        elif name == "ollama":
            configured = True  # always reachable in principle
        info["providers"].append(
            {
                "name": name,
                "configured": configured,
                "key_env": key_env,
                "base_env": base_env,
                "default_model": preset.get("default_model"),
            }
        )

    selected = select_provider(env)
    if selected is not None:
        info["selected"] = {
            "name": selected.name,
            "model": selected.model,
            "api_base": selected.config.api_base,
        }
    else:
        info["selected"] = None
    info["explicit"] = _env("KB_LLM_PROVIDER", env)
    return info


__all__ = [
    "KNOWN_PRESETS",
    "LLMConfig",
    "LLMResponse",
    "LLMTransportError",
    "LLMUnavailable",
    "OpenAICompatibleProvider",
    "build_provider",
    "provider_status",
    "select_provider",
]
