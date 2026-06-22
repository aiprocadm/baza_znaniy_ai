"""Provider that delegates text generation to an external HTTP API."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING, runtime_checkable, cast

from app.llm.exceptions import ModelNotReadyError, NonRetryableProviderError, RetryableProviderError

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from app.core.config import Settings as SettingsType
else:

    @runtime_checkable
    class SettingsType(Protocol):
        llm_api_base_url: str | None
        llm_api_key: str | None
        llm_api_model: str
        llm_temperature: float
        llm_top_p: float
        llm_max_tokens: int
        llm_api_timeout_sec: float
        llm_api_retries: int
        llm_api_backoff_sec: float


try:  # pragma: no cover - optional dependency may be missing in tests
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


_GENERATION_KEYS = {"temperature", "top_p", "max_tokens", "stop"}
LOGGER = logging.getLogger(__name__)


def _get_settings() -> SettingsType:
    from app.core.config import get_settings as _get_settings

    return cast(SettingsType, _get_settings())


@dataclass(slots=True)
class ApiProvider:
    """OpenAI-compatible API provider for remote LLM backends."""

    settings: SettingsType = field(default_factory=_get_settings)

    name: str = "api"

    def ensure_model(self) -> None:
        self.ensure_ready()

    def ensure_ready(self) -> None:
        if httpx is None:
            raise ModelNotReadyError("httpx is required for API-backed providers")

        base_url = (self.settings.llm_api_base_url or "").strip()
        if not base_url:
            raise ModelNotReadyError("LLM_API_BASE_URL must be configured for API provider")

        model = (self.settings.llm_api_model or "").strip()
        if not model:
            raise ModelNotReadyError("LLM_API_MODEL must be configured for API provider")

    def ensure_adapter(self) -> None:
        return None

    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        self.ensure_ready()

        assert httpx is not None  # narrow type after readiness check
        # ``ensure_ready`` above guarantees a non-empty base URL.
        base_url = cast(str, self.settings.llm_api_base_url)
        url = base_url.rstrip("/") + "/v1/chat/completions"

        payload: dict[str, Any] = {
            "model": self.settings.llm_api_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.settings.llm_temperature),
            "top_p": float(self.settings.llm_top_p),
            "max_tokens": int(self.settings.llm_max_tokens),
        }

        if context:
            for key, value in context.items():
                if key in _GENERATION_KEYS:
                    payload[key] = value
            options = context.get("options") if isinstance(context, Mapping) else None
            if isinstance(options, Mapping):
                for key, value in options.items():
                    if key in _GENERATION_KEYS:
                        payload[key] = value

        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        timeout = max(1.0, float(self.settings.llm_api_timeout_sec))
        retries = max(0, int(self.settings.llm_api_retries))
        backoff_sec = max(0.0, float(self.settings.llm_api_backoff_sec))

        data: Any = {}
        last_retryable_exc: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as exc:
                    LOGGER.warning(
                        "External LLM API returned invalid JSON payload",
                        extra={"endpoint": url, "attempt": attempt},
                    )
                    raise NonRetryableProviderError(
                        "External LLM API returned invalid JSON"
                    ) from exc
                break
            except httpx.TimeoutException as exc:
                last_retryable_exc = exc
                LOGGER.warning(
                    "External LLM API timeout",
                    extra={"endpoint": url, "attempt": attempt},
                )
                if attempt > retries:
                    raise RetryableProviderError("External LLM API request timed out") from exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                LOGGER.warning(
                    "External LLM API returned HTTP error",
                    extra={"endpoint": url, "attempt": attempt, "status_code": status_code},
                )
                if status_code == 429 or 500 <= status_code <= 599:
                    last_retryable_exc = exc
                    if attempt > retries:
                        raise RetryableProviderError(
                            "External LLM API request failed with retryable HTTP status"
                        ) from exc
                else:
                    raise NonRetryableProviderError(
                        "External LLM API request failed with non-retryable HTTP status"
                    ) from exc
            except httpx.RequestError as exc:
                last_retryable_exc = exc
                LOGGER.warning(
                    "External LLM API request error",
                    extra={"endpoint": url, "attempt": attempt},
                )
                if attempt > retries:
                    raise RetryableProviderError("External LLM API request failed") from exc

            sleep_time = backoff_sec * (2 ** (attempt - 1))
            if sleep_time > 0:
                time.sleep(sleep_time)
        else:  # pragma: no cover - defensive; loop should always break/raise
            raise ModelNotReadyError("External LLM API request failed") from last_retryable_exc

        choices = data.get("choices", []) if isinstance(data, Mapping) else []
        if not choices:
            return ""

        first = choices[0] if isinstance(choices[0], Mapping) else {}
        message = first.get("message") if isinstance(first, Mapping) else None
        if isinstance(message, Mapping) and "content" in message:
            return str(message.get("content", "")).strip()

        return str(first.get("text", "")).strip() if isinstance(first, Mapping) else ""
