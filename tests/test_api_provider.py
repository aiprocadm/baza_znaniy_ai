from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.llm.api_provider import ApiProvider
from app.llm.exceptions import ModelNotReadyError, NonRetryableProviderError, RetryableProviderError
from app.llm.providers import get_llm_provider


def test_factory_selects_api_provider() -> None:
    settings = Settings(llm_provider="api", llm_api_base_url="https://example.org")
    provider = get_llm_provider(settings)
    assert isinstance(provider, ApiProvider)


def test_api_provider_requires_base_url() -> None:
    provider = ApiProvider(settings=Settings(llm_provider="api", llm_api_base_url=None))
    with pytest.raises(ModelNotReadyError):
        provider.ensure_ready()


def test_api_provider_generate_uses_openai_compatible_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {"content": "Готово"},
                    }
                ]
            }

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("app.llm.api_provider.httpx", SimpleNamespace(post=fake_post))

    settings = Settings(
        llm_provider="api",
        llm_api_base_url="https://api.example.com",
        llm_api_key="secret",
        llm_api_model="custom-model",
        llm_temperature=0.3,
    )
    provider = ApiProvider(settings=settings)
    text = provider.generate("Hello", context={"options": {"max_tokens": 77}})

    assert text == "Готово"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "custom-model"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["max_tokens"] == 77
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer secret"


def test_api_provider_generate_timeout_error_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TimeoutException(Exception):
        pass

    class _RequestError(Exception):
        pass

    class _HTTPStatusError(Exception):
        def __init__(self, response: object) -> None:
            super().__init__("status error")
            self.response = response

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float):
        raise _TimeoutException("timeout")

    monkeypatch.setattr(
        "app.llm.api_provider.httpx",
        SimpleNamespace(
            post=fake_post,
            TimeoutException=_TimeoutException,
            HTTPStatusError=_HTTPStatusError,
            RequestError=_RequestError,
        ),
    )

    settings = Settings(
        llm_provider="api",
        llm_api_base_url="https://api.example.com",
        llm_api_retries=0,
    )
    provider = ApiProvider(settings=settings)

    with pytest.raises(RetryableProviderError):
        provider.generate("Hello")


def test_api_provider_generate_retries_on_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TimeoutException(Exception):
        pass

    class _RequestError(Exception):
        pass

    class _Response:
        def __init__(self, status_code: int, payload: dict[str, object] | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise _HTTPStatusError(self)

        def json(self) -> dict[str, object]:
            return self._payload

    class _HTTPStatusError(Exception):
        def __init__(self, response: _Response) -> None:
            super().__init__("status error")
            self.response = response

    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return _Response(status_code=429)
        return _Response(
            status_code=200,
            payload={"choices": [{"message": {"content": "после retry"}}]},
        )

    monkeypatch.setattr(
        "app.llm.api_provider.httpx",
        SimpleNamespace(
            post=fake_post,
            TimeoutException=_TimeoutException,
            HTTPStatusError=_HTTPStatusError,
            RequestError=_RequestError,
        ),
    )
    monkeypatch.setattr(
        "app.llm.api_provider.time.sleep", lambda seconds: sleep_calls.append(seconds)
    )

    settings = Settings(
        llm_provider="api",
        llm_api_base_url="https://api.example.com",
        llm_api_retries=2,
        llm_api_backoff_sec=0.1,
    )
    provider = ApiProvider(settings=settings)

    text = provider.generate("Hello")

    assert text == "после retry"
    assert attempts["count"] == 2
    assert sleep_calls == [0.1]


def test_api_provider_generate_invalid_payload_is_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TimeoutException(Exception):
        pass

    class _RequestError(Exception):
        pass

    class _HTTPStatusError(Exception):
        def __init__(self, response: object) -> None:
            super().__init__("status error")
            self.response = response

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            raise ValueError("invalid json")

    def fake_post(url: str, *, json: dict[str, object], headers: dict[str, str], timeout: float):
        return _Response()

    monkeypatch.setattr(
        "app.llm.api_provider.httpx",
        SimpleNamespace(
            post=fake_post,
            TimeoutException=_TimeoutException,
            HTTPStatusError=_HTTPStatusError,
            RequestError=_RequestError,
        ),
    )

    settings = Settings(
        llm_provider="api",
        llm_api_base_url="https://api.example.com",
    )
    provider = ApiProvider(settings=settings)

    with pytest.raises(NonRetryableProviderError):
        provider.generate("Hello")
