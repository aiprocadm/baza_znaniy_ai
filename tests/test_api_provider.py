from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.llm.api_provider import ApiProvider
from app.llm.exceptions import ModelNotReadyError
from app.llm.providers import get_llm_provider


def test_factory_selects_api_provider() -> None:
    settings = Settings(llm_provider="api", llm_api_base_url="https://example.org")
    provider = get_llm_provider(settings)
    assert isinstance(provider, ApiProvider)


def test_api_provider_requires_base_url() -> None:
    provider = ApiProvider(settings=Settings(llm_provider="api", llm_api_base_url=None))
    with pytest.raises(ModelNotReadyError):
        provider.ensure_ready()


def test_api_provider_generate_uses_openai_compatible_shape(monkeypatch: pytest.MonkeyPatch) -> None:
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
