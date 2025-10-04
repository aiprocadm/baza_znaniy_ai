"""Regression tests for app.llm.cache module imports."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def preserve_modules(*names: str) -> Iterator[None]:
    """Temporarily remove modules from ``sys.modules`` and restore them."""

    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules.pop(name, None)
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
            original = saved.get(name)
            if original is not None:
                sys.modules[name] = original


def test_get_cached_provider_imports_without_settings() -> None:
    """``app.llm.cache`` should tolerate missing ``Settings`` symbol."""

    modules_to_manage = (
        "app.core.config",
        "app.llm.cache",
        "app.llm.providers",
        "app.llm.llama_cpp_provider",
    )

    with preserve_modules(*modules_to_manage):
        sentinel_settings = object()

        # Stub ``app.core.config`` without exporting ``Settings``
        config_stub = types.ModuleType("app.core.config")

        def fake_get_settings() -> object:
            return sentinel_settings

        config_stub.get_settings = fake_get_settings  # type: ignore[attr-defined]
        sys.modules["app.core.config"] = config_stub

        # Stub provider modules used by ``app.llm.cache``
        provider_stub = types.ModuleType("app.llm.providers")
        llama_stub = types.ModuleType("app.llm.llama_cpp_provider")

        class DummyProvider:
            """Minimal stand-in for the real provider implementation."""

            name = "dummy"

            def __init__(self, settings: object | None = None) -> None:
                self.settings = settings

            def ensure_model(self) -> None:  # pragma: no cover - unused in test
                return None

            def ensure_ready(self) -> None:  # pragma: no cover - unused in test
                return None

            def ensure_adapter(self) -> None:  # pragma: no cover - unused in test
                return None

            def generate(self, prompt: str, *, context: object | None = None) -> str:
                return prompt

        def fake_get_llm_provider(settings: object | None = None) -> DummyProvider:
            return DummyProvider(settings)

        provider_stub.LLMProvider = DummyProvider  # type: ignore[attr-defined]
        provider_stub.LlamaCppProvider = DummyProvider  # type: ignore[attr-defined]
        provider_stub.get_llm_provider = fake_get_llm_provider  # type: ignore[attr-defined]
        llama_stub.LlamaCppProvider = DummyProvider  # type: ignore[attr-defined]

        sys.modules["app.llm.providers"] = provider_stub
        sys.modules["app.llm.llama_cpp_provider"] = llama_stub

        cache = importlib.import_module("app.llm.cache")

        provider = cache.get_cached_provider()

        assert isinstance(provider, DummyProvider)
        assert provider.settings is sentinel_settings
