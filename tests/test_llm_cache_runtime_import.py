"""Ensure the LLM cache module can load without the ``Settings`` symbol."""

from __future__ import annotations

import importlib
import sys


def test_get_cached_provider_without_settings(monkeypatch) -> None:
    """Import the cache module when ``Settings`` is absent and call the helper."""

    config_module = importlib.import_module("app.core.config")
    monkeypatch.delattr(config_module, "Settings", raising=False)

    sys.modules.pop("app.llm.cache", None)
    cache = importlib.import_module("app.llm.cache")

    sentinel_provider = object()
    settings_stub = object()

    def _factory(settings: object) -> object:
        assert settings is settings_stub
        return sentinel_provider

    monkeypatch.setattr(cache, "_resolve_factory", lambda: _factory)
    cache.reset_provider_cache()

    try:
        assert cache.get_cached_provider(settings_stub) is sentinel_provider
    finally:
        cache.reset_provider_cache()
