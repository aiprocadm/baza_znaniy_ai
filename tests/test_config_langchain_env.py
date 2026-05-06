from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    from app.core import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _reload_settings_module() -> None:
    import app.core.config as config_module

    importlib.reload(config_module)


def test_new_defaults_keep_legacy_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_settings_module()

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.langchain_enabled is False
    assert settings.langchain_mode == "legacy"
    assert settings.rate_limit_backend == "memory"
    assert settings.billing_enabled is False
    assert settings.billing_provider == "none"


def test_new_env_variables_parse_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_settings_module()

    monkeypatch.setenv("LANGCHAIN_ENABLED", "true")
    monkeypatch.setenv("LANGCHAIN_MODE", "lcel")
    monkeypatch.setenv("LANGCHAIN_USE_HISTORY_AWARE", "1")
    monkeypatch.setenv("LANGCHAIN_RETURN_SOURCE_DOCS", "yes")
    monkeypatch.setenv("LANGCHAIN_TRACING", "on")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "proj-x")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RATE_LIMIT_BACKEND", "redis")
    monkeypatch.setenv("API_KEY_HASH_SALT", "salt-123")
    monkeypatch.setenv("BILLING_ENABLED", "true")
    monkeypatch.setenv("BILLING_PROVIDER", "stripe")
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://u:p@localhost:5432/db")

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.langchain_enabled is True
    assert settings.langchain_mode == "lcel"
    assert settings.langchain_use_history_aware is True
    assert settings.langchain_return_source_docs is True
    assert settings.langchain_tracing is True
    assert settings.langchain_project == "proj-x"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.rate_limit_backend == "redis"
    assert settings.api_key_hash_salt == "salt-123"
    assert settings.billing_enabled is True
    assert settings.billing_provider == "stripe"
    assert settings.postgres_dsn == "postgresql://u:p@localhost:5432/db"


def test_invalid_modes_fall_back_to_legacy_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_settings_module()

    monkeypatch.setenv("LANGCHAIN_MODE", "unexpected")
    monkeypatch.setenv("RATE_LIMIT_BACKEND", "unknown")

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.langchain_mode == "legacy"
    assert settings.rate_limit_backend == "memory"
