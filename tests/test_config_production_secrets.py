"""Production must refuse to boot with the shipped placeholder secrets.

The ``/api/v1`` multi-tenant surface signs JWTs with ``secret_key`` and hashes
API keys with ``api_key_hash_salt``. Both ship with predictable defaults
(``change-me`` / ``kb-ai-salt``) that are fine for local dev but catastrophic
in production: a known signing key lets anyone forge tokens, and a known salt
enables rainbow-table attacks on API-key hashes. The guard fails fast at
settings construction instead of silently running insecure.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.core import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _production_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")


def test_production_rejects_default_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _production_env(monkeypatch)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("API_KEY_HASH_SALT", "a-real-random-salt")

    from app.core.config import Settings

    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings()


def test_production_rejects_default_api_key_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    _production_env(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", "a-real-random-secret")
    monkeypatch.delenv("API_KEY_HASH_SALT", raising=False)

    from app.core.config import Settings

    with pytest.raises(ValueError, match="API_KEY_HASH_SALT"):
        Settings()


def test_production_accepts_overridden_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    _production_env(monkeypatch)
    monkeypatch.setenv("SECRET_KEY", "a-real-random-secret")
    monkeypatch.setenv("API_KEY_HASH_SALT", "a-real-random-salt")

    from app.core.config import Settings

    settings = Settings()
    assert settings.app_env == "production"
    assert settings.secret_key == "a-real-random-secret"


def test_development_tolerates_default_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev/test must keep working with the placeholder defaults."""

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("API_KEY_HASH_SALT", raising=False)

    from app.core.config import Settings

    settings = Settings()
    assert settings.secret_key == "change-me"
    assert settings.api_key_hash_salt == "kb-ai-salt"
