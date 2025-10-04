"""Regression tests for environment alias handling in settings shim."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure ``get_settings`` observes environment mutations."""

    from app.core import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _reload_settings_module() -> None:
    """Reload the configuration module to pick up shim changes if necessary."""

    import app.core.config as config_module

    importlib.reload(config_module)


def _prepare_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove chat memory related overrides to avoid leakage between tests."""

    for key in {
        "CHAT_MEMORY_MAXTOK",
        "CHAT_MEMORY_MAX_TOKENS",
        "MEMORY_MAX_TOKENS",
    }:
        monkeypatch.delenv(key, raising=False)


def test_alias_choice_is_used_when_primary_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should accept values provided via ``AliasChoices`` entries."""

    _prepare_environment(monkeypatch)
    _reload_settings_module()

    monkeypatch.setenv("CHAT_MEMORY_MAXTOK", "9876")

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.chat_memory_max_tokens == 9876


def test_primary_environment_variable_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary environment variable names should override alias matches."""

    _prepare_environment(monkeypatch)
    _reload_settings_module()

    monkeypatch.setenv("CHAT_MEMORY_MAX_TOKENS", "1234")
    monkeypatch.setenv("CHAT_MEMORY_MAXTOK", "4321")

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.chat_memory_max_tokens == 1234
