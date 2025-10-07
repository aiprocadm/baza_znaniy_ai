"""Unit tests for configuration helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.core import config


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Ensure ``get_settings`` cache does not leak between tests."""

    config.get_settings.cache_clear()
    try:
        yield
    finally:
        config.get_settings.cache_clear()


def test_get_version_info_populates_fallback_versions() -> None:
    """Missing model or adapter versions are exposed as ``unknown``."""

    settings = config.Settings(
        app_version=" 0.2.0 ",
        llm_model_name="kb-llama",
        llm_model_version=None,
        llm_lora_adapter=None,
        lora_adapter_version=None,
        lora_adapter_path=None,
    )

    info = config.get_version_info(settings)

    assert info["app"]["version"] == "0.2.0"
    assert info["model"]["version"] == "unknown"
    assert info["lora"]["version"] == "unknown"
    assert info["lora"]["enabled"] is False


def test_get_version_info_keeps_explicit_versions() -> None:
    """Provided version strings remain untouched in the payload."""

    settings = config.Settings(
        app_version="1.0.0",
        llm_model_name="kb-llama",
        llm_model_version="1.2.3",
        llm_lora_adapter="adapter-name",
        lora_adapter_version="4.5.6",
    )

    info = config.get_version_info(settings)

    assert info["app"]["version"] == "1.0.0"
    assert info["model"]["version"] == "1.2.3"
    assert info["lora"]["version"] == "4.5.6"
    assert info["lora"]["enabled"] is True
