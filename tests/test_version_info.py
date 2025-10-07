"""Unit tests for version information helpers."""

from app.core.config import Settings
from app.core.versioning import build_version_payload


def test_lora_flag_disabled_by_sentinel_value() -> None:
    settings = Settings(llm_lora_adapter="none", lora_adapter_path=None)

    info = build_version_payload(settings)

    assert info["lora"]["enabled"] is False


def test_lora_flag_disabled_by_zero_string() -> None:
    settings = Settings(llm_lora_adapter="0", lora_adapter_path=None)

    info = build_version_payload(settings)

    assert info["lora"]["enabled"] is False


def test_lora_flag_enabled_for_adapter_name() -> None:
    settings = Settings(llm_lora_adapter="stub-adapter", lora_adapter_path=None)

    info = build_version_payload(settings)

    assert info["lora"]["enabled"] is True
    assert info["lora"]["adapter"] == "stub-adapter"
