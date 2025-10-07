"""Helpers for reporting service and model versions."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings, get_version_info

_DISABLED_SENTINELS = {
    "",
    "0",
    "false",
    "no",
    "off",
    "none",
    "null",
    "disabled",
}


def _is_truthy_flag(value: str | None) -> bool:
    """Return ``True`` when the supplied string represents an enabled flag."""

    if value is None:
        return False
    normalised = value.strip().lower()
    if normalised in _DISABLED_SENTINELS:
        return False
    return bool(normalised)


def resolve_lora_enabled(settings: Settings) -> bool:
    """Return whether a LoRA adapter is effectively enabled."""

    if settings.lora_adapter_path:
        return True
    return _is_truthy_flag(settings.llm_lora_adapter)


def build_version_payload(settings: Settings | None = None) -> dict[str, Any]:
    """Return version payload with normalised LoRA enablement."""

    resolved = settings or get_settings()
    payload = get_version_info(resolved)
    lora_section = payload.setdefault("lora", {})
    lora_section["enabled"] = resolve_lora_enabled(resolved)
    return payload


__all__ = ["build_version_payload", "resolve_lora_enabled"]
