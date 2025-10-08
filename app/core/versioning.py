"""Helpers for reporting service and model versions."""

from __future__ import annotations

import os
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

    env_path = os.getenv("LORA_ADAPTER_PATH")
    if env_path and env_path.strip():
        return True

    env_name = os.getenv("LLM_LORA_ADAPTER")
    if env_name and env_name.strip():
        return True

    if settings.lora_adapter_path:
        return True
    return _is_truthy_flag(settings.llm_lora_adapter)


def build_version_payload(settings: Settings | None = None) -> dict[str, Any]:
    """Return version payload with normalised LoRA enablement."""

    resolved = settings or get_settings()
    payload = get_version_info(resolved)
    env_version = os.getenv("APP_VERSION")
    if env_version is not None:
        cleaned_version = env_version.strip()
        if cleaned_version:
            payload.setdefault("app", {})["version"] = cleaned_version

    env_model_version = os.getenv("LLM_MODEL_VERSION")
    if env_model_version is not None:
        cleaned_model = env_model_version.strip()
        payload.setdefault("model", {})["version"] = cleaned_model or "unknown"

    env_adapter_version = os.getenv("LORA_ADAPTER_VERSION")
    if env_adapter_version is not None:
        cleaned_adapter = env_adapter_version.strip()
        payload.setdefault("lora", {})["version"] = cleaned_adapter or "unknown"

    env_lora_flag = os.getenv("LORA_ADAPTER_PATH") or os.getenv("LLM_LORA_ADAPTER")
    if env_lora_flag:
        payload.setdefault("lora", {})["enabled"] = True
    lora_section = payload.setdefault("lora", {})
    lora_section["enabled"] = resolve_lora_enabled(resolved)
    return payload


__all__ = ["build_version_payload", "resolve_lora_enabled"]
