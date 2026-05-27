from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import Settings
from app.llm.cache import get_cached_provider
from app.llm.exceptions import LoRAAdapterNotFoundError
from app.llm.lora_runtime import (
    AdapterCompatibilityError,
    AdapterInfo,
    active_adapter,
    set_active_adapter,
)


class AdapterAlreadyLoadedError(RuntimeError):
    """Raised when attempting to load an adapter that is already active."""


class AdapterNotLoadedError(RuntimeError):
    """Raised when attempting to unload an adapter when none are active."""


class UnsupportedAdapterFormatError(AdapterCompatibilityError):
    """Raised when the adapter file type is not recognised."""


class InvalidScalingError(ValueError):
    """Raised when a scaling factor falls outside supported bounds."""


@dataclass(slots=True)
class LoraRuntimeSnapshot:
    """Container describing the LoRA runtime state after an operation."""

    info: Optional[AdapterInfo]


class LoraRuntimeManager:
    """Coordinate loading and unloading LoRA adapters for the runtime."""

    def __init__(self, *, settings: Settings, provider=None) -> None:
        self._settings = settings
        self._provider = provider or get_cached_provider(settings)

    @property
    def settings(self) -> Settings:
        return self._settings

    def _resolve_provider(self):
        provider = self._provider
        ensure_model = getattr(provider, "ensure_model", None)
        if callable(ensure_model):
            ensure_model()
        ensure_ready = getattr(provider, "ensure_ready", None)
        if callable(ensure_ready):
            ensure_ready()
        return provider

    @staticmethod
    def _normalise_path(path: Path | str) -> Path:
        resolved = Path(path).expanduser().resolve()
        return resolved

    @staticmethod
    def _detect_format(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".gguf":
            return "gguf"
        if suffix in {".safetensors", ".bin"}:
            return "peft"
        return "unknown"

    @staticmethod
    def _validate_scaling(scaling: float | int) -> float:
        numeric = float(scaling)
        if not math.isfinite(numeric) or numeric <= 0.0 or numeric > 10.0:
            raise InvalidScalingError(f"Invalid scaling factor: {scaling}")
        return numeric

    def _build_info(self, path: Path, *, adapter_type: str, scaling: float) -> AdapterInfo:
        created_at = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
        return AdapterInfo(
            name=path.stem,
            base=self._settings.llm_model_name,
            adapter_type=adapter_type,
            seq_len=0,
            created_at=created_at,
            directory=path.parent,
            payload=path,
            manifest_path=path,
            scaling=scaling,
        )

    async def load_adapter(self, path: Path | str, scaling: float) -> LoraRuntimeSnapshot:
        resolved = self._normalise_path(path)
        if not resolved.is_file():
            raise LoRAAdapterNotFoundError(resolved)

        current = active_adapter()
        if current and current.payload == resolved:
            raise AdapterAlreadyLoadedError(f"Adapter {resolved} already active")

        provider = self._resolve_provider()
        adapter_type = self._detect_format(resolved)
        numeric_scaling = self._validate_scaling(scaling)

        if adapter_type == "unknown":
            raise UnsupportedAdapterFormatError(f"Unsupported adapter format for {resolved.name}")

        load_kwargs = {"scaling": numeric_scaling}

        if adapter_type == "gguf":
            load_fn = getattr(provider, "load_lora", None)
            if not callable(load_fn):
                raise AdapterCompatibilityError(
                    "Current LLM provider does not support GGUF adapters"
                )
            load_fn(resolved, **load_kwargs)
        else:
            load_fn = getattr(provider, "load_peft_adapter", None)
            if not callable(load_fn):
                raise AdapterCompatibilityError(
                    "Current LLM provider does not support PEFT adapters"
                )
            load_fn(resolved, **load_kwargs)

        info = self._build_info(resolved, adapter_type=adapter_type, scaling=numeric_scaling)
        set_active_adapter(info)
        return LoraRuntimeSnapshot(info=info)

    async def unload_adapter(self, path: Path | str | None = None) -> LoraRuntimeSnapshot:
        current = active_adapter()
        if current is None:
            raise AdapterNotLoadedError("No active LoRA adapter")

        if path is not None:
            resolved = self._normalise_path(path)
            if resolved != current.payload:
                raise AdapterNotLoadedError(f"Adapter {resolved} is not currently active")

        provider = self._resolve_provider()
        unload_fn = getattr(provider, "unload_lora", None)
        if callable(unload_fn):
            unload_fn()

        set_active_adapter(None)
        return LoraRuntimeSnapshot(info=None)


__all__ = [
    "AdapterAlreadyLoadedError",
    "AdapterNotLoadedError",
    "InvalidScalingError",
    "UnsupportedAdapterFormatError",
    "LoraRuntimeManager",
    "LoraRuntimeSnapshot",
]
