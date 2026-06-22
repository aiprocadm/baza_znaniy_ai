"""Concurrency-safe helpers for managing llama.cpp LoRA adapters."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class LlamaSettingsProtocol(Protocol):
    """Protocol describing the minimal configuration needed by the manager."""

    llm_model_name: str
    llama_cpp_model_path: Path | str | None


def _ensure_llm_package_exports() -> None:
    """Populate the ``app.llm`` package when tests replace it with a stub."""

    import sys

    package = sys.modules.get("app.llm")
    if package is None:
        return
    try:
        from . import cache as _cache
        from .providers import LLMProvider
    except Exception:  # pragma: no cover - optional dependencies may be missing
        return

    llama_cpp_provider: type[object] | None
    try:
        from .llama_cpp_provider import LlamaCppProvider as _LlamaCppProvider
    except ImportError as exc:  # pragma: no cover - Settings may be absent in tests
        missing = getattr(exc, "name", None)
        if missing in {"Settings", "app.core.config"} or "Settings" in str(exc):
            llama_cpp_provider = None
        else:
            raise
    except Exception:  # pragma: no cover - optional dependency errors
        return
    else:
        llama_cpp_provider = _LlamaCppProvider

    config_module = sys.modules.get("app.core.config")
    if config_module is not None and not hasattr(config_module, "get_settings"):
        settings_cls = getattr(config_module, "Settings", object)

        def _stub_get_settings():  # pragma: no cover - simple compatibility shim
            return settings_cls() if callable(settings_cls) else settings_cls

        def _cache_clear():  # pragma: no cover - compatibility helper
            return None

        _stub_get_settings.cache_clear = _cache_clear  # type: ignore[attr-defined]
        setattr(config_module, "get_settings", _stub_get_settings)

    exports = {
        "LLMProvider": LLMProvider,
        "get_llm_provider": _cache.get_llm_provider,
        "get_cached_provider": _cache.get_cached_provider,
        "reset_provider_cache": _cache.reset_provider_cache,
        "get_llm_client": _cache.get_llm_client,
        "LLMProviderError": _cache.LLMProviderError,
        "ModelNotFoundError": _cache.ModelNotFoundError,
        "ModelNotReadyError": _cache.ModelNotReadyError,
        "LoRAAdapterNotFoundError": _cache.LoRAAdapterNotFoundError,
    }

    if llama_cpp_provider is not None:
        exports["LlamaCppProvider"] = llama_cpp_provider

    for name, value in exports.items():
        setattr(package, name, value)

    existing = set(getattr(package, "__all__", []))
    # setattr avoids mypy's "Cannot assign to __all__ of a module" on the
    # ModuleType stub; identical to ``package.__all__ = ...`` at runtime.
    setattr(package, "__all__", sorted(existing | set(exports)))


_ensure_llm_package_exports()


def _unwrap_scaling_candidate(candidate: object, *, _depth: int = 0) -> object:
    """Return the innermost value stored in supported wrapper types."""

    if _depth > 4:  # pragma: no cover - recursion guard for pathological inputs
        return candidate

    if isinstance(candidate, SimpleNamespace):
        if hasattr(candidate, "scaling"):
            return _unwrap_scaling_candidate(getattr(candidate, "scaling"), _depth=_depth + 1)
        if hasattr(candidate, "value"):
            return _unwrap_scaling_candidate(getattr(candidate, "value"), _depth=_depth + 1)

    return candidate


def ensure_valid_scaling(scaling: object) -> float:
    """Return *scaling* as a float if it is finite and within supported bounds."""

    candidate = _unwrap_scaling_candidate(scaling)

    try:
        value = float(candidate)  # type: ignore[arg-type]  # candidate is intentionally object; a non-numeric raises TypeError, caught below
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
        raise ValueError("Scaling factor must be a numeric value") from exc

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Scaling factor must be a finite number greater than zero")
    if value > 10.0:
        raise ValueError("Scaling factor must not exceed 10.0")

    return value


@dataclass(slots=True)
class LoraStatus:
    """Snapshot describing the currently active adapter."""

    loaded: bool
    path: Path | None = None
    scaling: float | None = None
    adapter_name: str | None = None


@dataclass(slots=True)
class _AdapterState:
    """Internal representation of the loaded adapter."""

    path: Path
    scaling: float
    adapter_name: str


class LoraManagerError(RuntimeError):
    """Base exception raised for LoRA manager errors."""


class AdapterAlreadyLoadedError(LoraManagerError):
    """Raised when attempting to load an adapter that is already active."""


class AdapterNotLoadedError(LoraManagerError):
    """Raised when attempting to operate on a missing adapter."""


SCALING_MIN = 0.0
SCALING_MAX = 10.0


class InvalidScalingError(LoraManagerError):
    """Raised when an invalid scaling factor is supplied."""


class LlamaLoraManager:
    """Manage LoRA adapters for a ``llama_cpp.Llama`` instance."""

    def __init__(
        self,
        settings: LlamaSettingsProtocol,
        llama_factory: Callable[[], object] | None = None,
    ) -> None:
        self._settings = settings
        self._llama_factory = llama_factory or self._build_default_factory(settings)
        self._lock = asyncio.Lock()
        self._llama: object | None = None
        self._adapter: _AdapterState | None = None

    @staticmethod
    def _resolve_settings_class() -> type[object] | None:
        """Attempt to import the ``Settings`` class lazily."""

        try:  # pragma: no cover - import failures handled in tests
            from app.core.config import Settings as SettingsClass  # type: ignore
        except Exception:  # pragma: no cover - optional dependency missing
            return None
        return SettingsClass

    @staticmethod
    def _require_setting(settings: LlamaSettingsProtocol, attribute: str) -> Any:
        """Return ``attribute`` from *settings* or raise a helpful ``AttributeError``."""

        try:
            return getattr(settings, attribute)
        except AttributeError as exc:  # pragma: no cover - exceptional path
            settings_cls = LlamaLoraManager._resolve_settings_class()
            expected = (
                settings_cls.__name__
                if settings_cls is not None
                else "an object matching the Settings interface"
            )
            raise AttributeError(
                "LlamaLoraManager requires settings with attribute "
                f"'{attribute}', but received {type(settings).__name__!r} without it. "
                f"Provide {expected}."
            ) from exc

    @staticmethod
    def _build_default_factory(settings: LlamaSettingsProtocol) -> Callable[[], object]:
        """Return a callable constructing a new ``llama_cpp.Llama`` instance."""

        def factory() -> object:
            override = getattr(settings, "llama_cpp_model_path", None)
            model_reference = override or LlamaLoraManager._require_setting(
                settings, "llm_model_name"
            )

            from llama_cpp import Llama  # imported lazily to keep dependency optional

            return Llama(model_path=str(model_reference))

        return factory

    @staticmethod
    def _normalise_path(path: Path) -> Path:
        candidate = Path(path).expanduser()
        try:
            return candidate.resolve()
        except FileNotFoundError:
            return candidate

    @staticmethod
    def _adapter_name_from_path(path: Path) -> str:
        stem = path.stem or "adapter"
        sanitized = stem.replace(" ", "_")
        return f"lora::{sanitized}"

    def _current_status(self) -> LoraStatus:
        if self._adapter is None:
            return LoraStatus(loaded=False)
        return LoraStatus(
            loaded=True,
            path=self._adapter.path,
            scaling=self._adapter.scaling,
            adapter_name=self._adapter.adapter_name,
        )

    async def _rebuild_llama(self) -> object:
        llama = await asyncio.to_thread(self._llama_factory)
        self._llama = llama
        self._adapter = None
        return llama

    @staticmethod
    def _validate_scaling(scaling: float | int) -> float:
        """Return a validated scaling factor within ``(0, 10]``."""

        try:
            scaling_value = float(scaling)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise InvalidScalingError("Scaling must be a finite float") from exc

        if math.isnan(scaling_value) or not math.isfinite(scaling_value):
            raise InvalidScalingError("Scaling must be finite")

        if scaling_value <= 0.0 or scaling_value > 10.0:
            raise InvalidScalingError("Scaling must be within (0, 10]")

        return scaling_value

    async def load_adapter(self, path: Path, scaling: float) -> LoraStatus:
        """Load a LoRA adapter with *scaling* and make it active."""

        scaling_value = ensure_valid_scaling(scaling)

        candidate = self._normalise_path(path)
        if not candidate.is_file():
            raise FileNotFoundError(str(candidate))

        async with self._lock:
            if self._adapter and candidate == self._adapter.path:
                raise AdapterAlreadyLoadedError(str(candidate))

            llama = await self._rebuild_llama()
            adapter_name = self._adapter_name_from_path(candidate)

            if hasattr(llama, "load_adapter"):
                llama.load_adapter(str(candidate), adapter_name=adapter_name, scale=scaling_value)
            if hasattr(llama, "set_adapter"):
                llama.set_adapter(adapter_name)

            self._adapter = _AdapterState(
                path=candidate,
                scaling=scaling_value,
                adapter_name=adapter_name,
            )
            return self._current_status()

    async def unload_adapter(self, expected_path: Path | None = None) -> LoraStatus:
        """Unload the currently active adapter, optionally verifying *expected_path*."""

        async with self._lock:
            if self._adapter is None:
                raise AdapterNotLoadedError("No adapter is currently loaded")

            if expected_path is not None:
                candidate = self._normalise_path(expected_path)
                if candidate != self._adapter.path:
                    raise AdapterNotLoadedError("A different adapter is active")

            adapter_name = self._adapter.adapter_name
            llama = self._llama
            try:
                if llama is not None and hasattr(llama, "unload_adapter"):
                    llama.unload_adapter(adapter_name)
            finally:
                await self._rebuild_llama()
            return self._current_status()

    async def get_status(self) -> LoraStatus:
        """Return the current adapter status."""

        async with self._lock:
            return self._current_status()


__all__ = [
    "AdapterAlreadyLoadedError",
    "AdapterNotLoadedError",
    "ensure_valid_scaling",
    "InvalidScalingError",
    "LlamaLoraManager",
    "LoraStatus",
]
