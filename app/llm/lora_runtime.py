"""Runtime helpers for managing LoRA adapters and registry metadata."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.core.config import get_settings
from app.llm.cache import get_cached_provider

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AdapterInfo:
    """Metadata describing a registered LoRA adapter."""

    name: str
    base: str
    adapter_type: str
    seq_len: int
    created_at: str
    directory: Path
    payload: Path
    manifest_path: Path
    scaling: float = 1.0

    @property
    def format(self) -> str:
        adapter_type = self.adapter_type.lower()
        if adapter_type in {"gguf", "llama.cpp"}:
            return "gguf"
        if adapter_type in {"peft", "hf", "transformers"}:
            return "peft"
        return adapter_type

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "base": self.base,
            "type": self.adapter_type,
            "seq_len": self.seq_len,
            "created_at": self.created_at,
            "path": str(self.payload),
            "scaling": float(self.scaling),
        }

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        base: str,
        adapter_type: str | None = None,
        scaling: float = 1.0,
    ) -> "AdapterInfo":
        adapter_type = (adapter_type or path.suffix.lstrip(".")).lower()
        if adapter_type in {"", "gguf", "llama", "llama.cpp"}:
            adapter_type = "gguf"
        elif adapter_type in {"safetensors", "peft", "adapter_model", "bin"}:
            adapter_type = "peft"
        created_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        return cls(
            name=path.stem,
            base=base,
            adapter_type=adapter_type,
            seq_len=0,
            created_at=created_at,
            directory=path.parent,
            payload=path,
            manifest_path=path,
            scaling=float(scaling),
        )


_ACTIVE_ADAPTER: AdapterInfo | None = None


class RegistryError(RuntimeError):
    """Raised when the adapter registry contains invalid metadata."""


class AdapterCompatibilityError(RuntimeError):
    """Raised when an adapter is incompatible with the configured runtime."""


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Manifest {path} contains invalid JSON: {exc}") from exc


def _discover_payload(directory: Path, adapter_type: str) -> Path:
    adapter_type = adapter_type.lower()
    if adapter_type in {"gguf", "llama.cpp"}:
        candidates = sorted(directory.glob("*.gguf"))
    else:
        candidates = sorted(directory.glob("*.safetensors"))
        if not candidates:
            candidates = sorted(directory.glob("adapter_model.bin"))
    if not candidates:
        raise RegistryError(f"No adapter payload found in {directory}")
    return candidates[0]


def _iter_manifests(registry_dir: Path) -> Iterable[AdapterInfo]:
    for entry in sorted(registry_dir.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_manifest(manifest_path)
        try:
            info = AdapterInfo(
                name=str(manifest["name"]),
                base=str(manifest["base"]),
                adapter_type=str(manifest.get("type", "peft")),
                seq_len=int(manifest.get("seq_len", 0)),
                created_at=str(manifest.get("created_at", "")),
                directory=entry,
                payload=_discover_payload(entry, str(manifest.get("type", "peft"))),
                manifest_path=manifest_path,
                scaling=float(manifest.get("scaling", 1.0)),
            )
        except KeyError as exc:
            raise RegistryError(f"Manifest {manifest_path} missing required field: {exc}") from exc
        except Exception as exc:
            raise RegistryError(f"Failed to parse manifest {manifest_path}: {exc}") from exc
        yield info


def list_adapters() -> list[AdapterInfo]:
    """Return adapter metadata registered in the filesystem registry."""

    settings = get_settings()
    registry = settings.lora_registry_path
    if not registry.exists():
        registry.mkdir(parents=True, exist_ok=True)
    adapters = list(_iter_manifests(registry))
    if not adapters:
        LOGGER.debug("LoRA registry %s is empty", registry)
    return adapters


def _ensure_compatible(adapter: AdapterInfo) -> None:
    settings = get_settings()
    configured_base = settings.llm_model_name
    if adapter.base and adapter.base != configured_base:
        raise AdapterCompatibilityError(
            f"Adapter {adapter.name} targets base model {adapter.base}, expected {configured_base}",
        )


def load_adapter(name: str) -> AdapterInfo:
    """Activate the adapter with *name* from the registry."""

    adapters = list_adapters()
    try:
        candidate = next(adapter for adapter in adapters if adapter.name == name)
    except StopIteration as exc:
        raise RegistryError(f"Adapter {name!r} not found in registry") from exc

    _ensure_compatible(candidate)

    settings = get_settings()
    provider = get_cached_provider(settings)
    ensure_model = getattr(provider, "ensure_model", None)
    if callable(ensure_model):
        ensure_model()

    if candidate.format == "gguf":
        load_fn = getattr(provider, "load_lora", None)
        if not callable(load_fn):
            raise AdapterCompatibilityError("Current LLM provider does not support GGUF adapters")
        LOGGER.info("Loading GGUF LoRA adapter: %s", candidate.payload)
        load_fn(candidate.payload)
    elif candidate.format == "peft":
        load_fn = getattr(provider, "load_peft_adapter", None)
        if not callable(load_fn):
            raise AdapterCompatibilityError(
                "Current LLM provider does not support PEFT adapters; convert to GGUF first",
            )
        LOGGER.info("Loading PEFT LoRA adapter: %s", candidate.payload)
        load_fn(candidate.payload)
    else:
        raise AdapterCompatibilityError(f"Unsupported adapter format: {candidate.format}")

    scaling_value = getattr(settings, "lora_scaling", None)
    if scaling_value is None:
        numeric_scaling = float(candidate.scaling)
    else:
        try:
            numeric_scaling = float(scaling_value)
        except (TypeError, ValueError) as exc:
            raise AdapterCompatibilityError("Invalid LoRA scaling factor configured") from exc
        if not math.isfinite(numeric_scaling):
            raise AdapterCompatibilityError("Invalid LoRA scaling factor configured")

    candidate.scaling = numeric_scaling
    set_active_adapter(candidate)
    return candidate


def unload_adapter(name: str | None = None) -> None:
    """Deactivate the currently loaded adapter if one is active."""

    global _ACTIVE_ADAPTER
    if _ACTIVE_ADAPTER is None:
        return
    if name is not None and _ACTIVE_ADAPTER.name != name:
        raise RegistryError(f"Adapter {name} is not active")

    provider = get_cached_provider(get_settings())
    unload_fn = getattr(provider, "unload_lora", None)
    if callable(unload_fn):
        LOGGER.info("Unloading LoRA adapter: %s", _ACTIVE_ADAPTER.payload)
        unload_fn()
    set_active_adapter(None)


def active_adapter() -> AdapterInfo | None:
    return _ACTIVE_ADAPTER


def set_active_adapter(info: AdapterInfo | None) -> None:
    global _ACTIVE_ADAPTER
    _ACTIVE_ADAPTER = info


__all__ = [
    "AdapterInfo",
    "AdapterCompatibilityError",
    "RegistryError",
    "list_adapters",
    "load_adapter",
    "unload_adapter",
    "active_adapter",
    "set_active_adapter",
]
