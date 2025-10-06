"""API endpoints for managing llama.cpp LoRA adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import require_admin_user
from app.core.deps import get_lora_manager
from app.llm.manager import (
    AdapterAlreadyLoadedError,
    AdapterNotLoadedError,
    InvalidScalingError,
    LlamaLoraManager,
    SCALING_MAX,
    SCALING_MIN,
    ensure_valid_scaling,
)
from app.models.lora import LoraLoadRequest, LoraStatusResponse, LoraUnloadRequest

router = APIRouter(prefix="/lora", tags=["lora"], dependencies=[Depends(require_admin_user)])

HTTP_NOT_FOUND = getattr(status, "HTTP_404_NOT_FOUND", 404)
HTTP_CONFLICT = getattr(status, "HTTP_409_CONFLICT", 409)
HTTP_UNPROCESSABLE_ENTITY = getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)
HTTP_SERVER_ERROR = getattr(status, "HTTP_500_INTERNAL_SERVER_ERROR", 500)


_MISSING = object()


def _extract_payload_value(payload: object, field: str) -> Any:
    """Return ``field`` from *payload* without mutating it."""

    if isinstance(payload, Mapping):
        try:
            return payload.get(field)
        except Exception:  # pragma: no cover - defensive guard
            return None

    try:
        value = getattr(payload, field)
    except AttributeError:
        value = _MISSING
    if value is not _MISSING:
        return value

    for accessor_name in ("model_dump", "dict"):
        accessor = getattr(payload, accessor_name, None)
        if not callable(accessor):
            continue
        try:
            data = accessor()
        except Exception:  # pragma: no cover - accessor raised unexpectedly
            continue
        if isinstance(data, Mapping):
            try:
                return data.get(field)
            except Exception:  # pragma: no cover - defensive guard
                continue
    return None


def _unwrap_scaling_candidate(candidate: object, *, _depth: int = 0) -> object:
    """Return the innermost scaling candidate from supported wrapper types."""

    if _depth > 4:
        return candidate

    if isinstance(candidate, SimpleNamespace):
        if hasattr(candidate, "scaling"):
            return _unwrap_scaling_candidate(getattr(candidate, "scaling"), _depth=_depth + 1)
        if hasattr(candidate, "value"):
            return _unwrap_scaling_candidate(getattr(candidate, "value"), _depth=_depth + 1)

    return candidate


def _coerce_scaling_value(candidate: object) -> float:
    """Validate and normalise the supplied scaling candidate."""

    unwrapped = _unwrap_scaling_candidate(candidate)

    try:
        numeric = float(unwrapped)
    except (TypeError, ValueError) as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    if not math.isfinite(numeric) or numeric <= SCALING_MIN or numeric > SCALING_MAX:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING")

    try:
        return ensure_valid_scaling(numeric)
    except ValueError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc


def _coerce_adapter_path(raw_path: object) -> Path:
    """Return ``raw_path`` as a normalised :class:`Path`."""

    if raw_path in {None, "", Ellipsis}:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING")

    try:
        return Path(str(raw_path)).expanduser()
    except (TypeError, ValueError) as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc


@router.post("/load", response_model=LoraStatusResponse)
async def load_lora_adapter(
    payload: LoraLoadRequest,
    manager: LlamaLoraManager = Depends(get_lora_manager),
) -> LoraStatusResponse:
    """Load a LoRA adapter into the configured llama.cpp instance."""

    raw_path = _extract_payload_value(payload, "path")
    raw_scaling = _extract_payload_value(payload, "scaling")

    adapter_path = _coerce_adapter_path(raw_path)
    scaling_value = _coerce_scaling_value(raw_scaling)

    try:
        adapter_status = await manager.load_adapter(adapter_path, scaling_value)
    except FileNotFoundError as exc:
        raise HTTPException(HTTP_NOT_FOUND, detail="ADAPTER_NOT_FOUND") from exc
    except AdapterAlreadyLoadedError as exc:
        raise HTTPException(HTTP_CONFLICT, detail="ADAPTER_ALREADY_LOADED") from exc
    except (InvalidScalingError, ValueError) as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc
    except Exception as exc:  # pragma: no cover - defensive guard for unexpected errors
        raise HTTPException(HTTP_SERVER_ERROR, detail="ADAPTER_LOAD_FAILED") from exc

    return LoraStatusResponse.from_status(adapter_status)


@router.post("/unload", response_model=LoraStatusResponse)
async def unload_lora_adapter(
    payload: LoraUnloadRequest,
    manager: LlamaLoraManager = Depends(get_lora_manager),
) -> LoraStatusResponse:
    """Unload the currently active LoRA adapter."""

    try:
        adapter_status = await manager.unload_adapter(payload.path)
    except AdapterNotLoadedError as exc:
        raise HTTPException(HTTP_CONFLICT, detail="ADAPTER_NOT_LOADED") from exc
    return LoraStatusResponse.from_status(adapter_status)


__all__ = ["router"]
