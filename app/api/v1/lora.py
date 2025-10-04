"""API endpoints for managing llama.cpp LoRA adapters."""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import require_admin_user
from app.core.deps import get_lora_manager
from app.llm.manager import (
    AdapterAlreadyLoadedError,
    AdapterNotLoadedError,
    LlamaLoraManager,
)
from app.models.lora import LoraLoadRequest, LoraStatusResponse, LoraUnloadRequest

router = APIRouter(prefix="/lora", tags=["lora"], dependencies=[Depends(require_admin_user)])

HTTP_NOT_FOUND = getattr(status, "HTTP_404_NOT_FOUND", 404)
HTTP_CONFLICT = getattr(status, "HTTP_409_CONFLICT", 409)
HTTP_SERVER_ERROR = getattr(status, "HTTP_500_INTERNAL_SERVER_ERROR", 500)

HTTP_UNPROCESSABLE = getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)


def _ensure_scaling_is_valid(raw_scaling: object) -> float:
    """Return *raw_scaling* as a validated ``float`` suitable for llama.cpp."""

    try:
        scaling_value = float(raw_scaling)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise HTTPException(HTTP_UNPROCESSABLE, detail="INVALID_SCALING") from exc

    if not math.isfinite(scaling_value):
        raise HTTPException(HTTP_UNPROCESSABLE, detail="INVALID_SCALING")
    if scaling_value <= 0.0 or scaling_value > 10.0:
        raise HTTPException(HTTP_UNPROCESSABLE, detail="INVALID_SCALING")
    return scaling_value


@router.post("/load", response_model=LoraStatusResponse)
async def load_lora_adapter(
    payload: LoraLoadRequest,
    manager: LlamaLoraManager = Depends(get_lora_manager),
) -> LoraStatusResponse:
    """Load a LoRA adapter into the configured llama.cpp instance."""

    scaling_value = _ensure_scaling_is_valid(payload.scaling)

    try:
        adapter_status = await manager.load_adapter(payload.path, scaling_value)
    except FileNotFoundError as exc:
        raise HTTPException(HTTP_NOT_FOUND, detail="ADAPTER_NOT_FOUND") from exc
    except AdapterAlreadyLoadedError as exc:
        raise HTTPException(HTTP_CONFLICT, detail="ADAPTER_ALREADY_LOADED") from exc
    except ValueError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE, detail="INVALID_SCALING") from exc
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
