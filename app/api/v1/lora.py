"""API endpoints for managing llama.cpp LoRA adapters."""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.auth import require_admin_user
from app.core.deps import get_lora_manager
from app.llm.manager import (
    AdapterAlreadyLoadedError,
    AdapterNotLoadedError,
    InvalidScalingError,
    LlamaLoraManager,
    ensure_valid_scaling,
)
from app.models.lora import LoraLoadRequest, LoraStatusResponse, LoraUnloadRequest

router = APIRouter(prefix="/lora", tags=["lora"], dependencies=[Depends(require_admin_user)])

HTTP_NOT_FOUND = getattr(status, "HTTP_404_NOT_FOUND", 404)
HTTP_CONFLICT = getattr(status, "HTTP_409_CONFLICT", 409)
HTTP_UNPROCESSABLE_ENTITY = getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422)
HTTP_SERVER_ERROR = getattr(status, "HTTP_500_INTERNAL_SERVER_ERROR", 500)


@router.post("/load", response_model=LoraStatusResponse)
async def load_lora_adapter(
    payload: LoraLoadRequest,
    manager: LlamaLoraManager = Depends(get_lora_manager),
) -> LoraStatusResponse:
    """Load a LoRA adapter into the configured llama.cpp instance."""

    try:
        scaling_candidate = ensure_valid_scaling(payload.scaling)
    except ValueError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    try:
        scaling_value = float(scaling_candidate)
    except (TypeError, ValueError) as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    try:
        if not math.isfinite(scaling_value):  # Defensive guard for unexpected NaN/Inf
            raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING")

        if scaling_value <= 0:  # Defensive guard for bypassed validation
            raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING")
    except (TypeError, ValueError) as exc:  # Defensive guard for non-numeric inputs
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    try:
        scaling_value = ensure_valid_scaling(scaling_value)
    except ValueError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    try:
        adapter_status = await manager.load_adapter(
            payload_copy.path, payload_copy.scaling
        )
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
