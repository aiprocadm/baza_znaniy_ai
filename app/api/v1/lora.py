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


    scaling = float(payload.scaling)
    if not math.isfinite(scaling) or scaling <= 0.0 or scaling > 10.0:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING")


    scaling = float(payload.scaling)
    if math.isnan(scaling) or scaling <= 0 or scaling > 10:
        raise HTTPException(status_code=422, detail="INVALID_SCALING")

    scaling = payload.scaling
    # Defensive guard: although the request model validates scaling, runtime callers
    # may bypass Pydantic and supply unexpected values. We normalise to ``float`` and
    # ensure the number is finite and strictly positive before invoking llama.cpp.
    try:
        scaling_value = float(scaling)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            HTTP_UNPROCESSABLE_ENTITY,
            detail="Scaling factor must be a finite number greater than zero.",
        ) from exc

    if not math.isfinite(scaling_value) or scaling_value <= 0.0:
        raise HTTPException(
            HTTP_UNPROCESSABLE_ENTITY,
            detail="Scaling factor must be a finite number greater than zero.",
        )



    try:
        adapter_status = await manager.load_adapter(payload.path, scaling_value)
    except FileNotFoundError as exc:
        raise HTTPException(HTTP_NOT_FOUND, detail="ADAPTER_NOT_FOUND") from exc
    except AdapterAlreadyLoadedError as exc:
        raise HTTPException(HTTP_CONFLICT, detail="ADAPTER_ALREADY_LOADED") from exc
    except InvalidScalingError as exc:
        raise HTTPException(status_code=422, detail="INVALID_SCALING") from exc
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
