"""API endpoints for managing llama.cpp LoRA adapters."""

from __future__ import annotations

import math
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

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

    payload_dict: dict[str, Any]
    model_dump: Callable[[], dict[str, Any]] | None = getattr(payload, "model_dump", None)
    if callable(model_dump):
        payload_dict = model_dump()
    else:
        as_dict: Callable[[], dict[str, Any]] | None = getattr(payload, "dict", None)
        if callable(as_dict):
            payload_dict = as_dict()
        else:
            payload_dict = {
                "path": getattr(payload, "path", None),
                "scaling": getattr(payload, "scaling", None),
            }

    validator: Callable[[Any], Any] | None = getattr(LoraLoadRequest, "model_validate", None)
    try:
        payload_copy = (
            validator(payload_dict)
            if callable(validator)
            else LoraLoadRequest(**payload_dict)
        )
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    scaling_candidate = getattr(payload_copy, "scaling", None)
    try:
        scaling_value = float(scaling_candidate)
    except (TypeError, ValueError) as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    if not math.isfinite(scaling_value) or scaling_value <= 0:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING")

    try:
        scaling_value = ensure_valid_scaling(scaling_value)
    except ValueError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_ENTITY, detail="INVALID_SCALING") from exc

    try:
        adapter_status = await manager.load_adapter(
            getattr(payload_copy, "path"), scaling_value
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
