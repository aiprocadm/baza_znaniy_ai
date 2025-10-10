"""Runtime LoRA adapter management endpoints."""

from __future__ import annotations

import math
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.status_codes import HTTP_UNPROCESSABLE_CONTENT
from app.core.config import get_settings
from app.llm.exceptions import LoRAAdapterNotFoundError
from app.llm.lora_runtime import AdapterCompatibilityError, active_adapter
from app.models.lora import LoraLoadRequest, LoraStatusResponse, LoraUnloadRequest
from app.services.lora_manager import (
    AdapterAlreadyLoadedError,
    AdapterNotLoadedError,
    InvalidScalingError,
    LoraRuntimeManager,
    UnsupportedAdapterFormatError,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/lora", tags=["lora"])


def get_lora_manager() -> LoraRuntimeManager:
    settings = get_settings()
    return LoraRuntimeManager(settings=settings)


def ensure_valid_scaling(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise HTTPException(HTTP_UNPROCESSABLE_CONTENT, detail="INVALID_SCALING") from None

    if math.isnan(numeric) or math.isinf(numeric):
        raise HTTPException(HTTP_UNPROCESSABLE_CONTENT, detail="INVALID_SCALING")
    if numeric <= 0.0 or numeric > 10.0:
        raise HTTPException(HTTP_UNPROCESSABLE_CONTENT, detail="INVALID_SCALING")
    return numeric


@router.post("/load", response_model=LoraStatusResponse)
async def load_lora_adapter(
    payload: LoraLoadRequest,
    manager: LoraRuntimeManager = Depends(get_lora_manager),
) -> LoraStatusResponse:
    scaling = ensure_valid_scaling(payload.scaling)
    try:
        snapshot = await manager.load_adapter(payload.path, scaling)
    except LoRAAdapterNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="ADAPTER_NOT_FOUND") from exc
    except AdapterAlreadyLoadedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="ADAPTER_ALREADY_LOADED") from exc
    except InvalidScalingError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_CONTENT, detail="INVALID_SCALING") from exc
    except UnsupportedAdapterFormatError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except AdapterCompatibilityError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Failed to load LoRA adapter from %s", payload.path)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="FAILED_TO_LOAD_ADAPTER") from exc

    return LoraStatusResponse.from_runtime(snapshot.info)


@router.post("/unload", response_model=LoraStatusResponse)
async def unload_lora_adapter(
    payload: LoraUnloadRequest | None = None,
    manager: LoraRuntimeManager = Depends(get_lora_manager),
) -> LoraStatusResponse:
    path = payload.path if payload else None
    try:
        await manager.unload_adapter(path)
    except AdapterNotLoadedError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="ADAPTER_NOT_LOADED") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Failed to unload LoRA adapter")
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="FAILED_TO_UNLOAD_ADAPTER") from exc

    return LoraStatusResponse.from_runtime(active_adapter())


__all__ = [
    "router",
    "get_lora_manager",
    "ensure_valid_scaling",
    "load_lora_adapter",
    "unload_lora_adapter",
    "InvalidScalingError",
]
