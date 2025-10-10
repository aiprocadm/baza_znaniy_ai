"""Administrative endpoints for managing LoRA adapters at runtime."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.status_codes import HTTP_UNPROCESSABLE_CONTENT
from app.core.auth import require_admin_user
from app.llm.lora_runtime import (
    AdapterCompatibilityError,
    RegistryError,
    active_adapter,
    list_adapters,
    load_adapter,
    unload_adapter,
)
from app.models.lora import LoraAdapterInfo, LoraAdapterNamePayload, LoraStatusResponse

router = APIRouter(prefix="/admin/lora", tags=["lora"], dependencies=[Depends(require_admin_user)])

HTTP_BAD_REQUEST = getattr(status, "HTTP_400_BAD_REQUEST", 400)
HTTP_NOT_FOUND = getattr(status, "HTTP_404_NOT_FOUND", 404)
HTTP_UNPROCESSABLE = HTTP_UNPROCESSABLE_CONTENT


@router.get("/list", response_model=list[LoraAdapterInfo])
async def list_registered_adapters() -> list[LoraAdapterInfo]:
    return [LoraAdapterInfo.from_runtime(info) for info in list_adapters()]


@router.get("/status", response_model=LoraStatusResponse)
async def get_adapter_status() -> LoraStatusResponse:
    return LoraStatusResponse.from_runtime(active_adapter())


@router.post("/load", response_model=LoraStatusResponse)
async def load_registered_adapter(payload: LoraAdapterNamePayload) -> LoraStatusResponse:
    try:
        info = load_adapter(payload.name)
    except RegistryError as exc:
        raise HTTPException(HTTP_NOT_FOUND, detail=str(exc)) from exc
    except AdapterCompatibilityError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(HTTP_BAD_REQUEST, detail=str(exc)) from exc
    return LoraStatusResponse.from_runtime(info)


@router.post("/unload", response_model=LoraStatusResponse)
async def unload_current_adapter(payload: LoraAdapterNamePayload | None = None) -> LoraStatusResponse:
    try:
        unload_adapter(payload.name if payload else None)
    except RegistryError as exc:
        raise HTTPException(HTTP_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(HTTP_BAD_REQUEST, detail=str(exc)) from exc
    return LoraStatusResponse.from_runtime(active_adapter())


__all__ = ["router"]
