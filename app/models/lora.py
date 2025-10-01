"""Pydantic schemas for managing LoRA adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover - circular import guard for type checking
    from app.llm.manager import LoraStatus


class LoraBaseRequest(BaseModel):
    """Common payload data for LoRA adapter operations."""

    path: Path = Field(..., description="Filesystem path to the adapter file")
    scaling: float = Field(
        1.0,
        gt=0.0,
        le=10.0,
        description="Scaling factor applied to the adapter weights",
    )

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        if value in {None, "", Ellipsis}:
            raise ValueError("Adapter path must be provided")
        return Path(str(value)).expanduser()

    @field_validator("scaling")
    @classmethod
    def _validate_scaling(cls, value: float) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("Scaling factor must be a finite number")
        if float(value) <= 0.0:
            raise ValueError("Scaling factor must be greater than zero")
        if float(value) > 10.0:
            raise ValueError("Scaling factor must not exceed 10.0")
        return float(value)


class LoraLoadRequest(LoraBaseRequest):
    """Payload accepted by the LoRA load endpoint."""


class LoraUnloadRequest(LoraBaseRequest):
    """Payload accepted by the LoRA unload endpoint."""


class LoraStatusResponse(BaseModel):
    """Response describing the current LoRA adapter state."""

    loaded: bool = Field(..., description="Whether an adapter is currently active")
    path: str | None = Field(
        None, description="Absolute path of the active adapter if loaded"
    )
    scaling: float | None = Field(
        None, description="Scaling factor of the active adapter"
    )
    adapter: str | None = Field(
        None, description="Adapter name assigned inside llama.cpp"
    )

    @classmethod
    def from_status(cls, status: "LoraStatus") -> "LoraStatusResponse":
        path_value = str(status.path) if status.path is not None else None
        return cls(
            loaded=status.loaded,
            path=path_value,
            scaling=status.scaling,
            adapter=status.adapter_name,
        )


__all__ = [
    "LoraLoadRequest",
    "LoraStatusResponse",
    "LoraUnloadRequest",
]
