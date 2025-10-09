"""Pydantic schemas for managing LoRA adapters."""

from __future__ import annotations

import math  # Standard library is required for runtime validators.
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover - circular import guard for type checking
    from app.llm.manager import LoraStatus


ScalingFactor = Annotated[
    float,
    Field(
        gt=0.0,
        le=10.0,
        description="Scaling factor applied to the adapter weights",
    ),
]


class LoraBaseRequest(BaseModel):
    """Common payload data for LoRA adapter operations."""

    path: Path = Field(..., description="Filesystem path to the adapter file")
    scaling: ScalingFactor = Field(default=1.0)

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - simple assignment
        object.__setattr__(self, "scaling", float(self.scaling))

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: object) -> Path:
        if value in {None, "", Ellipsis}:
            raise ValueError("Adapter path must be provided")
        return Path(str(value)).expanduser()

    @field_validator("scaling")
    @classmethod
    def _validate_scaling(cls, value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("Scaling factor must be a finite number") from None

        if not math.isfinite(numeric):
            raise ValueError("Scaling factor must be a finite number")
        if numeric <= 0.0:
            raise ValueError("Scaling factor must be greater than zero")
        if numeric > 10.0:
            raise ValueError("Scaling factor must not exceed 10.0")
        return numeric


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
