"""Pydantic models used by LoRA administration endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover - circular import guard for type checking
    from app.llm.lora_runtime import AdapterInfo


class LoraAdapterName(BaseModel):
    """Payload describing an adapter identified by name."""

    name: str = Field(..., min_length=1, description="Registry name of the adapter")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Adapter name must not be empty")
        return cleaned


class LoraAdapterInfo(BaseModel):
    """Adapter metadata returned to API clients."""

    name: str
    base: str
    type: str
    seq_len: int
    created_at: str
    path: str

    @classmethod
    def from_runtime(cls, info: "AdapterInfo") -> "LoraAdapterInfo":
        return cls(
            name=info.name,
            base=info.base,
            type=info.format,
            seq_len=int(info.seq_len),
            created_at=info.created_at,
            path=str(info.payload),
        )


class LoraStatusResponse(BaseModel):
    """Response describing the currently active adapter."""

    loaded: bool = Field(..., description="Whether an adapter is currently active")
    adapter: LoraAdapterInfo | None = Field(default=None, description="Metadata for the active adapter")

    @classmethod
    def empty(cls) -> "LoraStatusResponse":
        return cls(loaded=False, adapter=None)

    @classmethod
    def from_runtime(cls, info: "AdapterInfo" | None) -> "LoraStatusResponse":
        if info is None:
            return cls.empty()
        return cls(loaded=True, adapter=LoraAdapterInfo.from_runtime(info))


__all__ = ["LoraAdapterName", "LoraAdapterInfo", "LoraStatusResponse"]
