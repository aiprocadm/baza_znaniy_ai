"""Pydantic models used by LoRA administration endpoints."""

from __future__ import annotations

import math  # Standard library is required for runtime validators.
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationError

try:  # pragma: no cover - compatibility with pydantic stubs used in unit tests
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
except ImportError:  # pragma: no cover - pydantic v1/test stubs
    GetCoreSchemaHandler = Any  # type: ignore[misc,assignment]
    GetJsonSchemaHandler = Any  # type: ignore[misc,assignment]
from pydantic_core import PydanticCustomError, core_schema

if TYPE_CHECKING:  # pragma: no cover - circular import guard for type checking
    from app.llm.lora_runtime import AdapterInfo


class LoraAdapterName(str):
    """Validated adapter identifier that behaves like a string."""

    _MIN_LENGTH = 1
    _MAX_LENGTH = 128

    def __new__(cls, value: object = None, *, name: object | None = None) -> "LoraAdapterName":
        if value is None:
            if name is None:
                raise TypeError("LoraAdapterName requires a value")
            value = name
        elif name is not None:
            raise TypeError("Specify either 'value' or 'name', not both")

        try:
            validated = cls._validate_string(value)
        except PydanticCustomError as exc:
            raise ValidationError.from_exception_data(
                "LoraAdapterName",
                [
                    {
                        "type": exc.type,
                        "loc": ("name",),
                        "msg": exc.message,
                        "input": value,
                        "ctx": exc.context,
                    }
                ],
            ) from exc
        except ValueError as exc:
            raise ValidationError.from_exception_data(
                "LoraAdapterName",
                [
                    {
                        "type": "value_error",
                        "loc": ("name",),
                        "msg": str(exc),
                        "input": value,
                    }
                ],
            ) from exc
        return str.__new__(cls, validated)

    @property
    def name(self) -> str:
        return str(self)

    @classmethod
    def _validate_string(cls, candidate: object) -> str:
        string = str(candidate) if candidate is not None else ""
        string = string.strip()
        if len(string) < cls._MIN_LENGTH:
            raise PydanticCustomError(
                "string_too_short",
                "Adapter name must not be empty",
                {"min_length": cls._MIN_LENGTH},
            )
        if len(string) > cls._MAX_LENGTH:
            raise PydanticCustomError(
                "string_too_long",
                f"Adapter name must not exceed {cls._MAX_LENGTH} characters",
                {"max_length": cls._MAX_LENGTH},
            )
        return string

    @classmethod
    def _validate(cls, candidate: object) -> "LoraAdapterName":
        if isinstance(candidate, cls):
            return candidate
        return cls(cls._validate_string(candidate))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type[object], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, Any]:
        schema = {"type": "string", "minLength": cls._MIN_LENGTH, "maxLength": cls._MAX_LENGTH}
        schema["description"] = "Human readable adapter identifier"
        return schema


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

    @field_validator("path")
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


class LoraAdapterInfo(BaseModel):
    """Adapter metadata returned to API clients."""

    name: str
    base: str
    type: str
    seq_len: int
    created_at: str
    path: str
    scaling: float = Field(default=1.0, description="Scaling factor applied when loaded")

    @classmethod
    def from_runtime(cls, info: "AdapterInfo") -> "LoraAdapterInfo":
        return cls(
            name=info.name,
            base=info.base,
            type=info.format,
            seq_len=int(info.seq_len),
            created_at=info.created_at,
            path=str(info.payload),
            scaling=float(getattr(info, "scaling", 1.0)),
        )


class LoraLoadRequest(LoraBaseRequest):
    """Payload for dynamically loading a LoRA adapter from disk."""


class LoraAdapterNamePayload(BaseModel):
    """Request payload containing the adapter name."""

    name: LoraAdapterName


class LoraUnloadRequest(BaseModel):
    """Payload for unloading an active adapter."""

    model_config = ConfigDict(extra="ignore")

    path: Path | None = Field(default=None, description="Path of the adapter to unload")


class LoraStatusResponse(BaseModel):
    """Response describing the currently active adapter."""

    loaded: bool = Field(..., description="Whether an adapter is currently active")
    adapter: LoraAdapterInfo | None = Field(
        default=None, description="Metadata for the active adapter"
    )

    @classmethod
    def empty(cls) -> "LoraStatusResponse":
        return cls(loaded=False, adapter=None)

    @classmethod
    def from_runtime(cls, info: "AdapterInfo" | None) -> "LoraStatusResponse":
        if info is None:
            return cls.empty()
        return cls(loaded=True, adapter=LoraAdapterInfo.from_runtime(info))


__all__ = [
    "LoraAdapterName",
    "LoraAdapterInfo",
    "LoraAdapterNamePayload",
    "LoraLoadRequest",
    "LoraUnloadRequest",
    "LoraStatusResponse",
]
