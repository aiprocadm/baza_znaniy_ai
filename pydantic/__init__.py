"""Lightweight subset of the Pydantic API used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar


@dataclass
class FieldInfo:
    default: Any = ...
    default_factory: Optional[Callable[[], Any]] = None
    metadata: Dict[str, Any] | None = None
    alias: Any | None = None


class AliasChoices(tuple):
    """Minimal stand-in for :class:`pydantic.alias_generators.AliasChoices`."""

    def __new__(cls, *choices: str) -> "AliasChoices":
        if not choices:
            raise ValueError("AliasChoices requires at least one value")
        normalised: Tuple[str, ...] = tuple(str(choice) for choice in choices)
        return super().__new__(cls, normalised)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        values = ", ".join(repr(choice) for choice in self)
        return f"AliasChoices({values})"


def Field(
    default: Any = ...,
    *,
    default_factory: Optional[Callable[[], Any]] = None,
    alias: Any | None = None,
    **metadata: Any,
) -> FieldInfo:
    return FieldInfo(
        default=default,
        default_factory=default_factory,
        metadata=metadata,
        alias=alias,
    )


T = TypeVar("T", bound="BaseModel")


class BaseModel:
    """Minimal model implementation supporting validation and dumping."""

    def __init__(self, **data: Any) -> None:
        annotations = getattr(self, "__annotations__", {})
        for name, annotation in annotations.items():
            if name in data:
                value = data[name]
            else:
                value = self._default_for(name)
            if annotation is Path and not isinstance(value, Path):
                value = Path(value)
            setattr(self, name, value)

    @classmethod
    def _default_for(cls, name: str) -> Any:
        field = getattr(cls, name, ...)
        if isinstance(field, FieldInfo):
            if field.default is ... and field.default_factory is None:
                raise ValueError(f"Field '{name}' is required")
            if field.default_factory is not None:
                return field.default_factory()
            return field.default
        if field is ...:
            raise ValueError(f"Field '{name}' is required")
        return field

    def model_dump(self, mode: str | None = None) -> Dict[str, Any]:
        annotations = getattr(self, "__annotations__", {})
        return {name: getattr(self, name) for name in annotations}

    @classmethod
    def model_validate(cls: Type[T], data: Any) -> T:
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError("model_validate expects a mapping")
        return cls(**data)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        values = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.__annotations__)
        return f"{self.__class__.__name__}({values})"


__all__ = [
    "AliasChoices",
    "BaseModel",
    "Field",
]
