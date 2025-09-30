"""Lightweight subset of the Pydantic API used in tests."""

from __future__ import annotations

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, get_type_hints


def _load_real_module() -> object | None:
    stub_dir = Path(__file__).resolve().parent
    for entry in sys.path:
        try:
            if Path(entry).resolve() == stub_dir:
                continue
        except OSError:  # pragma: no cover - non-filesystem entries
            continue
        candidate = Path(entry) / "pydantic" / "__init__.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("_real_pydantic", candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                loader = spec.loader
                assert loader is not None  # narrow type
                sys.modules.setdefault("pydantic", module)
                loader.exec_module(module)
                return module
    return None


_EXISTING = sys.modules.get("pydantic")
if _EXISTING is not None and getattr(_EXISTING, "__file__", "") != __file__:
    _REAL = _EXISTING
else:
    _REAL = _load_real_module()

if _REAL is not None:  # pragma: no cover - exercised when real dependency available
    globals().update(_REAL.__dict__)
else:

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
            raw_annotations = getattr(self, "__annotations__", {})
            resolved_annotations = get_type_hints(self.__class__, include_extras=True)

            for name, annotation in raw_annotations.items():
                resolved = resolved_annotations.get(name, annotation)

                if name in data:
                    value = data[name]
                else:
                    value = self._default_for(name)

                if annotation is Path and not isinstance(value, Path):
                    value = Path(value)

                target = resolved
                if target is Path and isinstance(value, str):
                    value = Path(value)
                elif target is bool and isinstance(value, str):
                    value = value.lower() in {"1", "true", "yes", "on"}
                elif target is int and isinstance(value, str):
                    value = int(value)

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


    class ValidationError(Exception):
        """Compatibility exception used by third-party clients."""

        pass


    class EmailStr(str):
        """Placeholder email type used by sqlmodel during tests."""

        @classmethod
        def __get_validators__(cls):  # pragma: no cover - compatibility shim
            yield cls

        def __new__(cls, value: Any):
            return str.__new__(cls, str(value))


    __all__ = [
        "AliasChoices",
        "BaseModel",
        "EmailStr",
        "Field",
        "ValidationError",
    ]
