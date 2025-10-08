"""Lightweight subset of the Pydantic API used in tests."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


def _load_real_module() -> object | None:
    stub_dir = Path(__file__).resolve().parent
    for entry in sys.path:
        try:
            if Path(entry).resolve() == stub_dir:
                continue
        except OSError:  # pragma: no cover - non-filesystem entries
            continue
        candidate = Path(entry) / "pydantic" / "__init__.py"
        try:
            if candidate.resolve() == Path(__file__).resolve():
                continue
        except OSError:  # pragma: no cover - non-resolvable paths
            continue
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


_PREFER_LOCAL_IMPLEMENTATION = __name__.startswith("tests.stubs.")

if _PREFER_LOCAL_IMPLEMENTATION:
    _REAL = None
else:
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
        metadata: Dict[str, Any] = dataclass_field(default_factory=dict)
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
        meta = dict(metadata)
        alias_value = alias
        if alias_value is None:
            alias_value = meta.get("alias")
        if alias_value is None and "validation_alias" in meta:
            alias_value = meta["validation_alias"]
        return FieldInfo(
            default=default,
            default_factory=default_factory,
            metadata=meta,
            alias=alias_value,
        )


    T = TypeVar("T", bound="BaseModel")


    class BaseModel:
        """Minimal model implementation supporting validation and dumping."""

        def __init__(self, **data: Any) -> None:
            raw_annotations = getattr(self, "__annotations__", {})
            module = sys.modules.get(self.__class__.__module__)
            module_ns = getattr(module, "__dict__", {}) if module is not None else {}
            resolved_annotations = get_type_hints(
                self.__class__,
                globalns=module_ns,
                localns=module_ns,
                include_extras=True,
            )

            values: Dict[str, Any] = {}
            provided_fields: set[str] = set()
            validator_store = self._get_field_validator_store()

            for name, annotation in raw_annotations.items():
                resolved = resolved_annotations.get(name, annotation)

                value, provided = self._extract_value(name, data)
                if provided:
                    provided_fields.add(name)
                else:
                    value = self._default_for(name)

                before_validators = validator_store["before"].get(name, [])
                value = self._run_field_validators(before_validators, name, value)

                field_info = self._field_info_for(name)
                coerced = self._coerce_value(name, value, resolved, field_info)
                if annotation is Path and not isinstance(coerced, Path):
                    coerced = Path(coerced)

                after_validators = validator_store["after"].get(name, [])
                coerced = self._run_field_validators(after_validators, name, coerced)

                coerced = self._apply_field_constraints(name, coerced, field_info)

                values[name] = coerced

            for name, value in values.items():
                setattr(self, name, value)

            self.__pydantic_fields_set__ = set(provided_fields)

        @classmethod
        def _field_info_for(cls, name: str) -> FieldInfo | None:
            field = getattr(cls, name, None)
            if isinstance(field, FieldInfo):
                return field
            return None

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

        @classmethod
        def _aliases_for(cls, field: Any) -> Tuple[str, ...]:
            if not isinstance(field, FieldInfo):
                return tuple()

            def _collect(value: Any, bucket: list[str]) -> None:
                if value is None:
                    return
                if isinstance(value, AliasChoices):
                    bucket.extend(str(item) for item in value)
                elif isinstance(value, (tuple, list, set, frozenset)):
                    bucket.extend(str(item) for item in value)
                else:
                    bucket.append(str(value))

            aliases: list[str] = []
            _collect(field.alias, aliases)
            for key in ("alias", "validation_alias"):
                if key in field.metadata:
                    _collect(field.metadata[key], aliases)
            # Preserve order while removing duplicates
            seen: Dict[str, None] = {}
            for alias in aliases:
                if alias not in seen:
                    seen[alias] = None
            return tuple(seen.keys())

        @classmethod
        def _resolve_field_name(cls, key: Any) -> str:
            annotations = getattr(cls, "__annotations__", {})
            if key in annotations:
                return key  # type: ignore[return-value]
            key_text = str(key)
            if key_text in annotations:
                return key_text
            for name in annotations:
                field = getattr(cls, name, None)
                if key_text in cls._aliases_for(field):
                    return name
            return key_text

        def _extract_value(self, name: str, data: Dict[str, Any]) -> Tuple[Any, bool]:
            if name in data:
                return data[name], True

            field = getattr(self.__class__, name, None)
            for alias in self._aliases_for(field):
                if alias in data:
                    return data[alias], True
            return None, False

        def _coerce_value(
            self, name: str, value: Any, target: Any, field: FieldInfo | None
        ) -> Any:
            if value is None:
                coerced = None
            else:
                origin = get_origin(target)
                if origin is Annotated:
                    args = get_args(target)
                    if args:
                        coerced = self._coerce_value(name, value, args[0], field)
                    else:
                        coerced = value
                elif origin is list:
                    args = get_args(target)
                    item_type = args[0] if args else Any
                    coerced = self._coerce_list(value, item_type)
                elif origin is Union:
                    coerced = self._coerce_union(name, value, get_args(target), field)
                else:
                    coerced = self._coerce_simple(value, target)

            return self._apply_field_constraints(name, coerced, field)

        def _coerce_simple(self, value: Any, target: Any) -> Any:
            if value is None or target is None:
                return value

            if target is Path:
                if isinstance(value, Path):
                    return value
                return Path(value)
            if target is bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"1", "true", "yes", "on"}:
                        return True
                    if lowered in {"0", "false", "no", "off"}:
                        return False
                if isinstance(value, (int, float)):
                    return bool(value)
                return bool(value)
            if target is int:
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
                if isinstance(value, (float, str)):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return value
            if target is float:
                if isinstance(value, float):
                    return value
                if isinstance(value, (int, str)):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return value
            if target is str:
                return str(value)
            return value

        def _coerce_list(self, value: Any, item_type: Any) -> list[Any]:
            if isinstance(value, str) and item_type is str:
                items = [item.strip() for item in value.split(",")]
                return [item for item in items if item]
            if isinstance(value, (list, tuple, set)):
                return [self._coerce_simple(item, item_type) for item in value]
            return [self._coerce_simple(value, item_type)]

        def _coerce_union(
            self,
            name: str,
            value: Any,
            args: Tuple[Any, ...],
            field: FieldInfo | None,
        ) -> Any:
            for arg in args:
                if arg is type(None) and value is None:
                    return None
            for arg in args:
                if arg is type(None):
                    continue
                converted = self._coerce_value(name, value, arg, field)
                if converted is not value:
                    return converted
                try:
                    if isinstance(converted, arg):
                        return converted
                except TypeError:  # pragma: no cover - non-instantiable typing args
                    return converted
            return value

        @classmethod
        def _get_field_validator_store(cls) -> Dict[str, Dict[str, List[Callable[..., Any]]]]:
            cache = cls.__dict__.get("__pydantic_validator_store__")
            if cache is not None:
                return cache

            store: Dict[str, Dict[str, List[Callable[..., Any]]]] = {
                "before": {},
                "after": {},
            }

            for base in reversed(cls.__mro__):
                members = getattr(base, "__dict__", {})
                for name, attribute in members.items():
                    function = attribute
                    if isinstance(function, (classmethod, staticmethod)):
                        underlying = function.__func__
                    else:
                        underlying = function
                    metadata = getattr(underlying, "__pydantic_field_validators__", None)
                    if not metadata:
                        continue
                    bound = getattr(cls, name)
                    for entry in metadata:
                        mode = entry["mode"]
                        for field_name in entry["fields"]:
                            bucket = store.setdefault(mode, {}).setdefault(
                                field_name, []
                            )
                            if bound not in bucket:
                                bucket.append(bound)

            setattr(cls, "__pydantic_validator_store__", store)
            return store

        @classmethod
        def _run_field_validators(
            cls,
            validators: Iterable[Callable[..., Any]],
            name: str,
            value: Any,
        ) -> Any:
            result = value
            for validator in validators:
                try:
                    candidate = validator(result)
                except ValidationError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive path
                    raise ValidationError(f"Field '{name}': {exc}") from exc
                result = candidate
            return result

        @classmethod
        def _apply_field_constraints(
            cls, name: str, value: Any, field: FieldInfo | None
        ) -> Any:
            if field is None:
                return value
            metadata = field.metadata
            if not metadata:
                return value

            numeric_constraints = {
                key: metadata[key]
                for key in ("gt", "ge", "lt", "le")
                if key in metadata
            }
            if not numeric_constraints or value is None:
                return value

            if isinstance(value, bool):
                number = None
            elif isinstance(value, (int, float)):
                number = float(value)
            else:
                number = None

            if number is None:
                return value

            if not math.isfinite(number):
                raise ValidationError(
                    f"Field '{name}' must be a finite number"
                )

            lower_inclusive = numeric_constraints.get("ge")
            lower_exclusive = numeric_constraints.get("gt")
            upper_inclusive = numeric_constraints.get("le")
            upper_exclusive = numeric_constraints.get("lt")

            if lower_inclusive is not None and number < float(lower_inclusive):
                raise ValidationError(
                    f"Field '{name}' must be >= {lower_inclusive}"
                )
            if lower_exclusive is not None and number <= float(lower_exclusive):
                raise ValidationError(
                    f"Field '{name}' must be > {lower_exclusive}"
                )
            if upper_inclusive is not None and number > float(upper_inclusive):
                raise ValidationError(
                    f"Field '{name}' must be <= {upper_inclusive}"
                )
            if upper_exclusive is not None and number >= float(upper_exclusive):
                raise ValidationError(
                    f"Field '{name}' must be < {upper_exclusive}"
                )

            return value

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

        def model_copy(self: T, *, update: Optional[Dict[str, Any]] = None) -> T:
            annotations = getattr(self, "__annotations__", {})
            data = {name: getattr(self, name) for name in annotations}
            update_fields: Dict[str, Any] = {}
            if update:
                for key, value in update.items():
                    field_name = self._resolve_field_name(key)
                    update_fields[field_name] = value
                data.update(update_fields)
            copied = self.__class__.__new__(self.__class__)
            BaseModel.__init__(copied, **data)
            for key, value in self.__dict__.items():
                if key not in annotations and key != "__pydantic_fields_set__":
                    setattr(copied, key, value)
            original_fields = set(getattr(self, "__pydantic_fields_set__", set()))
            if update:
                original_fields.update(update_fields.keys())
            copied.__pydantic_fields_set__ = original_fields
            return copied

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


    def computed_field(*args: Any, **__: Any):
        def decorator(func: Any) -> property:
            if isinstance(func, property):
                return func
            return property(func)

        if args:
            target = args[0]
            if isinstance(target, property):
                return target
            if callable(target):
                return decorator(target)
        return decorator

    def field_validator(*names: str, mode: str | None = None, **_: Any):
        if not names:
            raise ValueError("field_validator requires at least one field name")

        normalized = tuple(str(name) for name in names)
        mode_value = (mode or "after").lower()
        if mode_value not in {"before", "after"}:
            raise ValueError("field_validator mode must be 'before' or 'after'")

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            target = function
            if isinstance(function, (classmethod, staticmethod)):
                target = function.__func__

            existing: List[Dict[str, Any]] = getattr(
                target, "__pydantic_field_validators__", []
            )
            existing.append({"fields": normalized, "mode": mode_value})
            setattr(target, "__pydantic_field_validators__", existing)
            return function

        return decorator


    def model_validator(*fields: str, **_: Any):
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


    ConfigDict = dict


    __all__ = [
        "AliasChoices",
        "BaseModel",
        "ConfigDict",
        "EmailStr",
        "Field",
        "computed_field",
        "field_validator",
        "model_validator",
        "ValidationError",
    ]
