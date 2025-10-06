"""Utilities that make the SQLModel integration resilient to lightweight stubs."""

from __future__ import annotations

import logging
from typing import Any, Iterable

from sqlmodel import SQLModel


logger = logging.getLogger(__name__)


def is_sqlmodel_stub() -> bool:
    """Return ``True`` when the lightweight test stub is active instead of SQLModel."""

    metadata = getattr(SQLModel, "metadata", None)
    metadata_type = type(metadata)
    if metadata_type.__module__.startswith("tests.stubs") or metadata_type.__name__ == "FakeMetaData":
        return True

    module_name = getattr(SQLModel, "__module__", "")
    return module_name in {"sqlmodel", "tests.stubs.sqlmodel"} or ".tests.stubs." in module_name


def iter_sqlmodel_model_classes() -> list[type[Any]]:
    """Return all discovered ``SQLModel`` subclasses."""

    try:
        subclasses = list(SQLModel.__subclasses__())
    except Exception:  # pragma: no cover - defensive guard against exotic metaclasses
        return []

    discovered: list[type[Any]] = []
    seen: set[type[Any]] = set()

    while subclasses:
        candidate = subclasses.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        discovered.append(candidate)

        try:
            nested = list(candidate.__subclasses__())
        except Exception:  # pragma: no cover - subclass introspection is best-effort
            nested = []
        subclasses.extend(nested)

    return discovered


def collect_sqlmodel_tables() -> list[tuple[type[Any], Any]]:
    """Gather pairs of SQLModel classes and their bound SQLAlchemy tables."""

    tables: list[tuple[type[Any], Any]] = []

    for model_cls in iter_sqlmodel_model_classes():
        try:
            table = getattr(model_cls, "__table__", None)
        except Exception:  # pragma: no cover - attribute access is best-effort
            continue

        if table is None:
            continue

        tables.append((model_cls, table))

    return tables


def install_stub_model_initializers(models: Iterable[type[Any]]) -> None:
    """Attach a keyword-only ``__init__`` when running against the SQLModel stub."""

    if not is_sqlmodel_stub():
        return

    for model_cls in models:
        if not isinstance(model_cls, type):
            continue

        if "__init__" in model_cls.__dict__:
            continue

        annotations = getattr(model_cls, "__annotations__", {})
        field_names = [name for name in annotations if isinstance(name, str)]
        defaults = {name: getattr(model_cls, name, None) for name in field_names}

        def _make_init(cls: type[Any], names: list[str], default_values: dict[str, Any]):
            def __init__(self, **data: Any) -> None:
                for field_name in names:
                    if field_name in data:
                        value = data[field_name]
                    else:
                        default = default_values.get(field_name)
                        value = default() if callable(default) else default
                    setattr(self, field_name, value)

                for key, value in data.items():
                    if key in names:
                        continue
                    setattr(self, key, value)

            __init__.__qualname__ = f"{cls.__name__}.__init__"  # type: ignore[attr-defined]
            return __init__

        try:
            model_cls.__init__ = _make_init(model_cls, field_names, defaults)  # type: ignore[assignment]
        except Exception:  # pragma: no cover - attaching the shim is best-effort
            logger.debug(
                "Failed to attach SQLModel stub __init__ to %s", getattr(model_cls, "__name__", model_cls),
                exc_info=True,
            )


__all__ = [
    "collect_sqlmodel_tables",
    "install_stub_model_initializers",
    "is_sqlmodel_stub",
    "iter_sqlmodel_model_classes",
]
