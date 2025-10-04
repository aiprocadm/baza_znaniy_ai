"""Lightweight stub of ``sqlmodel`` for unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class FakeMetaData:
    """Minimal metadata stub exposing ``create_all``/``drop_all`` as no-ops."""

    def create_all(self, engine: Any) -> None:  # pragma: no cover - trivial no-op
        return None

    def drop_all(self, engine: Any) -> None:  # pragma: no cover - trivial no-op
        return None


class SQLModel:
    metadata = FakeMetaData()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        kwargs.pop("table", None)
        for key, value in kwargs.items():
            setattr(cls, key, value)


def Field(default: Any | None = None, **_: Any) -> Any | None:
    return default


class Session:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs

    def __enter__(self) -> "Session":  # pragma: no cover - context manager support
        return self

    def __exit__(self, *exc: Any) -> None:  # pragma: no cover - context manager support
        return None


@dataclass
class _Engine:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    name: str = field(init=False)
    driver: str = field(init=False)
    dialect: "_Dialect" = field(init=False)

    class _Dialect:
        __slots__ = ("name", "driver")

        def __init__(self, name: str, driver: str) -> None:
            self.name = name
            self.driver = driver

    def __post_init__(self) -> None:
        url = ""
        if self.args:
            url = str(self.args[0])
        elif "url" in self.kwargs:
            url = str(self.kwargs["url"])

        scheme = url.split(":", 1)[0] if url else "sqlite"
        if "+" in scheme:
            name, driver = scheme.split("+", 1)
        else:
            name = scheme
            driver = scheme

        self.name = name
        self.driver = driver
        self.dialect = self._Dialect(name, driver)

    def dispose(self) -> None:  # pragma: no cover - trivial no-op
        return None


def create_engine(*args: Any, **kwargs: Any) -> _Engine:
    return _Engine(args, dict(kwargs))


def select(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return "select", args, kwargs


def delete(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return "delete", args, kwargs
