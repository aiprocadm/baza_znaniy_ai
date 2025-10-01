"""Lightweight stub of ``sqlmodel`` for unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SQLModel:
    metadata = None

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


def create_engine(*args: Any, **kwargs: Any) -> Any:
    @dataclass
    class _Engine:
        args: tuple[Any, ...]
        kwargs: dict[str, Any]

    return _Engine(args, dict(kwargs))


def select(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return "select", args, kwargs


def delete(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    return "delete", args, kwargs
