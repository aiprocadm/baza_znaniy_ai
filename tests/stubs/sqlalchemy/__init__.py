"""Minimal stub of :mod:`sqlalchemy` used for tests."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


class Column:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class JSON:  # pragma: no cover - simple marker type
    pass


class Text:  # pragma: no cover - simple marker type
    pass


class UniqueConstraint:
    def __init__(self, *columns: str, **kwargs: Any) -> None:
        self.columns = columns
        self.kwargs = kwargs


def text(value: str) -> str:
    return value


class MetaData:
    def __init__(self) -> None:
        self.bound_engines: list[Any] = []

    def create_all(self, engine: Any) -> None:
        self.bound_engines.append(engine)


engine_module = ModuleType("sqlalchemy.engine")


class Engine:  # pragma: no cover - placeholder
    pass


def make_url(url: str) -> str:
    return url


engine_module.Engine = Engine
engine_module.make_url = make_url
sys.modules[engine_module.__name__] = engine_module


asyncio_module = ModuleType("sqlalchemy.ext.asyncio")


async def create_async_engine(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return args, kwargs


asyncio_module.create_async_engine = create_async_engine
sys.modules[asyncio_module.__name__] = asyncio_module


__all__ = [
    "Column",
    "JSON",
    "MetaData",
    "Text",
    "UniqueConstraint",
    "text",
]
