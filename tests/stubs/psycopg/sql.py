"""Minimal subset of psycopg.sql for unit testing."""

from __future__ import annotations

from typing import Iterable, Tuple, Union

SQLLike = Union["SQL", "Identifier", "Composed"]


class SQL:
    """Represent a literal SQL snippet."""

    __slots__ = ("text", "_obj")

    def __init__(self, text: str) -> None:
        self.text = text
        self._obj = (text,)

    def __add__(self, other: SQLLike) -> "Composed":
        return Composed((self, other))

    def format(self, *args: SQLLike) -> "Composed":
        parts = self.text.split("{}")
        if len(parts) - 1 != len(args):
            raise ValueError("Number of placeholders does not match arguments")

        components: list[SQLLike] = []
        for index, part in enumerate(parts):
            if part:
                components.append(SQL(part))
            if index < len(args):
                components.append(args[index])
        return Composed(tuple(components))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SQL({self.text!r})"


class Identifier:
    """Represent an SQL identifier (optionally schema-qualified)."""

    __slots__ = ("_obj",)

    def __init__(self, *names: str) -> None:
        if not names:
            raise ValueError("Identifier requires at least one name component")
        self._obj = tuple(names)

    def __add__(self, other: SQLLike) -> "Composed":
        return Composed((self, other))

    def __iter__(self):  # pragma: no cover - convenience for tests
        return iter(self._obj)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Identifier(" + ", ".join(repr(name) for name in self._obj) + ")"


class Composed:
    """Composition of SQL snippets and identifiers."""

    __slots__ = ("_obj",)

    def __init__(self, parts: Iterable[SQLLike]):
        flattened: list[SQLLike] = []
        for part in parts:
            if isinstance(part, Composed):
                flattened.extend(part._obj)
            else:
                flattened.append(part)
        self._obj: Tuple[SQLLike, ...] = tuple(flattened)

    def __add__(self, other: SQLLike) -> "Composed":
        if isinstance(other, Composed):
            return Composed(self._obj + other._obj)
        return Composed(self._obj + (other,))

    def __iter__(self):  # pragma: no cover - convenience for tests
        return iter(self._obj)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Composed({self._obj!r})"


__all__ = ["SQL", "Identifier", "Composed"]
