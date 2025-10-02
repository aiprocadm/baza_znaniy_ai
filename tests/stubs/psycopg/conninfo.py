"""Subset of :mod:`psycopg.conninfo` required by the tests."""

from __future__ import annotations

from typing import Any


def _format_value(value: Any) -> str:
    if value is None:
        raise ValueError("None is not a valid DSN component")
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return "''"
    if any(ch.isspace() for ch in text) or "'" in text:
        escaped = text.replace("'", "''")
        return f"'{escaped}'"
    return text


def make_conninfo(dsn: str | None = None, /, **kwargs: Any) -> str:
    """Construct a libpq-style DSN string from keyword arguments."""

    parts: list[str] = []
    if dsn:
        parts.append(str(dsn))
    for key, value in kwargs.items():
        if value is None:
            continue
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


__all__ = ["make_conninfo"]
