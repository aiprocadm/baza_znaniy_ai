"""Minimal psycopg.conninfo stub used in unit tests."""

from __future__ import annotations

from typing import Any


def make_conninfo(**kwargs: Any) -> str:
    """Return a simple DSN string constructed from keyword arguments."""

    parts = []
    for key, value in kwargs.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


__all__ = ["make_conninfo"]
