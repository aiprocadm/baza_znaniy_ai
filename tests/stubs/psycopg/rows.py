"""Row factories for the psycopg stub."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def dict_row(cursor: Any) -> Any:  # noqa: ANN401 - match psycopg signature
    """Return a callable that converts query results to dictionaries."""

    def factory(row: Iterable[Any]) -> Mapping[str, Any]:
        if isinstance(row, Mapping):
            return dict(row)
        if hasattr(cursor, "description"):
            keys = [col[0] for col in cursor.description]
            return dict(zip(keys, row))
        raise TypeError("Row data must be mapping-like when cursor lacks description")

    return factory


__all__ = ["dict_row"]
