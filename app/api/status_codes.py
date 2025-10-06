"""Compatibility helpers for HTTP status codes across FastAPI versions."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import status


def _prefer_status(*names: str, fallback: int) -> int:
    """Return the first available HTTP status code constant."""

    for name in names:
        value = getattr(status, name, None)
        if isinstance(value, int):
            return value
    return fallback


HTTP_CONTENT_TOO_LARGE = _prefer_status(
    "HTTP_413_CONTENT_TOO_LARGE",
    "HTTP_413_REQUEST_ENTITY_TOO_LARGE",
    fallback=int(HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
)
"""HTTP status code used when an uploaded payload exceeds the configured limit."""

HTTP_UNPROCESSABLE_CONTENT = _prefer_status(
    "HTTP_422_UNPROCESSABLE_CONTENT",
    "HTTP_422_UNPROCESSABLE_ENTITY",
    fallback=int(HTTPStatus.UNPROCESSABLE_ENTITY),
)
"""HTTP status code used when the request payload fails validation."""


__all__ = [
    "HTTP_CONTENT_TOO_LARGE",
    "HTTP_UNPROCESSABLE_CONTENT",
]
