"""Shared error response helpers ensuring consistent API envelopes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.status_codes import HTTP_UNPROCESSABLE_CONTENT

LOGGER = logging.getLogger(__name__)


def _normalise_detail(detail: Any) -> tuple[str, Any | None]:
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("detail")
        details = detail.get("details")
        if details is None:
            extras = {k: v for k, v in detail.items() if k not in {"message", "detail", "details"}}
            details = extras or None
        return (str(message) if message else "UNKNOWN_ERROR", details)

    if isinstance(detail, (list, tuple)):
        return "VALIDATION_ERROR", list(detail)

    if detail in {None, ""}:
        return "UNKNOWN_ERROR", None

    return str(detail), None


def _build_payload(status_code: int, message: str, details: Any | None = None) -> dict[str, Any]:
    return {"status": status_code, "message": message, "details": details}


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message, details = _normalise_detail(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_build_payload(exc.status_code, message, details),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    status_code = HTTP_UNPROCESSABLE_CONTENT
    return JSONResponse(
        status_code=status_code,
        content=_build_payload(status_code, "VALIDATION_ERROR", exc.errors()),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled application error", exc_info=exc)
    status_code = getattr(status, "HTTP_500_INTERNAL_SERVER_ERROR", 500)
    return JSONResponse(
        status_code=status_code,
        content=_build_payload(status_code, "INTERNAL_SERVER_ERROR"),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Install standardised exception handlers on *app*."""

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


__all__ = [
    "register_error_handlers",
    "http_exception_handler",
    "validation_exception_handler",
    "unhandled_exception_handler",
]

