from __future__ import annotations

import logging
import uuid
from http import HTTPStatus
from typing import Any, Type

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.constants import TRACE_HEADER

logger = logging.getLogger(__name__)


def _get_trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if trace_id is None:
        trace_id = uuid.uuid4().hex
        request.state.trace_id = trace_id
    return trace_id


def _response_payload(*, code: str, message: str, trace_id: str, details: Any) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "details": details,
        "trace_id": trace_id,
    }


def _json_response(*, status_code: int, payload: dict[str, Any], trace_id: str) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content=payload)
    response.headers.setdefault(TRACE_HEADER, trace_id)
    return response


def _register_http_exception_handler(app: FastAPI, exc_type: Type[HTTPException]) -> None:
    @app.exception_handler(exc_type)  # type: ignore[misc]
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        trace_id = _get_trace_id(request)
        status_code = exc.status_code
        try:
            status_phrase = HTTPStatus(status_code).phrase
        except ValueError:
            status_phrase = "HTTP Error"

        details: Any = None
        message: str
        if isinstance(exc.detail, str):
            message = exc.detail
        elif exc.detail is None:
            message = status_phrase
        else:
            message = status_phrase
            details = exc.detail

        payload = _response_payload(
            code=f"http_{status_code}",
            message=message,
            trace_id=trace_id,
            details=details,
        )
        return _json_response(status_code=status_code, payload=payload, trace_id=trace_id)


def install_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers for the FastAPI application."""

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        trace_id = _get_trace_id(request)
        payload = _response_payload(
            code="validation_error",
            message="Request validation failed",
            trace_id=trace_id,
            details=exc.errors(),
        )
        return _json_response(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, payload=payload, trace_id=trace_id)

    _register_http_exception_handler(app, HTTPException)
    _register_http_exception_handler(app, StarletteHTTPException)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        trace_id = _get_trace_id(request)
        logger.exception("Unhandled application exception", extra={"trace_id": trace_id})
        payload = _response_payload(
            code="internal_server_error",
            message="Internal server error",
            trace_id=trace_id,
            details=None,
        )
        return _json_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, payload=payload, trace_id=trace_id
        )


__all__ = ["install_error_handlers"]
