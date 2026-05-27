"""Structured logging helpers and request context propagation."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_tenant_id: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id", default="-")
_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")
_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="-")


def bind_log_context(
    *,
    request_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    task_id: str | None = None,
) -> None:
    if request_id is not None:
        _request_id.set(request_id or "-")
    if tenant_id is not None:
        _tenant_id.set(tenant_id or "-")
    if user_id is not None:
        _user_id.set(user_id or "-")
    if task_id is not None:
        _task_id.set(task_id or "-")


def current_log_context() -> dict[str, str]:
    return {
        "request_id": _request_id.get(),
        "tenant_id": _tenant_id.get(),
        "user_id": _user_id.get(),
        "task_id": _task_id.get(),
    }


class JsonLogFormatter(logging.Formatter):
    """Format log records as JSON with contextual fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **current_log_context(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Enable JSON logging for API and worker processes."""

    root = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root.handlers = [handler]
    root.setLevel(level)
