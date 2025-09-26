"""Minimal FastAPI compatibility layer tailored for the unit tests."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from . import status
from .responses import HTMLResponse, JSONResponse


class HTTPException(Exception):
    """Simple exception carrying an HTTP status code."""

    def __init__(self, status_code: int, detail: Any | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Request:
    """Placeholder request object."""

    def __init__(self, scope: Optional[dict[str, Any]] = None) -> None:
        self.scope = scope or {}


class UploadFile:
    """Very small subset of the real UploadFile implementation."""

    def __init__(self, filename: str | None = None, content: bytes | None = None) -> None:
        self.filename = filename
        self._content = content or b""

    async def read(self) -> bytes:
        return self._content


def Depends(dependency: Callable[..., Any] | None = None) -> Callable[..., Any] | None:
    return dependency


def File(default: Any | None = None, **_: Any) -> Any | None:
    return default


def Query(default: Any | None = None, **_: Any) -> Any | None:
    return default


@dataclass
class _Route:
    method: str
    path: str
    handler: Callable[..., Any]
    status_code: int

    def match(self, path: str) -> Optional[Dict[str, str]]:
        template_parts = [part for part in self.path.strip("/").split("/") if part]
        path_parts = [part for part in path.strip("/").split("/") if part]
        if template_parts == [""] and path_parts == [""]:
            template_parts = []
            path_parts = []
        if len(template_parts) != len(path_parts):
            return None
        params: Dict[str, str] = {}
        for template, value in zip(template_parts, path_parts):
            if template.startswith("{") and template.endswith("}"):
                params[template[1:-1]] = value
            elif template != value:
                return None
        return params


class _RouterBase:
    def __init__(self, *, prefix: str = "") -> None:
        self._prefix = prefix.rstrip("/")
        self._routes: List[_Route] = []
        self._event_handlers: Dict[str, List[Callable[[], Any]]] = {
            "startup": [],
            "shutdown": [],
        }

    def _combine_path(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        if self._prefix:
            return f"{self._prefix}{path}".replace("//", "/")
        return path

    def _add_route(
        self, method: str, path: str, *, status_code: int | None = None, **_: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        full_path = self._combine_path(path)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._routes.append(_Route(method, full_path, func, status_code or 200))
            return func

        return decorator

    def get(self, path: str, **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("GET", path, **options)

    def post(self, path: str, **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("POST", path, **options)

    def delete(self, path: str, **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("DELETE", path, **options)

    def include_router(self, router: "APIRouter") -> None:
        for route in router._routes:
            self._routes.append(route)
        for key, handlers in router._event_handlers.items():
            self._event_handlers[key].extend(handlers)

    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._event_handlers.setdefault(event_type, []).append(func)
            return func

        return decorator

    def _find_route(self, method: str, path: str) -> tuple[_Route, Dict[str, str]] | tuple[None, None]:
        for route in self._routes:
            if route.method != method:
                continue
            params = route.match(path)
            if params is not None:
                return route, params
        return None, None


class APIRouter(_RouterBase):
    def __init__(self, *, prefix: str = "", **_: Any) -> None:
        super().__init__(prefix=prefix)


class FastAPI(_RouterBase):
    def __init__(self, *, title: str = "", version: str = "") -> None:
        super().__init__(prefix="")
        self.title = title
        self.version = version
        self._middlewares: List[Any] = []

    def add_middleware(self, middleware_cls: Any, **options: Any) -> None:
        self._middlewares.append((middleware_cls, options))

    @property
    def routes(self) -> List[_Route]:
        return list(self._routes)


def _serialise(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, list):
        return [_serialise(item) for item in data]
    if isinstance(data, dict):
        return {key: _serialise(value) for key, value in data.items()}
    if isinstance(data, (datetime, date)):
        return data.isoformat()
    return data


def _build_call_arguments(
    handler: Callable[..., Any], body: Any, path_params: Dict[str, str]
) -> Dict[str, Any]:
    signature = inspect.signature(handler)
    kwargs: Dict[str, Any] = {}
    body_assigned = False
    for name, parameter in signature.parameters.items():
        if name in path_params:
            kwargs[name] = path_params[name]
            continue
        if not body_assigned and body is not None:
            annotation = parameter.annotation
            if isinstance(annotation, str):
                annotation = handler.__globals__.get(annotation, annotation)
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                kwargs[name] = annotation.model_validate(body)
            else:
                kwargs[name] = body
            body_assigned = True
            continue
        if parameter.default is not inspect._empty:
            kwargs[name] = parameter.default
    return kwargs


__all__ = [
    "APIRouter",
    "Depends",
    "FastAPI",
    "File",
    "HTMLResponse",
    "HTTPException",
    "JSONResponse",
    "Query",
    "Request",
    "UploadFile",
    "status",
]

from .testclient import TestClient  # noqa: E402  # isort:skip

__all__.append("TestClient")
