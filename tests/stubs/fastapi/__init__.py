"""Minimal FastAPI compatibility layer tailored for the unit tests."""

from __future__ import annotations

import inspect
import io
from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace

from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    IO,
    List,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from tempfile import SpooledTemporaryFile


from pydantic import BaseModel

from . import status
from .responses import HTMLResponse, JSONResponse, StreamingResponse

try:  # pragma: no cover - optional dependency
    from starlette.requests import Request as StarletteRequest
except Exception:  # pragma: no cover - fallback when Starlette is unavailable

    StarletteRequest = None  # type: ignore[assignment]


T = TypeVar("T")


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

    @property
    def app(self):
        return self.scope.get("app")


class WebSocket:
    """Stub WebSocket for testing."""

    def __init__(self, scope: Optional[dict[str, Any]] = None) -> None:
        self.scope = scope or {}
        self.client_state = None
        self.application_state = None

    @property
    def app(self):
        return self.scope.get("app")


class WebSocketDisconnect(Exception):
    """Stub exception for WebSocket disconnect."""

    def __init__(self, code: int = 1000) -> None:
        self.code = code


class UploadFile:
    """Very small subset of the real UploadFile implementation."""

    def __init__(
        self,
        filename: str | None = None,
        file: IO[bytes] | None = None,
        *,
        content: bytes | None = None,
        content_type: str | None = None,
        headers: Any | None = None,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self.headers = headers

        if file is None:
            stream = SpooledTemporaryFile(mode="w+b")
            if content:
                stream.write(content)
            stream.seek(0)
            file_obj: IO[bytes] = stream
            self._owns_file = True
        else:
            file_obj = file
            self._owns_file = False

        if file_obj is None:
            file_obj = io.BytesIO()
            self._owns_file = True

        self.file = file_obj

        if hasattr(self.file, "seek"):
            try:
                self.file.seek(0)
            except Exception:  # pragma: no cover - defensive
                pass

    async def read(self, size: int = -1) -> bytes:
        data = self.file.read(size)
        if isinstance(data, str):
            return data.encode()
        if data is None:
            return b""
        return data

    async def close(self) -> None:
        if hasattr(self.file, "close") and not getattr(self.file, "closed", False):
            self.file.close()


class BackgroundTasks:
    """Collect and execute callables after the response is returned."""

    def __init__(self) -> None:
        self._tasks: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = []

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._tasks.append((func, args, kwargs))

    def drain(self) -> list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]:
        tasks, self._tasks = self._tasks, []
        return tasks

    def has_tasks(self) -> bool:
        return bool(self._tasks)


def Depends(dependency: Callable[..., Any] | None = None) -> Callable[..., Any] | None:
    return dependency


def File(default: Any | None = None, **_: Any) -> Any | None:
    return default


def Form(default: Any | None = None, **_: Any) -> Any | None:
    return default


def Query(default: Any | None = None, **_: Any) -> Any | None:
    return default


def Header(default: Any | None = None, **_: Any) -> Any | None:
    return default


def Path(default: Any | None = None, **_: Any) -> Any | None:
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

    def put(self, path: str, **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("PUT", path, **options)

    def patch(
        self, path: str, **options: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("PATCH", path, **options)

    def delete(
        self, path: str, **options: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("DELETE", path, **options)

    def options(
        self, path: str, **options: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("OPTIONS", path, **options)

    def head(self, path: str, **options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return self._add_route("HEAD", path, **options)

    def include_router(
        self,
        router: "APIRouter",
        *,
        prefix: str = "",
        **_: Any,
    ) -> None:
        normalised_prefix = prefix.rstrip("/")
        for route in router._routes:
            path = route.path
            if normalised_prefix:
                path = f"{normalised_prefix}{path}".replace("//", "/")
            self._routes.append(_Route(route.method, path, route.handler, route.status_code))
        for key, handlers in router._event_handlers.items():
            self._event_handlers[key].extend(handlers)

    def on_event(self, event_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._event_handlers.setdefault(event_type, []).append(func)
            return func

        return decorator

    def _find_route(
        self, method: str, path: str
    ) -> tuple[_Route, Dict[str, str]] | tuple[None, None]:
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
        self.dependency_overrides: Dict[Callable[..., Any], Callable[..., Any]] = {}
        self.state = SimpleNamespace()
        self.extra: Dict[str, Any] = {}

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


def _resolve_dependency(dependency: Callable[..., Any] | Any, app: "FastAPI") -> Any:
    if not callable(dependency):
        return dependency

    override = (
        app.dependency_overrides.get(dependency) if hasattr(app, "dependency_overrides") else None
    )
    target = override or dependency

    signature = inspect.signature(target)
    type_hints = get_type_hints(target, include_extras=True)
    kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        annotation = type_hints.get(name, parameter.annotation)
        metadata: List[Any] = []
        if get_origin(annotation) is Annotated:
            args = list(get_args(annotation))
            if args:
                annotation = args.pop(0)
            metadata.extend(args)

        dependency_callable = next((meta for meta in metadata if callable(meta)), None)
        if dependency_callable is not None:
            kwargs[name] = _resolve_dependency(dependency_callable, app)
            continue

        default = parameter.default
        if default is inspect._empty:
            raise RuntimeError(f"Dependency '{target.__name__}' requires parameter '{name}'")
        kwargs[name] = _resolve_dependency(default, app)

    result = target(**kwargs)
    if inspect.isgenerator(result):
        try:
            return next(result)
        finally:  # pragma: no cover - generator cleanup is best-effort
            try:
                result.close()
            except Exception:
                pass
    return result


def _build_call_arguments(
    handler: Callable[..., Any], body: Any, path_params: Dict[str, str], app: "FastAPI"
) -> tuple[Dict[str, Any], BackgroundTasks | None]:
    signature = inspect.signature(handler)
    type_hints = get_type_hints(handler, include_extras=True)
    kwargs: Dict[str, Any] = {}
    body_assigned = False
    background_tasks: BackgroundTasks | None = None
    for name, parameter in signature.parameters.items():
        if name in path_params:
            kwargs[name] = path_params[name]
            continue

        annotation = type_hints.get(name, parameter.annotation)
        metadata: List[Any] = []
        if get_origin(annotation) is Annotated:
            args = list(get_args(annotation))
            if args:
                annotation = args.pop(0)
            metadata.extend(args)

        dependency_callable = next((meta for meta in metadata if callable(meta)), None)
        if dependency_callable is not None:
            kwargs[name] = _resolve_dependency(dependency_callable, app)
            continue

        if annotation is BackgroundTasks:
            background_tasks = background_tasks or BackgroundTasks()
            kwargs[name] = background_tasks
            continue

        if annotation in {Request, StarletteRequest}:
            kwargs[name] = Request({"app": app})
            continue

        if body is not None and isinstance(body, dict) and name in body:
            kwargs[name] = body[name]
            continue

        if not body_assigned and body is not None:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                kwargs[name] = annotation.model_validate(body)
            else:
                kwargs[name] = body
            body_assigned = True
            continue

        if parameter.default is not inspect._empty:
            kwargs[name] = _resolve_dependency(parameter.default, app)
    return kwargs, background_tasks


__all__ = [
    "APIRouter",
    "BackgroundTasks",
    "Depends",
    "FastAPI",
    "File",
    "Form",
    "Header",
    "HTMLResponse",
    "HTTPException",
    "JSONResponse",
    "Path",
    "Query",
    "Request",
    "StreamingResponse",
    "UploadFile",
    "WebSocket",
    "WebSocketDisconnect",
    "status",
    "TestClient",
]

from .testclient import TestClient  # noqa: E402  # isort:skip
