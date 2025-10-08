"""Minimal test client compatible with the subset of FastAPI used in tests."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import TYPE_CHECKING, Any, Iterable

from . import BackgroundTasks, HTTPException, UploadFile, _build_call_arguments, _serialise
from . import responses as _responses
from .uploads import coerce_uploads, ensure_list

HTMLResponse = _responses.HTMLResponse
JSONResponse = _responses.JSONResponse
Response = _responses.Response
StreamingResponse = getattr(_responses, "StreamingResponse", None)

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .responses import StreamingResponse as StreamingResponseType
else:  # pragma: no cover - runtime fallback when streaming support is unavailable
    StreamingResponseType = Any

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from . import FastAPI


class _SimpleResponse:
    def __init__(self, status_code: int, content: Any) -> None:
        self.status_code = status_code
        self._content = content

    def json(self) -> Any:
        if isinstance(self._content, (bytes, bytearray, memoryview)):
            data = bytes(self._content)
            try:
                return json.loads(data.decode())
            except Exception:
                return data.decode(errors="ignore")
        if isinstance(self._content, str):
            try:
                return json.loads(self._content)
            except Exception:
                return self._content
        return self._content

    @property
    def content(self) -> Any:
        return self._content

    @property
    def text(self) -> str:
        content = self._content
        if isinstance(content, (bytes, bytearray, memoryview)):
            try:
                return bytes(content).decode()
            except Exception:
                return repr(bytes(content))
        return str(content)


class TestClient:
    """Very small subset of ``fastapi.testclient.TestClient`` used in tests."""

    def __init__(self, app: "FastAPI") -> None:
        self.app = app
        for handler in self.app._event_handlers.get("startup", []):  # type: ignore[attr-defined]
            self._run_handler(handler)

    def __enter__(self) -> "TestClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handler in reversed(self.app._event_handlers.get("shutdown", [])):  # type: ignore[attr-defined]
            self._run_handler(handler)
        return False

    def close(self) -> None:
        for handler in reversed(self.app._event_handlers.get("shutdown", [])):  # type: ignore[attr-defined]
            self._run_handler(handler)

    # ------------------------------------------------------------------
    # Public request helpers
    # ------------------------------------------------------------------
    def get(self, path: str, **options: Any) -> _SimpleResponse:
        return self._request("GET", path, **options)

    def post(
        self,
        path: str,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        files: Any | None = None,
        **options: Any,
    ) -> _SimpleResponse:
        return self._request_with_body("POST", path, json=json, data=data, files=files, **options)

    def put(
        self,
        path: str,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        files: Any | None = None,
        **options: Any,
    ) -> _SimpleResponse:
        return self._request_with_body("PUT", path, json=json, data=data, files=files, **options)

    def patch(
        self,
        path: str,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        files: Any | None = None,
        **options: Any,
    ) -> _SimpleResponse:
        return self._request_with_body("PATCH", path, json=json, data=data, files=files, **options)

    def delete(self, path: str, **options: Any) -> _SimpleResponse:
        return self._request("DELETE", path, **options)

    def options(self, path: str, **options: Any) -> _SimpleResponse:
        return self._request("OPTIONS", path, **options)

    def head(self, path: str, **options: Any) -> _SimpleResponse:
        return self._request("HEAD", path, **options)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request_with_body(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        data: dict[str, Any] | None = None,
        files: Any | None = None,
        **options: Any,
    ) -> _SimpleResponse:
        if data is not None or files is not None:
            payload: dict[str, Any] = dict(data or {})
            if files:
                for key, value in self._iter_files(files):
                    uploads = self._coerce_files(value)
                    self._merge_uploads(payload, key, uploads)
            return self._request(method, path, body=payload, **options)

        return self._request(method, path, body=json, **options)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Any | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        **_: Any,
    ) -> _SimpleResponse:
        route, params = self.app._find_route(method, path)  # type: ignore[attr-defined]
        if route is None or params is None:
            raise AssertionError(f"No route registered for {method} {path}")

        kwargs, background = _build_call_arguments(route.handler, body, params, self.app)
        if data:
            for key, value in data.items():
                kwargs.setdefault(key, value)
        if files:
            for key, value in files.items():
                uploads = self._coerce_files(value)
                self._merge_uploads(kwargs, key, uploads)

        try:
            result = route.handler(**kwargs)
            if inspect.isawaitable(result):
                result = self._run_async(result)
        except HTTPException as exc:
            return _SimpleResponse(exc.status_code, {"detail": exc.detail})

        def _collect_backgrounds(*extra: BackgroundTasks | None) -> list[BackgroundTasks]:
            collected: list[BackgroundTasks] = []
            if isinstance(background, BackgroundTasks):
                collected.append(background)
            for candidate in extra:
                if isinstance(candidate, BackgroundTasks) and candidate not in collected:
                    collected.append(candidate)
            return collected

        if StreamingResponse is not None and isinstance(result, StreamingResponse):
            content = self._consume_streaming_response(result)
            return self._finalise_response(
                content,
                result.status_code,
                _collect_backgrounds(getattr(result, "background", None)),
            )

        if isinstance(result, (JSONResponse, HTMLResponse, Response)):
            return self._finalise_response(
                result.json(),
                result.status_code,
                _collect_backgrounds(getattr(result, "background", None)),
            )
        if isinstance(result, tuple):
            content, status_code = result if len(result) == 2 else (result[0], route.status_code)
            return self._finalise_response(
                _serialise(content),
                status_code,
                _collect_backgrounds(),
            )

        content = _serialise(result)
        return self._finalise_response(content, route.status_code, _collect_backgrounds())

    def _run_handler(self, handler: Any) -> None:
        result = handler()
        if inspect.isawaitable(result):
            self._run_async(result)

    def _iter_files(self, items: Any) -> Iterable[tuple[str, Any]]:
        if isinstance(items, dict):
            return items.items()
        return list(items)

    def _coerce_files(self, value: Any) -> list[UploadFile]:
        return coerce_uploads(value)

    def _merge_uploads(self, target: dict[str, Any], key: str, uploads: list[UploadFile]) -> None:
        existing = target.get(key)
        if existing is None:
            target[key] = list(uploads)
        else:
            combined = ensure_list(existing)
            combined.extend(uploads)
            target[key] = combined

    def _finalise_response(
        self,
        content: Any,
        status_code: int,
        backgrounds: Iterable[BackgroundTasks],
    ) -> _SimpleResponse:
        self._run_background_tasks(backgrounds)
        return _SimpleResponse(status_code, content)

    def _run_background_tasks(self, backgrounds: Iterable[BackgroundTasks]) -> None:
        seen: set[int] = set()
        for background in backgrounds:
            identifier = id(background)
            if identifier in seen or not background.has_tasks():
                continue
            seen.add(identifier)
            for func, args, kwargs in background.drain():
                self._execute_callable(func, *args, **kwargs)

    def _consume_streaming_response(self, response: StreamingResponseType) -> bytes:
        if StreamingResponse is None:
            raise RuntimeError("StreamingResponse support is unavailable in this stub")
        return self._run_async(response.read())

    def _execute_callable(self, func: Any, *args: Any, **kwargs: Any) -> None:
        outcome = func(*args, **kwargs)
        if inspect.isawaitable(outcome):
            self._run_async(outcome)

    def _run_async(self, coroutine: Any) -> Any:
        try:
            return asyncio.run(coroutine)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coroutine)
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                asyncio.set_event_loop(None)
                loop.close()


__all__ = ["TestClient"]
