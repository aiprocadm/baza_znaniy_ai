"""Minimal test client compatible with the subset of FastAPI used in tests."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, Iterable

from . import HTTPException, UploadFile, _build_call_arguments, _serialise
from .uploads import coerce_uploads, ensure_list
from .responses import HTMLResponse, JSONResponse, Response

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from . import FastAPI


class _SimpleResponse:
    def __init__(self, status_code: int, content: Any) -> None:
        self.status_code = status_code
        self._content = content

    def json(self) -> Any:
        return self._content

    @property
    def text(self) -> str:
        return str(self._content)


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

        kwargs = _build_call_arguments(route.handler, body, params, self.app)
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
                result = asyncio.run(result)
        except HTTPException as exc:
            return _SimpleResponse(exc.status_code, {"detail": exc.detail})

        if isinstance(result, (JSONResponse, HTMLResponse, Response)):
            return _SimpleResponse(result.status_code, result.json())
        if isinstance(result, tuple):
            content, status_code = result if len(result) == 2 else (result[0], route.status_code)
            return _SimpleResponse(status_code, _serialise(content))

        content = _serialise(result)
        return _SimpleResponse(route.status_code, content)

    def _run_handler(self, handler: Any) -> None:
        result = handler()
        if inspect.isawaitable(result):
            asyncio.run(result)

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


__all__ = ["TestClient"]
