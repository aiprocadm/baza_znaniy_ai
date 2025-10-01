"""Lightweight stand-in for :mod:`fastapi.testclient`."""

from __future__ import annotations

import asyncio
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, Any, Iterable

from io import BytesIO

from . import HTTPException, UploadFile, _build_call_arguments, _serialise
from .responses import HTMLResponse, JSONResponse, Response

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from . import FastAPI


class _SimpleResponse:
    def __init__(self, status_code: int, content: Any) -> None:
        self.status_code = status_code
        self._content = content

    def json(self) -> Any:
        return self._content


class TestClient:
    """Very small subset of the real ``TestClient`` used in tests."""

    def __init__(self, app: "FastAPI") -> None:
        self.app = app
        for handler in getattr(self.app, "_event_handlers", {}).get("startup", []):
            self._run_handler(handler)

    def __enter__(self) -> "TestClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        for handler in reversed(getattr(self.app, "_event_handlers", {}).get("shutdown", [])):
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
        if data is not None or files is not None:
            payload: dict[str, Any] = dict(data or {})
            if files:
                for key, uploads in _normalise_files(files):
                    payload.setdefault(key, []).extend(uploads)
            return self._request("POST", path, body=payload, **options)

        return self._request("POST", path, body=json, **options)

    def delete(self, path: str, **options: Any) -> _SimpleResponse:
        return self._request("DELETE", path, **options)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
            for key, uploads in files.items():
                kwargs.setdefault(key, []).extend(_normalise_entries(uploads))

        try:
            result = route.handler(**kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
        except HTTPException as exc:  # pragma: no cover - exercised in other tests
            return _SimpleResponse(exc.status_code, {"detail": exc.detail})

        if isinstance(result, (JSONResponse, HTMLResponse, Response)):
            return _SimpleResponse(result.status_code, result.json())

        return _SimpleResponse(route.status_code, _serialise(result))

    def _run_handler(self, handler: Any) -> None:
        result = handler()
        if asyncio.isawaitable(result):
            asyncio.run(result)


def _normalise_files(files: Any) -> Iterable[tuple[str, list[UploadFile]]]:
    if isinstance(files, dict):
        items = files.items()
    else:
        items = files

    for key, value in items:
        yield key, _normalise_entries(value)


def _normalise_entries(value: Any) -> list[UploadFile]:
    if isinstance(value, list):
        entries = value
    else:
        entries = [value]
    return [_coerce_entry(entry) for entry in entries]


def _coerce_entry(entry: Any) -> UploadFile:
    if isinstance(entry, UploadFile):
        return entry

    if isinstance(entry, (list, tuple)):
        filename = entry[0] if entry else "uploaded"
        content = entry[1] if len(entry) > 1 else b""
        content_type = entry[2] if len(entry) > 2 else None
    else:
        filename = str(entry)
        content = b""
        content_type = None

    if hasattr(content, "read"):
        file_obj = content
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
    else:
        if isinstance(content, (bytes, bytearray, memoryview)):
            data = bytes(content)
        elif content is None:
            data = b""
        else:
            data = str(content).encode()
        file_obj = SpooledTemporaryFile(mode="w+b")
        if data:
            file_obj.write(data)
        file_obj.seek(0)

    kwargs: dict[str, Any] = {"filename": filename, "file": file_obj}
    if content_type is not None:
        kwargs["content_type"] = content_type
    return UploadFile(**kwargs)

