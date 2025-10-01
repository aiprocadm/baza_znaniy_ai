"""Minimal test client compatible with the subset of FastAPI used in tests."""

from __future__ import annotations

import asyncio
import inspect
from io import BytesIO
from tempfile import SpooledTemporaryFile
from typing import Any, Iterable

from . import HTTPException, UploadFile, _build_call_arguments, _serialise
from .responses import HTMLResponse, JSONResponse, Response


class _SimpleResponse:
    def __init__(self, status_code: int, content: Any) -> None:
        self.status_code = status_code
        self._content = content

    def json(self) -> Any:
        return self._content

    @property
    def text(self) -> str:
        return str(self._content)


def _ensure_bytes(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        return data.encode()
    reader = getattr(data, "read", None)
    if callable(reader):
        result = reader()
        if isinstance(result, bytes):
            return result
        if isinstance(result, str):
            return result.encode()
    try:
        return bytes(data)
    except Exception:  # pragma: no cover - defensive
        return b""


class TestClient:
    """Very small subset of ``fastapi.testclient.TestClient`` used in tests."""

    def __init__(self, app):
        self.app = app
        for handler in self.app._event_handlers.get("startup", []):  # type: ignore[attr-defined]
            self._run_handler(handler)

    def __enter__(self) -> "TestClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
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
        if data is not None or files is not None:
            payload: dict[str, Any] = dict(data or {})
            if files:
                payload.update({key: self._coerce_files(value) for key, value in self._iter_files(files)})
            return self._request("POST", path, body=payload, **options)
        return self._request("POST", path, body=json, **options)

    def delete(self, path: str, **options: Any) -> _SimpleResponse:
        return self._request("DELETE", path, **options)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_handler(self, handler):
        result = handler()
        if inspect.iscoroutine(result):
            asyncio.run(result)

    def _iter_files(self, items: Any) -> Iterable[tuple[str, Any]]:
        if isinstance(items, dict):
            return items.items()
        return list(items)

    def _coerce_files(self, value: Any) -> list[UploadFile]:
        uploads: list[UploadFile] = []
        entries: Iterable[Any]
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            entries = value  # type: ignore[assignment]
        else:
            entries = [value]
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                filename = entry[0]
                raw_content = entry[1] if len(entry) > 1 else b""
                content_type = entry[2] if len(entry) > 2 else None
                if hasattr(raw_content, "read"):
                    file_obj = raw_content
                    if hasattr(file_obj, "seek"):
                        try:
                            file_obj.seek(0)
                        except Exception:  # pragma: no cover - defensive
                            pass
                else:
                    file_obj = SpooledTemporaryFile(mode="w+b")
                    payload = _ensure_bytes(raw_content)
                    if payload:
                        file_obj.write(payload)
                    file_obj.seek(0)
                uploads.append(UploadFile(filename=filename, file=file_obj, content_type=content_type))
            else:
                filename = str(entry)
                uploads.append(UploadFile(filename=filename, file=BytesIO()))
        return uploads

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

        if data:
            body = dict(body or {})
            body.update(data)
        if files:
            file_payload = {key: self._coerce_files(value) for key, value in files.items()}
            body = dict(body or {})
            body.update(file_payload)

        kwargs = _build_call_arguments(route.handler, body, params, self.app)
        result = route.handler(**kwargs)
        if inspect.iscoroutine(result):
            result = asyncio.run(result)

        status_code = route.status_code
        if isinstance(result, Response):
            return _SimpleResponse(result.status_code, result.content)
        if isinstance(result, (HTMLResponse, JSONResponse)):
            return _SimpleResponse(result.status_code, result.content)
        if isinstance(result, tuple):
            content, status_code = result if len(result) == 2 else (result[0], status_code)
            return _SimpleResponse(status_code, _serialise(content))
        return _SimpleResponse(status_code, _serialise(result))


__all__ = ["TestClient"]
