"""Minimal test client compatible with the subset of FastAPI used in tests."""

from __future__ import annotations

import asyncio
import inspect
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, Any, Iterable

from . import HTTPException, UploadFile, _build_call_arguments, _serialise
from .responses import HTMLResponse, JSONResponse, Response


if TYPE_CHECKING:  # pragma: no cover - typing aid only

if TYPE_CHECKING:  # pragma: no cover - used for type checkers only

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
        if data is not None or files is not None:
            payload: dict[str, Any] = dict(data or {})
            if files:

                for key, value in self._iter_files(files):
                    uploads = self._coerce_files(value)
                    self._merge_uploads(payload, key, uploads)

                def _iter_files(items: Any) -> Iterable[tuple[str, Any]]:
                    if isinstance(items, dict):
                        return items.items()
                    return list(items)

                for key, value in _iter_files(files):
                    entries = _normalise_file_entries(value)
                    uploads = [_build_upload_file(entry) for entry in entries]

                    existing = payload.get(key)
                    if existing is None:
                        payload[key] = uploads[0] if len(uploads) == 1 else uploads
                    else:
                        combined = _ensure_list(existing)
                        combined.extend(uploads)
                        payload[key] = combined


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

            file_payload: dict[str, Any] = {}
            for key, value in files.items():
                uploads = self._coerce_files(value)
                self._merge_uploads(file_payload, key, uploads)
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

            for key, value in files.items():
                entries = _normalise_file_entries(value)
                uploads = [_build_upload_file(entry) for entry in entries]

                existing = kwargs.get(key)
                if existing is None:
                    kwargs[key] = uploads[0] if len(uploads) == 1 else uploads
                else:
                    combined = _ensure_list(existing)
                    combined.extend(uploads)
                    kwargs[key] = combined
        try:
            result = route.handler(**kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
        except HTTPException as exc:
            return _SimpleResponse(exc.status_code, {"detail": exc.detail})

        if isinstance(result, (JSONResponse, HTMLResponse, Response)):
            return _SimpleResponse(result.status_code, result.json())

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
        entries = _normalise_file_entries(value)
        return [_build_upload_file(entry) for entry in entries]

    def _merge_uploads(self, target: dict[str, Any], key: str, uploads: list[UploadFile]) -> None:
        existing = target.get(key)
        if existing is None:
            target[key] = list(uploads)
        else:
            combined = _ensure_list(existing)
            combined.extend(uploads)
            target[key] = combined

def _normalise_file_entries(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], (list, tuple, UploadFile)):
        return list(value)
    return [value]


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _build_upload_file(entry: Any) -> UploadFile:
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
            try:
                file_obj.seek(0)
            except Exception:  # pragma: no cover - defensive
                pass
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



__all__ = ["TestClient"]


