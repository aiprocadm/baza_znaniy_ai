"""Minimal test client compatible with the subset of FastAPI used in tests."""

from __future__ import annotations

import asyncio
import inspect
from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, Any, Iterable

from . import HTTPException, UploadFile, _build_call_arguments, _serialise
from .responses import HTMLResponse, JSONResponse, Response

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
                upload_list: list[UploadFile] = []

                def _iter_files(items: Any) -> Iterable[tuple[str, Any]]:
                    if isinstance(items, dict):
                        return items.items()
                    return list(items)

                for key, value in _iter_files(files):
                    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
                        entries = value  # type: ignore[assignment]
                    else:
                        entries = [value]

                    for entry in entries:  # type: ignore[assignment]
                        if isinstance(entry, (list, tuple)):
                            filename = entry[0]
                            raw_content = entry[1] if len(entry) > 1 else b""
                            if hasattr(raw_content, "read"):
                                file_obj = raw_content
                                if hasattr(file_obj, "seek"):
                                    try:
                                        file_obj.seek(0)
                                    except Exception:  # pragma: no cover - defensive
                                        pass
                            else:
                                file_obj = SpooledTemporaryFile(mode="w+b")
                                if raw_content:
                                    if isinstance(raw_content, bytes):
                                        file_obj.write(raw_content)
                                    else:
                                        file_obj.write(str(raw_content).encode())
                                    file_obj.seek(0)
                            content_type = entry[2] if len(entry) > 2 else None
                        else:
                            filename = str(entry)
                            file_obj = SpooledTemporaryFile(mode="w+b")
                            content_type = None
                        upload_list.append(UploadFile(filename=filename, file=file_obj, content_type=content_type))

                if upload_list:
                    payload["files"] = upload_list
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
            for key, value in files.items():
                uploads: list[UploadFile] = []

                if isinstance(value, list):
                    entries = value
                else:
                    entries = [value]

                for entry in entries:
                    if isinstance(entry, (list, tuple)):
                        filename = entry[0]
                        raw_content = entry[1] if len(entry) > 1 else b""
                        if hasattr(raw_content, "read"):
                            file_obj = raw_content
                            if hasattr(file_obj, "seek"):
                                try:
                                    file_obj.seek(0)
                                except Exception:  # pragma: no cover - defensive
                                    pass
                        else:
                            file_obj = SpooledTemporaryFile(mode="w+b")
                            if raw_content:
                                if isinstance(raw_content, bytes):
                                    file_obj.write(raw_content)
                                else:
                                    file_obj.write(str(raw_content).encode())
                                file_obj.seek(0)
                        content_type = entry[2] if len(entry) > 2 else None
                    else:
                        filename = str(entry)
                        file_obj = SpooledTemporaryFile(mode="w+b")
                        content_type = None

                    uploads.append(UploadFile(filename=filename, file=file_obj, content_type=content_type))

                kwargs[key] = uploads
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
