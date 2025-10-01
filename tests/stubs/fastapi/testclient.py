"""Lightweight stand-in for :mod:`fastapi.testclient`."""

from __future__ import annotations

import asyncio

import inspect

from io import BytesIO
from tempfile import SpooledTemporaryFile
from typing import Any, Iterable

from tempfile import SpooledTemporaryFile



from tempfile import SpooledTemporaryFile
from typing import TYPE_CHECKING, Any, Iterable

from io import BytesIO



import io

 
from tempfile import SpooledTemporaryFile

from io import BytesIO
        main
        main


from typing import TYPE_CHECKING, Any, Iterable


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
    """Very small subset of the real ``TestClient`` used in tests."""

    def __init__(self, app):
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

                payload.update({key: self._coerce_files(value) for key, value in self._iter_files(files)})


                for key, uploads in _normalise_files(files):
                    payload.setdefault(key, []).extend(uploads)

                upload_list: list[UploadFile] = []


            if files:
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
        codex/update-upload-file-handling-and-tests
                            content = b""
                        upload_list.append(
                            UploadFile(filename=filename, file=io.BytesIO(_ensure_bytes(content)))
                        )

                            file_obj = SpooledTemporaryFile(mode="w+b")
                            content_type = None
                        upload_list.append(UploadFile(filename=filename, file=file_obj, content_type=content_type))

                        upload_list.append(_build_upload_file(entry))

        main
        main


                if upload_list:
                    payload["files"] = upload_list



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


            for key, uploads in files.items():
                kwargs.setdefault(key, []).extend(_normalise_entries(uploads))


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

                uploads: list[UploadFile] = []

                if isinstance(value, list):

                    uploads = [
                        UploadFile(filename=item[0], file=io.BytesIO(_ensure_bytes(item[1])))
                        for item in value
                    ]
                else:
                    filename, content, *_ = value
                    uploads = [
                        UploadFile(filename=filename, file=io.BytesIO(_ensure_bytes(content)))
                    ]


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


                    uploads = [_build_upload_file(item) for item in value]
                else:
                    uploads = [_build_upload_file(value)]

        main
        main

                kwargs[key] = uploads

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



def _normalise_file_entries(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], (list, tuple, UploadFile)):
        return list(value)
    return [value]


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _build_upload_file(entry: Any) -> UploadFile:

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


