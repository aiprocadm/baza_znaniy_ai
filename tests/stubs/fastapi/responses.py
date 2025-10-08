"""Minimal response primitives for the FastAPI compatibility layer."""

from __future__ import annotations

import inspect
import json
from typing import Any, AsyncIterable, Iterable, Iterator


class Response:
    """Simple HTTP response container used by the test client."""

    def __init__(
        self,
        content: Any,
        status_code: int = 200,
        *,
        background: Any | None = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.background = background

    def json(self) -> Any:
        content = self.content
        if isinstance(content, (bytes, bytearray, memoryview)):
            data = bytes(content)
            try:
                return json.loads(data.decode())
            except Exception:
                return data.decode(errors="ignore")
        if isinstance(content, str):
            try:
                return json.loads(content)
            except Exception:
                return content
        return content

    @property
    def text(self) -> str:
        content = self.content
        if isinstance(content, (bytes, bytearray, memoryview)):
            try:
                return bytes(content).decode()
            except Exception:
                return repr(bytes(content))
        return str(content)


class JSONResponse(Response):
    """JSON response wrapper returning structured content."""

    pass


class HTMLResponse(Response):
    """HTML response wrapper used by template rendering."""

    def __init__(self, content: str = "", status_code: int = 200, **kwargs: Any) -> None:
        super().__init__(content, status_code=status_code, **kwargs)


class StreamingResponse(Response):
    """Very small subset of ``StreamingResponse`` for unit tests."""

    def __init__(
        self,
        content: Iterable[Any] | AsyncIterable[Any],
        status_code: int = 200,
        *,
        background: Any | None = None,
        media_type: str | None = None,
    ) -> None:
        super().__init__(content, status_code=status_code, background=background)
        self.media_type = media_type

    def __iter__(self) -> Iterator[Any]:
        content = self.content
        if isinstance(content, (bytes, bytearray, memoryview, str)):
            yield content
            return
        if inspect.isgenerator(content):
            yield from content
            return
        if hasattr(content, "__iter__") and not inspect.isawaitable(content):
            yield from content  # type: ignore[misc]
            return
        raise TypeError("StreamingResponse content is not iterable")

    async def aiter(self) -> AsyncIterable[Any]:
        content = self.content
        if hasattr(content, "__aiter__"):
            async for item in content:  # type: ignore[attr-defined]
                yield item
            return
        if inspect.isasyncgen(content):
            async for item in content:  # type: ignore[misc]
                yield item
            return
        for item in self:
            yield item

    async def read(self) -> bytes:
        parts: list[Any] = []
        if hasattr(self.content, "__aiter__") or inspect.isasyncgen(self.content):
            async for item in self.aiter():
                parts.append(await _maybe_await(item))
        else:
            for item in self:
                parts.append(item)
        return _combine_stream_parts(parts)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _combine_stream_parts(parts: Iterable[Any]) -> bytes:
    buffer = bytearray()
    for part in parts:
        if isinstance(part, (bytes, bytearray, memoryview)):
            buffer.extend(part)
        else:
            buffer.extend(str(part).encode())
    return bytes(buffer)


__all__ = ["Response", "JSONResponse", "HTMLResponse", "StreamingResponse"]
