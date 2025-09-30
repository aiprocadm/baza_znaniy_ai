"""Minimal response primitives for the FastAPI compatibility layer."""

from __future__ import annotations

from typing import Any


class Response:
    """Simple HTTP response container used by the test client."""

    def __init__(self, content: Any, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def json(self) -> Any:
        return self.content

    @property
    def text(self) -> str:
        return str(self.content)


class JSONResponse(Response):
    """JSON response wrapper returning structured content."""

    pass


class HTMLResponse(Response):
    """HTML response wrapper used by template rendering."""

    def __init__(self, content: str = "", status_code: int = 200) -> None:
        super().__init__(content, status_code=status_code)
