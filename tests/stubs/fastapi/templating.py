"""Templates support for the FastAPI compatibility layer."""

from __future__ import annotations

from typing import Any

from .responses import HTMLResponse


class Jinja2Templates:
    """Very small subset of the ``Jinja2Templates`` API used in the project."""

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def TemplateResponse(self, name: str, context: dict[str, Any]) -> HTMLResponse:  # noqa: N802
        # The stub does not render templates; it simply returns an empty response.
        return HTMLResponse("", status_code=200)
