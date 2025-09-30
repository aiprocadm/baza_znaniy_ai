"""Routes serving the static operations console."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_PAGE_PATH = Path(__file__).with_name("index.html")


@router.get("/", response_class=HTMLResponse)
async def ui_index() -> HTMLResponse:
    """Return the single page application for the console."""

    html = _PAGE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


__all__ = ["router"]
