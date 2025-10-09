"""Tests for the modern operations console UI."""

from __future__ import annotations

import asyncio

from app.ui.router import ui_index


async def _get_index_html() -> str:
    response = await ui_index()
    body = response.body
    if isinstance(body, bytes):
        return body.decode("utf-8")
    if isinstance(body, str):
        return body
    return str(body)


def test_ui_index_contains_new_shell_elements() -> None:
    html = asyncio.run(_get_index_html())
    assert "data-app-shell" in html
    assert "status-grid" in html
    assert "toast-container" in html
    assert "theme-toggle" in html


def test_ui_index_uses_modern_components() -> None:
    html = asyncio.run(_get_index_html())
    assert "chat-citations" in html
    assert "metrics-grid" in html
    assert "search-hit__title" in html
    assert "card card--status" in html
