"""Compatibility layer exposing the same surface as the legacy service app."""

from __future__ import annotations

from typing import Iterable, Mapping

from app.main import app


def _format_answer(answer: str, citations: Iterable[Mapping[str, object]]) -> str:
    """Return ``answer`` formatted with a numbered list of ``citations``."""

    text = (answer or "").strip()
    entries = list(citations)
    if not entries:
        return text

    lines = [text, "", "Источники:"]
    for index, citation in enumerate(entries, start=1):
        file_id = citation.get("file") or citation.get("id") or "неизвестный источник"
        page = citation.get("page")
        suffix = f" — страница {page}" if page not in (None, "") else ""
        lines.append(f"[{index}] {file_id}{suffix}")

    return "\n".join(lines)


__all__ = ["app", "_format_answer"]
