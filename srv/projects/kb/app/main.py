"""Minimal FastAPI compatibility layer exposing ``_format_answer`` for tests."""

from __future__ import annotations

from typing import Any, Iterable


def _format_answer(answer: str, citations: Iterable[dict[str, Any]]) -> str:
    """Normalise ``answer`` text and append human-readable ``citations``."""

    text = (answer or "").strip()
    items = list(citations)
    if not items:
        return text

    lines = [text, "", "Источники:", ""]
    for index, info in enumerate(items, start=1):
        file_id = info.get("file") or info.get("chunk_id") or info.get("id") or "Источник"
        entry = f"[{index}] {file_id}"
        page = info.get("page")
        if page not in (None, ""):
            entry += f" — страница {page}"
        lines.append(entry)

    return "\n".join(lines)


__all__ = ["_format_answer"]
