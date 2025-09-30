"""Compatibility layer exposing the same surface as the legacy service app."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.main import app


def _format_answer(answer: str, citations: Iterable[Mapping[str, Any]]) -> str:
    """Normalise the answer text and append human-readable citations."""

    answer_text = (answer or "").strip()
    items = list(citations)
    if not items:
        return answer_text

    entries = []
    for idx, citation in enumerate(items, start=1):
        file_id = (
            citation.get("file")
            or citation.get("chunk_id")
            or citation.get("id")
            or "неизвестный источник"
        )
        page = citation.get("page")
        suffix = f" — страница {page}" if page not in (None, "") else ""
        entries.append(f"[{idx}] {file_id}{suffix}")

    return "\n\n".join([answer_text, "Источники:", "\n".join(entries)])


__all__ = ["app", "_format_answer"]
