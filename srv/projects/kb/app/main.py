"""Compatibility layer exposing the same surface as the legacy service app."""

from __future__ import annotations

from typing import List, Mapping

from app.main import app


def _format_answer(answer: str, citations: List[Mapping[str, object]]) -> str:
    """Legacy helper preserved for unit tests."""

    answer_text = answer.strip()
    if not citations:
        return answer_text

    entries = []
    for idx, citation in enumerate(citations, start=1):
        file_id = citation.get("file") or "неизвестный источник"
        page = citation.get("page")
        suffix = f" — страница {page}" if page is not None else ""
        entries.append(f"[{idx}] {file_id}{suffix}")

    return "\n\n".join([answer_text, "Источники:", "\n".join(entries)])


__all__ = ["app", "_format_answer"]
