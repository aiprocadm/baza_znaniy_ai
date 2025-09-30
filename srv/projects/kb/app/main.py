"""Compatibility layer exposing the same surface as the legacy service app."""

from __future__ import annotations

        codex/clean-up-codebase-by-removing-codex-markers
from typing import Iterable, Mapping

from typing import Any, Iterable, Mapping
main

from app.main import app


        codex/clean-up-codebase-by-removing-codex-markers
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
        main

    return "\n".join(lines)


__all__ = ["app", "_format_answer"]
