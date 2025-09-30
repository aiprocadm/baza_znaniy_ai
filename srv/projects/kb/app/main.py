        codex/create-sqlmodel-models-for-files-and-pages
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
        main
