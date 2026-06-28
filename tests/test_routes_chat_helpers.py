"""Characterization tests for the pure seams of the /api/chat handler.

These pin the prompt-assembly and citation-formatting logic lifted out of the
~196-line ``chat`` handler so the split is provably behaviour-preserving. The
retrieval-with-fallbacks seam stays covered by the /api/chat integration tests
(test_service_api.py / test_chat_llm_integration.py).
"""

from __future__ import annotations

import app.api.routes as routes

_build_chat_prompt = routes._build_chat_prompt
_format_answer_with_citations = routes._format_answer_with_citations


def test_build_chat_prompt_includes_all_sections() -> None:
    prompt = _build_chat_prompt(
        summary_text="SUMMARY",
        history_text="HISTORY",
        memory_text="MEMORY",
        context="CTX",
        message="QUESTION",
    )
    assert "### Система" in prompt
    assert "SUMMARY" in prompt
    assert "HISTORY" in prompt
    assert "MEMORY" in prompt
    assert "CTX" in prompt
    assert "QUESTION" in prompt


def test_build_chat_prompt_omits_empty_optionals() -> None:
    prompt = _build_chat_prompt(
        summary_text="",
        history_text="",
        memory_text="",
        context="",
        message="Q",
    )
    assert "Краткое содержание" not in prompt
    assert "Недавняя история" not in prompt
    assert "Долгосрочная память" not in prompt
    # Empty context falls back to the explicit placeholder.
    assert "(релевантные фрагменты не найдены)" in prompt


def test_format_answer_returns_answer_when_no_citations() -> None:
    assert _format_answer_with_citations("plain", [], False) == "plain"


def test_format_answer_skips_when_provider_handles_citations() -> None:
    citations = [{"file": "a.pdf", "page": 2}]
    assert _format_answer_with_citations("plain", citations, True) == "plain"


def test_format_answer_appends_sources_with_and_without_page() -> None:
    citations = [{"file": "a.pdf", "page": 3}, {"file": "b.pdf"}]
    out = _format_answer_with_citations("ans", citations, False)
    assert out.startswith("ans")
    assert "Источники:" in out
    assert "[1] a.pdf — страница 3" in out
    assert "[2] b.pdf" in out
