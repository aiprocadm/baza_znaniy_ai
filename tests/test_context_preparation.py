import sys
import types
import importlib.util
from pathlib import Path

import pytest

from app.rag import context as app_context

ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = ROOT / "srv" / "projects" / "kb" / "app"


def _load_rag():
    package_name = "kb_service_rag"

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            sys.modules.pop(module_name, None)

    package = types.ModuleType(package_name)
    package.__path__ = [str(SERVICE_ROOT)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    rag_spec = importlib.util.spec_from_file_location(
        f"{package_name}.rag", SERVICE_ROOT / "rag.py"
    )
    assert rag_spec and rag_spec.loader
    rag_module = importlib.util.module_from_spec(rag_spec)
    sys.modules[rag_spec.name] = rag_module
    rag_spec.loader.exec_module(rag_module)
    return rag_module


rag = _load_rag()
build_context = rag.build_context
select_citations = rag.select_citations


def test_build_context_respects_token_limit():
    hits = [
        {"text": "a" * 2500},
        {"text": "b" * 1000},
        {"text": "c" * 1000},
    ]

    context = build_context(hits, token_limit=3000)

    assert len(context) == 3000
    assert context.startswith("a" * 2500)
    assert context[2500:2502] == "\n\n"
    assert context.endswith("b" * (3000 - 2502))


def test_build_context_skips_empty_hits():
    hits = [
        {"text": ""},
        {"text": None},
        {"text": "useful"},
    ]

    context = build_context(hits, token_limit=10)

    assert context == "useful"


def test_select_citations_filters_duplicates_and_flags_shortage():
    hits = [
        {"file": "doc1.pdf", "page": 1, "score": 0.9},
        {"file": "doc1.pdf", "page": 1, "score": 0.8},
        {"file": "doc2.pdf", "page": 4, "score": 0.7},
    ]

    citations, has_minimum = select_citations(hits, minimum=3, maximum=5)

    assert len(citations) == 2
    assert [c["file"] for c in citations] == ["doc1.pdf", "doc2.pdf"]
    assert has_minimum is False


def test_select_citations_caps_at_maximum():
    hits = [{"file": f"doc{i}.pdf", "page": i, "score": 1 / (i + 1)} for i in range(10)]

    citations, has_minimum = select_citations(hits, minimum=3, maximum=5)

    assert len(citations) == 5
    assert citations[0]["file"] == "doc0.pdf"
    assert citations[-1]["file"] == "doc4.pdf"
    assert has_minimum is True


def test_app_build_context_skips_hits_with_empty_tokens(monkeypatch):
    def stub_tokenize(text: str):
        if text == "ignored":
            return []
        return list(text)

    monkeypatch.setattr(app_context, "tokenize", stub_tokenize)
    monkeypatch.setattr(app_context, "detokenize", lambda tokens: "".join(tokens))

    hits = [
        {"text": "ignored"},
        {"text": "kept"},
    ]

    context = app_context.build_context(hits, token_limit=10)

    assert context == "kept"


def test_app_build_context_returns_empty_string_for_empty_texts(monkeypatch):
    monkeypatch.setattr(app_context, "tokenize", lambda text: list(text))
    monkeypatch.setattr(app_context, "detokenize", lambda tokens: "".join(tokens))

    hits = [
        {"text": ""},
        {"text": None},
    ]

    context = app_context.build_context(hits, token_limit=10)

    assert context == ""


def test_app_build_context_breaks_when_separator_exhausts_limit(monkeypatch):
    monkeypatch.setattr(app_context, "tokenize", lambda text: list(text))
    monkeypatch.setattr(app_context, "detokenize", lambda tokens: "".join(tokens))

    hits = [
        {"text": "abc"},
        {"text": "def"},
        {"text": "ghi"},
    ]

    context = app_context.build_context(hits, token_limit=4)

    assert context == "abc\n"


def test_app_select_citations_raises_on_invalid_bounds():
    with pytest.raises(ValueError):
        app_context.select_citations([], minimum=3, maximum=2)


def test_app_select_citations_deduplicates_without_file_and_page():
    hits = [
        {"chunk_id": "chunk-a", "id": 1, "text": "alpha"},
        {"chunk_id": "chunk-b", "id": 2, "text": "beta"},
        {"chunk_id": "chunk-a", "id": 1, "text": "alpha"},
    ]

    citations, has_minimum = app_context.select_citations(hits, minimum=1, maximum=5)

    assert len(citations) == 2
    assert citations[0]["chunk_id"] == "chunk-a"
    assert citations[1]["chunk_id"] == "chunk-b"
    assert has_minimum is True
