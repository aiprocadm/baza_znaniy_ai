import importlib.util
from pathlib import Path
import sys
import types

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
    hits = [
        {"file": f"doc{i}.pdf", "page": i, "score": 1 / (i + 1)}
        for i in range(10)
    ]

    citations, has_minimum = select_citations(hits, minimum=3, maximum=5)

    assert len(citations) == 5
    assert citations[0]["file"] == "doc0.pdf"
    assert citations[-1]["file"] == "doc4.pdf"
    assert has_minimum is True
