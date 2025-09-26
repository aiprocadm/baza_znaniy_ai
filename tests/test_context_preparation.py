from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.context import build_context, select_citations


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


def test_select_citations_guarantees_minimum_with_duplicates():
    hits = [
        {"file": "doc1.pdf", "page": 1, "score": 0.9},
        {"file": "doc2.pdf", "page": 3, "score": 0.8},
    ]

    citations = select_citations(hits, minimum=3, maximum=5)

    assert len(citations) == 3
    assert citations[-1]["file"] == "doc2.pdf"


def test_select_citations_caps_at_maximum():
    hits = [
        {"file": f"doc{i}.pdf", "page": i, "score": 1 / (i + 1)}
        for i in range(10)
    ]

    citations = select_citations(hits, minimum=3, maximum=5)

    assert len(citations) == 5
    assert citations[0]["file"] == "doc0.pdf"
    assert citations[-1]["file"] == "doc4.pdf"
