from dataclasses import dataclass
from app.eval.adapter import EvalHit, make_retriever


@dataclass
class _Hit:
    document_id: int
    chunk_index: int
    text: str
    filename: str = ""
    document_title: str = "doc"


def test_make_retriever_resolves_global_chunk_id():
    # chunk_index repeats across documents; the (doc_id, chunk_index)->id map disambiguates.
    id_map = {(1, 0): 100, (1, 1): 101, (2, 0): 200}
    hits = [_Hit(2, 0, "from doc2", filename="b.pdf"), _Hit(1, 1, "from doc1")]
    retriever = make_retriever(lambda q, k: hits[:k], id_map)
    out = retriever("q", 5)
    assert out == [
        EvalHit(chunk_id=200, text="from doc2", title="b.pdf"),
        EvalHit(chunk_id=101, text="from doc1", title="doc"),
    ]


def test_make_retriever_skips_unmapped_hits():
    retriever = make_retriever(lambda q, k: [_Hit(9, 9, "orphan")], {(1, 0): 1})
    assert retriever("q", 5) == []
