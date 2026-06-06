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


def _search_hits(n):
    from app.services.kb_store import SearchHit

    return [
        SearchHit(
            document_id=1, document_title="d", chunk_index=i, text=f"t{i}", score=1.0 - i * 0.1
        )
        for i in range(n)
    ]


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, *, top_k):
        return list(self._hits[:top_k])


def test_reranking_search_passthrough_when_disabled():
    from app.eval.adapter import _reranking_search
    from app.services.kb_rerank import RerankConfig

    cfg = RerankConfig(enabled=False, model_name="x", candidates=20, top_n=5, batch_size=8)
    search = _reranking_search(_FakeStore(_search_hits(5)), cfg)
    # disabled rerank → raw bi-encoder order, truncated to k
    assert [h.chunk_index for h in search("q", 3)] == [0, 1, 2]


def test_reranking_search_applies_reranker_order():
    from app.eval.adapter import _reranking_search
    from app.services import kb_rerank

    class _FakeReranker:
        def rerank(self, query, hit_dicts, top_k):
            return list(reversed(hit_dicts))[:top_k]

    cfg = kb_rerank.RerankConfig(
        enabled=True, model_name="fake-rr", candidates=20, top_n=2, batch_size=8
    )
    kb_rerank.reset_cache()
    kb_rerank._RERANKER_CACHE["fake-rr"] = _FakeReranker()
    try:
        search = _reranking_search(_FakeStore(_search_hits(5)), cfg)
        out = search("q", 2)
    finally:
        kb_rerank.reset_cache()
    # reranker reversed the 5-hit shortlist, top_n=2 → chunk_index 4, 3
    assert [h.chunk_index for h in out] == [4, 3]
