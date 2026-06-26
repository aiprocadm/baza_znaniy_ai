from dataclasses import dataclass
from app.eval.adapter import EvalHit, make_retriever


@dataclass
class _Hit:
    document_id: int
    chunk_index: int
    text: str
    filename: str = ""
    document_title: str = "doc"


def test_make_retriever_resolves_composite_chunk_key():
    # chunk_index repeats across documents; the (doc_id, chunk_index)->key map disambiguates.
    key_map = {(1, 0): "a.pdf:0", (1, 1): "a.pdf:1", (2, 0): "b.pdf:0"}
    hits = [_Hit(2, 0, "from doc2", filename="b.pdf"), _Hit(1, 1, "from doc1")]
    retriever = make_retriever(lambda q, k: hits[:k], key_map)
    out = retriever("q", 5)
    assert out == [
        EvalHit(chunk_key="b.pdf:0", text="from doc2", title="b.pdf"),
        EvalHit(chunk_key="a.pdf:1", text="from doc1", title="doc"),
    ]


def test_make_retriever_skips_unmapped_hits():
    retriever = make_retriever(lambda q, k: [_Hit(9, 9, "orphan")], {(1, 0): "f.md:0"})
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


def test_build_key_map_joins_filename(tmp_path):
    from app.services.kb_store import KnowledgeBaseStore
    from app.eval.adapter import _build_key_map, build_global_id_key_map

    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"))
    doc = store.add_document(title="Contract", text="x", filename="contract.md")
    doc_id = doc.id
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO kb_chunks(document_id, chunk_index, text, embedding, embedder, dim) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, 1, "y", b"\x00" * 8, "hash", 2),
        )
        conn.commit()
    key_map = _build_key_map(store)
    assert key_map[(doc_id, 0)] == "contract.md:0"
    assert key_map[(doc_id, 1)] == "contract.md:1"
    gid = build_global_id_key_map(store)
    assert set(gid.values()) == {"contract.md:0", "contract.md:1"}


def test_make_retriever_emits_chunk_keys():
    from app.eval.adapter import make_retriever, EvalHit

    class _Hit:
        def __init__(self, doc, idx, text):
            self.document_id, self.chunk_index, self.text = doc, idx, text

    hits = [_Hit(1, 0, "a"), _Hit(1, 2, "b")]
    key_map = {(1, 0): "f.md:0", (1, 2): "f.md:2"}
    out = make_retriever(lambda q, k: hits[:k], key_map)("q", 5)
    assert [h.chunk_key for h in out] == ["f.md:0", "f.md:2"]
    assert isinstance(out[0], EvalHit)


# --------------------------------------------------------------------------- #
# Reranker strictness: serving degrades gracefully, the gate must NOT.
#
# rerank_hits swallows a model-load failure and falls back to bi-encoder order
# (correct for /api/kb/ask — a user query should not 500). But the eval retriever
# exists to MEASURE the reranker; a silent fallback there feeds the gate
# base-identical metrics and discards a possibly-good model as a false NO-GO. So
# the eval path forces strict=True, turning that silent degrade into a loud fail.
# --------------------------------------------------------------------------- #
class _BoomReranker:
    def rerank(self, *args, **kwargs):
        raise RuntimeError("model load failed")


def _boom_config(strict):
    from app.services import kb_rerank

    return kb_rerank.RerankConfig(
        enabled=True,
        model_name="boom-rr",
        candidates=20,
        top_n=5,
        batch_size=8,
        strict=strict,
    )


def test_rerank_hits_nonstrict_degrades_to_bi_encoder_on_failure():
    from app.services import kb_rerank

    kb_rerank.reset_cache()
    kb_rerank._RERANKER_CACHE["boom-rr"] = _BoomReranker()
    try:
        result = kb_rerank.rerank_hits("q", _search_hits(5), config=_boom_config(strict=False))
    finally:
        kb_rerank.reset_cache()
    # graceful: original bi-encoder order, truncated to top_n — never raises.
    assert [h.chunk_index for h in result.hits] == [0, 1, 2, 3, 4]


def test_rerank_hits_strict_reraises_on_model_failure():
    import pytest

    from app.services import kb_rerank

    kb_rerank.reset_cache()
    kb_rerank._RERANKER_CACHE["boom-rr"] = _BoomReranker()
    try:
        with pytest.raises(RuntimeError, match="model load failed"):
            kb_rerank.rerank_hits("q", _search_hits(5), config=_boom_config(strict=True))
    finally:
        kb_rerank.reset_cache()


def test_load_config_reads_strict_from_env():
    from app.services import kb_rerank

    assert kb_rerank.load_config({}).strict is False  # default: graceful (serving)
    assert kb_rerank.load_config({"KB_RERANK_STRICT": "1"}).strict is True


def test_reranking_retriever_forces_strict_so_gate_cannot_silently_degrade(monkeypatch):
    import pytest

    from app.eval import adapter
    from app.services import kb_rerank

    # key map needs a real store DB; the failure fires inside search() before any
    # key lookup, so an empty map is enough to isolate the strictness behaviour.
    monkeypatch.setattr(adapter, "_build_key_map", lambda store: {})
    cfg = kb_rerank.RerankConfig(
        enabled=False, model_name="boom-rr", candidates=20, top_n=5, batch_size=8, strict=False
    )
    kb_rerank.reset_cache()
    kb_rerank._RERANKER_CACHE["boom-rr"] = _BoomReranker()
    try:
        retriever = adapter.make_mvp_reranking_retriever(_FakeStore(_search_hits(5)), config=cfg)
        # adapter forces strict=True even though cfg.strict was False -> loud fail.
        with pytest.raises(RuntimeError, match="model load failed"):
            retriever("q", 3)
    finally:
        kb_rerank.reset_cache()
