"""Bridge the eval to the live MVP retriever using a stable global chunk id.

The eval's canonical chunk identity is ``kb_chunks.id`` — the same id
``synthetic_qa.iter_chunks`` stamps onto ``QAPair.source_chunk_id``. The MVP
``SearchHit`` exposes ``(document_id, chunk_index)`` but NOT the row id, and
``chunk_index`` is only unique *within* a document, so each hit is resolved to
its global id via a one-time map built from the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class EvalHit:
    chunk_id: int
    text: str
    title: str = ""


Retriever = Callable[[str, int], Sequence[EvalHit]]


class _SearchHitLike(Protocol):
    """Structural type for the live MVP ``SearchHit`` fields the eval reads."""

    document_id: int
    chunk_index: int
    text: str


def make_retriever(
    search: Callable[[str, int], Sequence[_SearchHitLike]],
    id_map: Mapping[tuple[int, int], int],
) -> Retriever:
    def _retrieve(query: str, top_k: int) -> list[EvalHit]:
        out: list[EvalHit] = []
        for h in search(query, top_k):
            cid = id_map.get((int(h.document_id), int(h.chunk_index)))
            if cid is None:
                continue
            title = getattr(h, "filename", "") or getattr(h, "document_title", "") or ""
            out.append(EvalHit(chunk_id=cid, text=h.text, title=title))
        return out

    return _retrieve


def _build_id_map(store) -> dict[tuple[int, int], int]:
    with store._connect() as conn:  # noqa: SLF001 — reuse store connection conventions
        rows = conn.execute("SELECT id, document_id, chunk_index FROM kb_chunks").fetchall()
    return {(int(doc_id), int(idx)): int(cid) for cid, doc_id, idx in rows}


def make_mvp_retriever(store) -> Retriever:
    """Wrap a live ``KnowledgeBaseStore`` as an eval Retriever."""
    return make_retriever(lambda q, k: store.search(q, top_k=k), _build_id_map(store))


def _reranking_search(store, config):
    """A ``(query, k)`` search fn mirroring ``kb_mvp.ask``: bi-encoder shortlist
    then cross-encoder rerank. Module-level (not a closure inside the retriever)
    so it is unit-testable with a fake store + injected reranker, no model load.
    """
    from app.services import kb_rerank

    def _search(query: str, k: int):
        shortlist = store.search(query, top_k=max(config.candidates, k))
        return kb_rerank.rerank_hits(query, shortlist, config=config, top_n=k).hits

    return _search


def make_mvp_reranking_retriever(store, config=None) -> Retriever:
    """Eval Retriever that ALWAYS applies the cross-encoder reranker (for gate C).

    ``make_mvp_retriever`` returns the raw bi-encoder order (``store.search``);
    reranking lives in ``kb_mvp.ask``, not the store — so this mirrors that path
    to make the reranker measurable via the eval harness. ``enabled`` is forced on
    (this retriever exists to rerank — avoids a silent passthrough when
    ``KB_RERANK_ENABLED`` is unset); ``config`` overrides model/candidates/top_n,
    otherwise ``KB_RERANK_*`` is used.
    """
    from dataclasses import replace

    from app.services import kb_rerank

    cfg = config if config is not None else kb_rerank.load_config()
    if not cfg.enabled:
        cfg = replace(cfg, enabled=True)
    return make_retriever(_reranking_search(store, cfg), _build_id_map(store))


def compute_signature(store):
    """Snapshot the live corpus for golden-set pinning.

    NOTE: ``embedder.dimension`` may trigger a one-time probe for remote
    embedders; acceptable here (run/generate already perform LLM calls).
    """
    from app.eval.dataset import CorpusSignature

    with store._connect() as conn:  # noqa: SLF001
        doc_count = int(conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0])
        row = conn.execute("SELECT MAX(id) FROM kb_chunks").fetchone()
        max_chunk_id = int(row[0]) if row and row[0] is not None else 0
    embedder = store.embedder
    return CorpusSignature(
        doc_count=doc_count,
        max_chunk_id=max_chunk_id,
        embedder_name=str(getattr(embedder, "name", "unknown")),
        dim=int(getattr(embedder, "dimension", 0)),
    )
