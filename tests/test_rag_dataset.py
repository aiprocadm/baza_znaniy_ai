"""Tests for app.services.rag_dataset — pure-logic RAG dataset builder."""

from __future__ import annotations


def test_module_imports() -> None:
    """Module imports without side effects."""
    from app.services import rag_dataset

    assert rag_dataset.__name__ == "app.services.rag_dataset"


def test_rag_variant_values() -> None:
    """The four canonical variants are exposed as string enum members."""
    from app.services.rag_dataset import RAGVariant

    assert {v.value for v in RAGVariant} == {
        "relevant",
        "irrelevant",
        "partial",
        "empty",
    }


def test_rag_sample_to_jsonl_line() -> None:
    """RAGSample.to_jsonl_line() emits one JSON object per line."""
    import json

    from app.services.rag_dataset import RAGSample, RAGVariant

    sample = RAGSample(
        instruction="Что такое отпуск?",
        input="",
        output="Отпуск — это [doc_chunk:7]",
        retrieved_context="Фрагмент [doc_chunk:7]: ...",
        variant=RAGVariant.RELEVANT,
        source_chunk_id=7,
        retrieved_chunk_ids=(7, 12),
    )
    line = sample.to_jsonl_line()
    assert line.endswith("\n")

    data = json.loads(line)
    assert data["instruction"] == "Что такое отпуск?"
    assert data["retrieved_context"].startswith("Фрагмент")
    assert data["meta"]["variant"] == "relevant"
    assert data["meta"]["source_chunk_id"] == 7
    assert data["meta"]["retrieved_chunk_ids"] == [7, 12]


from dataclasses import dataclass
from typing import Sequence

from app.services.synthetic_qa import QAPair


@dataclass(frozen=True)
class _FakeHit:
    """Minimal stand-in for a resolved retrieval hit.

    ``chunk_id`` is the GLOBAL ``kb_chunks.id`` — the same identity space as
    ``QAPair.source_chunk_id`` — and is what the dataset builder matches and
    cites on. ``chunk_index`` is the per-document ordinal; it is kept here (and
    defaults to a value that does NOT equal ``chunk_id``) so regression tests can
    prove matching uses the global id, never the ordinal.
    """

    chunk_id: int
    text: str
    chunk_index: int = -1
    document_id: int = 1
    document_title: str = "doc"
    score: float = 0.9
    source: str = "text"


def _retriever_with(hits_by_query: dict[str, list[_FakeHit]]):
    def _retrieve(query: str, top_k: int) -> Sequence[_FakeHit]:
        return list(hits_by_query.get(query, []))[:top_k]

    return _retrieve


def test_build_relevant_sample_joins_top_k_chunks() -> None:
    from app.services.rag_dataset import RAGVariant, build_relevant_sample

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    retriever = _retriever_with(
        {
            "Что такое отпуск?": [
                _FakeHit(chunk_id=7, text="Отпуск — это перерыв."),
                _FakeHit(chunk_id=12, text="Сотрудник имеет право."),
            ],
        }
    )

    sample = build_relevant_sample(seed, retriever=retriever, top_k=3)
    assert sample is not None
    assert sample.variant is RAGVariant.RELEVANT
    assert sample.source_chunk_id == 7
    assert 7 in sample.retrieved_chunk_ids
    assert "Отпуск — это перерыв." in sample.retrieved_context
    assert sample.output.endswith("[doc_chunk:7]")


def test_build_relevant_drops_when_source_chunk_missing() -> None:
    """If retrieval can't find the seed chunk, the sample is unsafe — drop it."""
    from app.services.rag_dataset import build_relevant_sample

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    retriever = _retriever_with(
        {"Что такое отпуск?": [_FakeHit(chunk_id=99, text="Совсем не про отпуск.")]}
    )

    assert build_relevant_sample(seed, retriever=retriever, top_k=3) is None


def test_build_relevant_matches_global_chunk_id_not_per_doc_index() -> None:
    """Regression: match the seed's GLOBAL kb_chunks.id, never the per-doc ordinal.

    ``source_chunk_id`` is the global ``kb_chunks.id`` (what ``iter_chunks``
    stamps). A hit's ``chunk_index`` is a per-document ordinal that is NOT unique
    across documents, so comparing the two conflates chunks. Here the correct hit
    carries ``chunk_id=42`` (its ordinal is 0), and a *different* chunk carries the
    colliding ordinal ``chunk_index=42``. The match — and the emitted
    ``retrieved_chunk_ids`` — must follow ``chunk_id``.
    """
    from app.services.rag_dataset import build_relevant_sample

    seed = QAPair(
        instruction="Вопрос про отпуск?",
        input="",
        output="Ответ. [doc_chunk:42]",
        source_chunk_id=42,
    )
    retriever = _retriever_with(
        {
            "Вопрос про отпуск?": [
                _FakeHit(chunk_id=42, chunk_index=0, text="Правильный фрагмент."),
                _FakeHit(chunk_id=7, chunk_index=42, text="Чужой фрагмент с ordinal 42."),
            ]
        }
    )

    sample = build_relevant_sample(seed, retriever=retriever, top_k=3)
    assert sample is not None
    # Global ids, in retrieval order — NOT the per-document ordinals (0, 42).
    assert sample.retrieved_chunk_ids == (42, 7)
    # The context cites the same global id the seed answer cites.
    assert "[doc_chunk:42]" in sample.retrieved_context


def test_build_relevant_drops_when_only_chunk_index_collides() -> None:
    """Regression: a per-doc ordinal that merely *equals* source_chunk_id is not a match.

    Before the fix, ``build_relevant_sample`` compared ``source_chunk_id`` (global
    id 5) against the hit's ``chunk_index`` (5) and produced a sample grounded in
    the WRONG chunk. The hit's real global id is 99, so the seed is not actually in
    the retrieval set and the sample must be dropped.
    """
    from app.services.rag_dataset import build_relevant_sample

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:5]",
        source_chunk_id=5,
    )
    retriever = _retriever_with(
        {"Что такое отпуск?": [_FakeHit(chunk_id=99, chunk_index=5, text="Чужой фрагмент.")]}
    )

    assert build_relevant_sample(seed, retriever=retriever, top_k=3) is None


def test_build_irrelevant_sample_uses_negative_chunks_and_refusal() -> None:
    from app.services.rag_dataset import (
        IRRELEVANT_REFUSAL,
        RAGVariant,
        build_irrelevant_sample,
    )

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    negative_chunks = [
        _FakeHit(chunk_id=200, text="Калибровка манометра — раз в год."),
        _FakeHit(chunk_id=201, text="Поверка средств измерения."),
    ]

    sample = build_irrelevant_sample(
        seed,
        negative_chunks=negative_chunks,
    )

    assert sample.variant is RAGVariant.IRRELEVANT
    assert sample.output == IRRELEVANT_REFUSAL
    assert sample.retrieved_chunk_ids == (200, 201)
    assert "Калибровка" in sample.retrieved_context
    assert sample.source_chunk_id == seed.source_chunk_id


def test_irrelevant_refusal_is_localised() -> None:
    """The refusal string mentions documents (not generic AI talk)."""
    from app.services.rag_dataset import IRRELEVANT_REFUSAL

    assert "документ" in IRRELEVANT_REFUSAL.lower()


def test_build_partial_sample_mixes_seed_with_distractors() -> None:
    from app.services.rag_dataset import (
        PARTIAL_PREFIX,
        RAGVariant,
        build_partial_sample,
    )

    seed = QAPair(
        instruction="Что такое отпуск?",
        input="",
        output="Это перерыв. [doc_chunk:7]",
        source_chunk_id=7,
    )
    seed_hit = _FakeHit(chunk_id=7, text="Отпуск — это перерыв в работе.")
    distractors = [
        _FakeHit(chunk_id=200, text="Калибровка манометра — раз в год."),
        _FakeHit(chunk_id=201, text="Поверка средств измерения."),
    ]

    sample = build_partial_sample(
        seed,
        seed_hit=seed_hit,
        distractor_chunks=distractors,
    )

    assert sample.variant is RAGVariant.PARTIAL
    assert sample.output.startswith(PARTIAL_PREFIX)
    assert "[doc_chunk:7]" in sample.output
    assert sample.retrieved_chunk_ids[0] == 7
    assert 200 in sample.retrieved_chunk_ids


def test_build_empty_sample_strips_citation_and_context() -> None:
    from app.services.rag_dataset import RAGVariant, build_empty_sample

    seed = QAPair(
        instruction="Какой сегодня день недели по тексту?",
        input="",
        output="Понедельник. [doc_chunk:42]",
        source_chunk_id=42,
    )
    sample = build_empty_sample(seed)

    assert sample.variant is RAGVariant.EMPTY
    assert sample.retrieved_context == ""
    assert sample.retrieved_chunk_ids == ()
    assert sample.output == "Понедельник."
    assert "[doc_chunk:" not in sample.output


def test_rag_sample_builder_respects_proportions() -> None:
    from collections import Counter

    from app.services.rag_dataset import RAGSampleBuilder, default_proportions

    seeds = [
        QAPair(
            instruction=f"Вопрос {i}?",
            input="",
            output=f"Ответ. [doc_chunk:{i}]",
            source_chunk_id=i,
        )
        for i in range(1, 21)
    ]
    seed_hits = {i: _FakeHit(chunk_id=i, text=f"Текст {i}") for i in range(1, 21)}

    def retriever(query: str, top_k: int):
        i = int(query.split()[1].rstrip("?"))
        return [seed_hits[i]]

    negatives = [_FakeHit(chunk_id=900 + j, text=f"Шум {j}") for j in range(5)]
    distractors = [_FakeHit(chunk_id=800 + j, text=f"Помеха {j}") for j in range(5)]

    builder = RAGSampleBuilder(
        retriever=retriever,
        negative_pool=negatives,
        distractor_pool=distractors,
        proportions=default_proportions(),
    )

    samples = list(builder.build(seeds, total=20))
    assert len(samples) == 20
    counts = Counter(s.variant.value for s in samples)
    assert counts["relevant"] == 14  # 70% of 20
    assert counts["irrelevant"] == 3  # 15% of 20
    assert counts["partial"] == 2  # 10% of 20
    assert counts["empty"] == 1  # 5% of 20


def test_rag_sample_builder_skips_relevant_when_source_missing() -> None:
    """If the retriever can't find the seed chunk, that slot is re-allocated."""
    from app.services.rag_dataset import RAGSampleBuilder, default_proportions

    seeds = [
        QAPair(
            instruction="Q1?",
            input="",
            output="A. [doc_chunk:1]",
            source_chunk_id=1,
        ),
        QAPair(
            instruction="Q2?",
            input="",
            output="A. [doc_chunk:2]",
            source_chunk_id=2,
        ),
    ]

    def retriever(query: str, top_k: int):
        return []  # always empty — every RELEVANT slot should be dropped

    builder = RAGSampleBuilder(
        retriever=retriever,
        negative_pool=[_FakeHit(chunk_id=900, text="нет")],
        distractor_pool=[_FakeHit(chunk_id=800, text="нет")],
        proportions=default_proportions(),
    )

    # We asked for 2 — both would have been RELEVANT but retrieval failed;
    # the builder is allowed to return fewer than requested.
    samples = list(builder.build(seeds, total=2))
    for s in samples:
        assert s.variant.value != "relevant"


def test_strip_citations_is_public_api() -> None:
    """W4 imports strip_citations directly — keep it on the module surface."""
    from app.services.rag_dataset import strip_citations

    assert strip_citations("Ответ. [doc_chunk:7]") == "Ответ."
    # The regex consumes surrounding whitespace, collapsing each marker
    # (and its adjacent spaces) into a single space.
    assert strip_citations("До [doc_chunk:1] середина [doc_chunk:2] конец") == "До середина конец"
    assert strip_citations("без цитат") == "без цитат"


def test_relevant_sample_matches_global_id_on_two_document_corpus(tmp_path) -> None:
    """End-to-end regression on a real 2-document corpus where id != chunk_index.

    Each short document yields a single chunk, so both chunks get
    ``chunk_index == 0`` while their global ``kb_chunks.id`` values diverge. The
    CLI's resolving retriever must map a ``SearchHit`` back to its global id so
    ``build_relevant_sample`` matches the seed by ``source_chunk_id`` (the global
    id). Before the fix this returned ``None`` — the global id was compared
    against the per-document ordinal 0.
    """
    from app.eval.adapter import _build_id_map, make_retriever
    from app.services.kb_store import KnowledgeBaseStore, SearchHit
    from app.services.rag_dataset import build_relevant_sample
    from app.services.synthetic_qa import QAPair

    store = KnowledgeBaseStore(db_path=tmp_path / "kb.sqlite")
    store.add_document("Doc A", "Альфа: про ежегодный оплачиваемый отпуск.")
    store.add_document("Doc B", "Бета: про калибровку манометра раз в год.")

    with store._connect() as conn:  # noqa: SLF001 - test reuse of internal helper
        rows = list(conn.execute("SELECT id, document_id, chunk_index FROM kb_chunks ORDER BY id"))

    # Single-chunk documents: per-document ordinals collide at 0, global ids are unique.
    assert len(rows) == 2
    (id_a, _doc_a, idx_a), (id_b, doc_b, idx_b) = rows
    assert idx_a == idx_b == 0  # ordinals collide across documents
    assert id_a != id_b  # global ids are unique
    assert id_b != idx_b  # the exact divergence the bug confused (e.g. 2 != 0)

    id_map = _build_id_map(store)
    assert id_map[(doc_b, idx_b)] == id_b  # (document_id, chunk_index) -> global id

    # Deterministic stub search surfacing Doc B's chunk — avoids depending on the
    # hashing embedder's near-random ranking on a tiny corpus.
    def fake_search(query: str, top_k: int):
        return [
            SearchHit(
                document_id=doc_b,
                document_title="Doc B",
                chunk_index=idx_b,
                text="Бета: про калибровку манометра раз в год.",
                score=0.99,
            )
        ][:top_k]

    retriever = make_retriever(fake_search, id_map)

    seed = QAPair(
        instruction="Как часто калибруют манометр?",
        input="",
        output=f"Раз в год. [doc_chunk:{id_b}]",
        source_chunk_id=id_b,  # GLOBAL id, not the per-document ordinal (0)
    )

    sample = build_relevant_sample(seed, retriever=retriever, top_k=3)
    assert sample is not None
    assert sample.retrieved_chunk_ids == (id_b,)
    assert f"[doc_chunk:{id_b}]" in sample.retrieved_context
