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
    """Minimal stand-in for app.services.kb_store.SearchHit."""

    chunk_index: int
    text: str
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
                _FakeHit(chunk_index=7, text="Отпуск — это перерыв."),
                _FakeHit(chunk_index=12, text="Сотрудник имеет право."),
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
        {"Что такое отпуск?": [_FakeHit(chunk_index=99, text="Совсем не про отпуск.")]}
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
        _FakeHit(chunk_index=200, text="Калибровка манометра — раз в год."),
        _FakeHit(chunk_index=201, text="Поверка средств измерения."),
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
    seed_hit = _FakeHit(chunk_index=7, text="Отпуск — это перерыв в работе.")
    distractors = [
        _FakeHit(chunk_index=200, text="Калибровка манометра — раз в год."),
        _FakeHit(chunk_index=201, text="Поверка средств измерения."),
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
    seed_hits = {i: _FakeHit(chunk_index=i, text=f"Текст {i}") for i in range(1, 21)}

    def retriever(query: str, top_k: int):
        i = int(query.split()[1].rstrip("?"))
        return [seed_hits[i]]

    negatives = [_FakeHit(chunk_index=900 + j, text=f"Шум {j}") for j in range(5)]
    distractors = [_FakeHit(chunk_index=800 + j, text=f"Помеха {j}") for j in range(5)]

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
        negative_pool=[_FakeHit(chunk_index=900, text="нет")],
        distractor_pool=[_FakeHit(chunk_index=800, text="нет")],
        proportions=default_proportions(),
    )

    # We asked for 2 — both would have been RELEVANT but retrieval failed;
    # the builder is allowed to return fewer than requested.
    samples = list(builder.build(seeds, total=2))
    for s in samples:
        assert s.variant.value != "relevant"
