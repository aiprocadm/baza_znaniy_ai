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
