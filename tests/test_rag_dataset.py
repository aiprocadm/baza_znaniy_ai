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
