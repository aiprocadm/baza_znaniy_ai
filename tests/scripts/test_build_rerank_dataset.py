"""Pure-function tests for the distillation dataset builder (no ML deps)."""

import json

import pytest

from scripts.build_rerank_dataset import (
    Pair,
    as_retrieve,
    build_pairs,
    dedupe_queries,
    normalize_question,
    write_pairs,
)


def _retrieve(query: str, k: int):
    return [(f"doc.md:{i}", f"text {i} for {query}") for i in range(k)]


def test_normalize_question_strips_case_space_punctuation():
    assert normalize_question("  Какой Срок?  ") == normalize_question("какой срок")
    assert normalize_question("Что это?!") == "что это"
    assert normalize_question("Что это…") == "что это"


def test_build_pairs_excludes_golden_queries():
    queries = [("Сколько дней отпуск?", "a.md:1"), ("Уникальный вопрос?", "a.md:2")]
    golden = frozenset({"Сколько дней отпуск?"})
    pairs = build_pairs(queries, _retrieve, golden, k=3)
    assert {p.query for p in pairs} == {"Уникальный вопрос?"}


def test_build_pairs_yields_k_candidates_per_query():
    pairs = build_pairs([("вопрос?", "a.md:0")], _retrieve, frozenset(), k=5)
    assert len(pairs) == 5
    assert pairs[0] == Pair(query="вопрос?", chunk_key="doc.md:0", text="text 0 for вопрос?")


def test_build_pairs_filters_normalized_golden_variants():
    # Spec §3.4: the leak filter must catch case/punctuation variants, not
    # just exact matches.
    queries = [("сколько ДНЕЙ отпуск", "a.md:1")]
    golden = frozenset({"Сколько дней отпуск?"})
    assert build_pairs(queries, _retrieve, golden, k=2) == []


def test_write_pairs_roundtrip(tmp_path):
    pairs = [Pair(query="q", chunk_key="d.md:0", text="t")]
    out = tmp_path / "pairs.jsonl"
    write_pairs(out, pairs, scores=[0.75], meta={"teacher": "x"})
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row == {"query": "q", "chunk_key": "d.md:0", "text": "t", "teacher_score": 0.75}
    meta = json.loads((tmp_path / "pairs.meta.json").read_text(encoding="utf-8"))
    assert meta["teacher"] == "x"


def test_write_pairs_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        write_pairs(tmp_path / "p.jsonl", [Pair("q", "k", "t")], scores=[], meta={})


class _Hit:
    def __init__(self, chunk_key: str, text: str):
        self.chunk_key = chunk_key
        self.text = text


def test_as_retrieve_adapts_eval_retriever_to_tuples():
    def eval_retriever(query: str, k: int):
        return [_Hit(f"d.md:{i}", f"t{i}") for i in range(k)]

    retrieve = as_retrieve(eval_retriever)
    assert retrieve("q", 2) == [("d.md:0", "t0"), ("d.md:1", "t1")]


def test_dedupe_queries_by_normalized_text_keeps_first():
    queries = [("Какой срок?", "a.md:0"), ("какой СРОК", "b.md:1"), ("Другой?", "c.md:2")]
    assert dedupe_queries(queries) == [("Какой срок?", "a.md:0"), ("Другой?", "c.md:2")]
