"""Pure-function tests for the distillation dataset builder (no ML deps)."""

import json

import pytest

from scripts.build_rerank_dataset import (
    Pair,
    append_rows,
    as_retrieve,
    build_pairs,
    completed_source_keys,
    count_rows,
    dedupe_queries,
    filter_done_queries,
    group_by_source,
    normalize_question,
    score_and_flush_by_chunk,
    select_chunks,
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
    # source_key threads the originating chunk through for resume bookkeeping.
    assert pairs[0] == Pair(
        query="вопрос?", chunk_key="doc.md:0", text="text 0 for вопрос?", source_key="a.md:0"
    )


def test_build_pairs_filters_normalized_golden_variants():
    # Spec §3.4: the leak filter must catch case/punctuation variants, not
    # just exact matches.
    queries = [("сколько ДНЕЙ отпуск", "a.md:1")]
    golden = frozenset({"Сколько дней отпуск?"})
    assert build_pairs(queries, _retrieve, golden, k=2) == []


def test_write_pairs_roundtrip(tmp_path):
    pairs = [Pair(query="q", chunk_key="d.md:0", text="t", source_key="s.md:0")]
    out = tmp_path / "pairs.jsonl"
    write_pairs(out, pairs, scores=[0.75], meta={"teacher": "x"})
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row == {
        "query": "q",
        "chunk_key": "d.md:0",
        "text": "t",
        "teacher_score": 0.75,
        "source_key": "s.md:0",
    }
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


def test_select_chunks_stride_samples_evenly():
    chunks = list(range(10))
    assert select_chunks(chunks, stride=3) == [0, 3, 6, 9]


def test_select_chunks_stride_then_limit():
    chunks = list(range(10))
    assert select_chunks(chunks, stride=2, limit=3) == [0, 2, 4]


def test_select_chunks_defaults_passthrough():
    chunks = list(range(3))
    assert select_chunks(chunks) == [0, 1, 2]


def test_select_chunks_offset_shifts_the_stride_window():
    chunks = list(range(10))
    assert select_chunks(chunks, stride=3, offset=1) == [1, 4, 7]


def test_select_chunks_offset_without_stride_skips_prefix():
    chunks = list(range(5))
    assert select_chunks(chunks, offset=2) == [2, 3, 4]


# --- resume / checkpointing helpers (used only by --resume) -----------------


def test_append_rows_appends_without_truncating(tmp_path):
    out = tmp_path / "pairs.jsonl"
    append_rows(out, [Pair("q1", "c.md:0", "t1", "s.md:0")], scores=[0.1])
    append_rows(out, [Pair("q2", "c.md:1", "t2", "s.md:1")], scores=[0.2])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["query"] for r in rows] == ["q1", "q2"]
    assert [r["source_key"] for r in rows] == ["s.md:0", "s.md:1"]


def test_append_rows_noop_on_empty(tmp_path):
    out = tmp_path / "pairs.jsonl"
    append_rows(out, [], scores=[])
    assert not out.exists()


def test_append_rows_rejects_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        append_rows(tmp_path / "p.jsonl", [Pair("q", "k", "t", "s")], scores=[])


def test_completed_source_keys_collects_done_chunks(tmp_path):
    out = tmp_path / "pairs.jsonl"
    append_rows(
        out,
        [Pair("q1", "c.md:0", "t", "s.md:0"), Pair("q2", "c.md:1", "t", "s.md:1")],
        scores=[0.1, 0.2],
    )
    assert completed_source_keys(out) == {"s.md:0", "s.md:1"}


def test_completed_source_keys_missing_file_is_empty(tmp_path):
    assert completed_source_keys(tmp_path / "absent.jsonl") == set()


def test_completed_source_keys_tolerates_torn_final_line(tmp_path):
    out = tmp_path / "pairs.jsonl"
    append_rows(out, [Pair("q1", "c.md:0", "t", "s.md:0")], scores=[0.1])
    with out.open("a", encoding="utf-8") as fh:
        fh.write('{"query": "partial", "source_key": "s.md:9"')  # torn, no newline/brace
    assert completed_source_keys(out) == {"s.md:0"}


def test_completed_source_keys_ignores_legacy_rows_without_source_key(tmp_path):
    out = tmp_path / "pairs.jsonl"
    out.write_text(
        json.dumps({"query": "q", "chunk_key": "c.md:0", "text": "t", "teacher_score": 0.1}) + "\n",
        encoding="utf-8",
    )
    assert completed_source_keys(out) == set()


def test_filter_done_queries_drops_completed_sources():
    queries = [("q1", "s.md:0"), ("q2", "s.md:1"), ("q3", "s.md:0")]
    assert filter_done_queries(queries, {"s.md:0"}) == [("q2", "s.md:1")]


def test_count_rows_counts_nonblank(tmp_path):
    out = tmp_path / "pairs.jsonl"
    out.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
    assert count_rows(out) == 2
    assert count_rows(tmp_path / "absent.jsonl") == 0


def test_group_by_source_groups_preserving_first_seen_order():
    queries = [("q1", "s.md:1"), ("q2", "s.md:0"), ("q3", "s.md:1")]
    assert group_by_source(queries) == [("s.md:1", ["q1", "q3"]), ("s.md:0", ["q2"])]


def test_score_and_flush_by_chunk_mines_scores_and_appends(tmp_path):
    out = tmp_path / "pairs.jsonl"
    queries = [("q1", "s.md:0"), ("q2", "s.md:0"), ("q3", "s.md:1")]

    def score_fn(pairs):
        return [0.5] * len(pairs)

    new_pairs = score_and_flush_by_chunk(queries, _retrieve, frozenset(), score_fn, out=out, k=2)
    # 3 queries x 2 candidates = 6 pairs, all flushed to disk.
    assert new_pairs == 6
    assert count_rows(out) == 6
    assert completed_source_keys(out) == {"s.md:0", "s.md:1"}


def test_score_and_flush_by_chunk_skips_golden_only_chunks(tmp_path):
    out = tmp_path / "pairs.jsonl"
    queries = [("Секрет?", "s.md:0")]
    golden = frozenset({"Секрет?"})

    def score_fn(pairs):
        raise AssertionError("score_fn must not run for a golden-filtered chunk")

    assert score_and_flush_by_chunk(queries, _retrieve, golden, score_fn, out=out, k=2) == 0
    assert not out.exists()
