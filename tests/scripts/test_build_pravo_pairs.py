"""Pure-function tests for the structural pravo miner (no ML deps)."""

import json as _json

from scripts.build_pravo_pairs import (
    articles_to_queries,
    limit_queries,
    load_golden_questions,
    plan_output,
)


def test_articles_to_queries_uses_heading_topic_and_source_key():
    docs = [
        ("gk_rf_0001.md", "Статья 196. Общий срок исковой давности", [0]),
        ("gk_rf_0002.md", "Статья 197. Специальные сроки", [0, 1]),
    ]
    assert articles_to_queries(docs) == [
        ("Общий срок исковой давности", "gk_rf_0001.md"),
        ("Специальные сроки", "gk_rf_0002.md"),
    ]


def test_articles_to_queries_skips_empty_topic():
    # A heading with no topic after the «Статья N.» prefix yields no query.
    docs = [("x.md", "Статья 5.", [0]), ("y.md", "Статья 6. Тема", [0])]
    assert articles_to_queries(docs) == [("Тема", "y.md")]


def test_load_golden_questions_reads_instruction_field(tmp_path):
    p = tmp_path / "golden_pravo.jsonl"
    p.write_text(
        _json.dumps(
            {"instruction": "Общий срок исковой давности", "meta": {"relevant_chunks": ["a.md:0"]}}
        )
        + "\n"
        + _json.dumps({"instruction": "Специальные сроки", "meta": {"relevant_chunks": ["b.md:0"]}})
        + "\n",
        encoding="utf-8",
    )
    assert load_golden_questions(p) == frozenset(
        {"Общий срок исковой давности", "Специальные сроки"}
    )


def test_load_golden_questions_missing_file_is_empty(tmp_path):
    assert load_golden_questions(tmp_path / "nope.jsonl") == frozenset()


def test_limit_queries_caps_to_first_n():
    q = [("q1", "a"), ("q2", "b"), ("q3", "c")]
    assert limit_queries(q, 2) == [("q1", "a"), ("q2", "b")]


def test_limit_queries_zero_or_negative_returns_all():
    q = [("q1", "a"), ("q2", "b")]
    assert limit_queries(q, 0) == q
    assert limit_queries(q, -1) == q


def _row(query, source_key):
    return _json.dumps(
        {
            "query": query,
            "chunk_key": "x.md:0",
            "text": "t",
            "teacher_score": 0.5,
            "source_key": source_key,
        },
        ensure_ascii=False,
    )


def test_plan_output_no_resume_mines_all_and_is_fresh(tmp_path):
    # Without --resume the run is fresh even if an old file exists: mine every
    # query and let the caller unlink the stale output.
    out = tmp_path / "p.jsonl"
    out.write_text(_row("q1", "a.md") + "\n", encoding="utf-8")
    q = [("q1", "a.md"), ("q2", "b.md")]
    queries, fresh = plan_output(out, q, resume=False)
    assert fresh is True
    assert queries == q


def test_plan_output_resume_skips_completed_source_keys(tmp_path):
    # --resume against an existing file appends: queries whose source article is
    # already mined are dropped, the rest are returned, and fresh is False (no unlink).
    out = tmp_path / "p.jsonl"
    out.write_text(_row("q1", "a.md") + "\n", encoding="utf-8")
    q = [("q1", "a.md"), ("q2", "b.md"), ("q3", "c.md")]
    queries, fresh = plan_output(out, q, resume=True)
    assert fresh is False
    assert queries == [("q2", "b.md"), ("q3", "c.md")]


def test_plan_output_resume_without_file_is_fresh(tmp_path):
    out = tmp_path / "missing.jsonl"
    q = [("q1", "a.md")]
    queries, fresh = plan_output(out, q, resume=True)
    assert fresh is True
    assert queries == q
