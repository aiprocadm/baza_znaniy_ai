"""Pure-function tests for the structural pravo miner (no ML deps)."""

import json as _json

from scripts.build_pravo_pairs import articles_to_queries, load_golden_questions


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


def test_load_golden_questions_reads_question_field(tmp_path):
    p = tmp_path / "golden_pravo.jsonl"
    p.write_text(
        _json.dumps({"question": "Общий срок исковой давности", "relevant_chunks": ["a.md:0"]}) + "\n"
        + _json.dumps({"question": "Специальные сроки", "relevant_chunks": ["b.md:0"]}) + "\n",
        encoding="utf-8",
    )
    assert load_golden_questions(p) == frozenset(
        {"Общий срок исковой давности", "Специальные сроки"}
    )


def test_load_golden_questions_missing_file_is_empty(tmp_path):
    assert load_golden_questions(tmp_path / "nope.jsonl") == frozenset()
