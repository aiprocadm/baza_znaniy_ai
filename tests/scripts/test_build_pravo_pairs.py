"""Pure-function tests for the structural pravo miner (no ML deps)."""

from scripts.build_pravo_pairs import articles_to_queries


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
