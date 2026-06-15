import json
from pathlib import Path

from scripts.ingest_pravo import article_slug, iter_articles


def test_article_slug_is_unique_and_collapses_whitespace():
    s1 = article_slug("ГК РФ ч.1", 12)
    s2 = article_slug("УК РФ", 12)
    assert s1 != s2
    assert " " not in s1
    assert s1 == "ГК_РФ_ч.1__a00012"


def test_iter_articles_parses_jsonl(tmp_path: Path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(
        json.dumps({"code": "ГК РФ ч.1", "article": "Статья 1. X", "text": "тело"}, ensure_ascii=False)
        + "\n\n"
        + json.dumps({"code": "ГК РФ ч.1", "article": "Статья 2. Y", "text": "тело2"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    arts = list(iter_articles(p))
    assert [a["article"] for a in arts] == ["Статья 1. X", "Статья 2. Y"]
