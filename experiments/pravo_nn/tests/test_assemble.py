import json

from experiments.pravo_nn.corpus_collector.assemble import write_corpus
from experiments.pravo_nn.corpus_collector.extract import Article


def _articles():
    return [
        Article("ГК РФ", "Статья 1", "тело один", "http://x", "1994-11-30"),
        Article("ГК РФ", "Статья 2", "тело два", "http://x", "1994-11-30"),
    ]


def test_write_corpus_emits_jsonl_one_object_per_article(tmp_path):
    write_corpus(_articles(), tmp_path)
    lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "code": "ГК РФ",
        "article": "Статья 1",
        "text": "тело один",
        "source_url": "http://x",
        "date": "1994-11-30",
    }


def test_write_corpus_emits_concatenated_txt(tmp_path):
    write_corpus(_articles(), tmp_path)
    txt = (tmp_path / "corpus.txt").read_text(encoding="utf-8")
    assert "Статья 1" in txt and "тело один" in txt and "тело два" in txt
