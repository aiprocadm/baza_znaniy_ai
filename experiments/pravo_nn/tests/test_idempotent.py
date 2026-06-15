from experiments.pravo_nn.corpus_collector.assemble import write_corpus
from experiments.pravo_nn.corpus_collector.extract import Article


def test_rewriting_same_articles_produces_identical_bytes(tmp_path):
    arts = [
        Article("ГК РФ", "Статья 1", "тело один", "http://x", "1994-11-30"),
        Article("ГК РФ", "Статья 2", "тело два", "http://x", "1994-11-30"),
    ]
    write_corpus(arts, tmp_path)
    first = (tmp_path / "corpus.jsonl").read_bytes()
    write_corpus(arts, tmp_path)  # second run over the same input
    second = (tmp_path / "corpus.jsonl").read_bytes()
    assert first == second  # corpus files carry no timestamps -> deterministic
