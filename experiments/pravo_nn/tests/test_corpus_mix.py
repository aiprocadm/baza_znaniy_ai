from experiments.pravo_nn.corpus_mix.assemble import mix_corpora


def test_mix_is_balanced_by_bytes():
    law = "закон " * 1000      # large
    wiki = "статья " * 100     # small
    mixed, manifest = mix_corpora(law, wiki)
    # the larger source is truncated to the smaller's budget -> roughly equal
    assert abs(manifest["law_bytes"] - manifest["wiki_bytes"]) <= len("закон ".encode())
    assert "закон" in mixed and "статья" in mixed


def test_mix_keeps_all_when_already_equal():
    law = "ё" * 100
    wiki = "я" * 100
    mixed, manifest = mix_corpora(law, wiki)
    assert manifest["law_bytes"] == len(law.encode("utf-8"))
    assert manifest["wiki_bytes"] == len(wiki.encode("utf-8"))
