import json

from experiments.pravo_nn.corpus_collector import cli
from experiments.pravo_nn.corpus_collector.config import CodeSpec


def test_collect_runs_full_pipeline_offline(tmp_path, monkeypatch):
    codes = (CodeSpec("ГК РФ", "gk-rf"), CodeSpec("УК РФ", "uk-rf"))
    raw_by_slug = {
        "gk-rf": "Статья 1\n" + "г" * 600 + "\nСтатья 2\nещё",
        "uk-rf": "Статья 1\n" + "у" * 600,
    }

    def fake_fetch(spec, *, source_base, cache_dir, **kw):
        return raw_by_slug[spec.slug]

    monkeypatch.setattr(cli.fetch, "fetch_raw", fake_fetch)

    cli.collect(
        codes=codes,
        source_base="http://src",
        source_label="mirror",
        data_dir=tmp_path,
        collected_at="2026-06-14",
    )

    corpus = (tmp_path / "corpus" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 3  # 2 articles in ГК + 1 in УК
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_documents"] == 2
    assert manifest["collected_at"] == "2026-06-14"
    assert all(d["suspiciously_small"] is False for d in manifest["documents"])
