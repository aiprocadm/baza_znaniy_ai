from experiments.pravo_nn.wiki_collector.assemble import (
    accumulate,
    build_manifest,
    write_wiki,
)


def test_accumulate_stops_at_target_and_dedupes():
    arts = [
        ("A", "x" * 100),
        ("A", "x" * 100),   # duplicate title — skipped
        ("B", "y" * 100),
        ("C", "z" * 100),   # should not be reached once target hit at B
    ]
    kept, total = accumulate(iter(arts), target_bytes=150)
    titles = [t for t, _ in kept]
    assert titles == ["A", "B"]      # dedup + stopped after crossing 150 bytes
    assert total >= 150


def test_write_wiki_and_manifest(tmp_path):
    kept = [("A", "альфа текст"), ("B", "бета текст")]
    write_wiki(kept, tmp_path)
    assert (tmp_path / "wiki.txt").exists()
    body = (tmp_path / "wiki.txt").read_text(encoding="utf-8")
    assert "альфа текст" in body and "бета текст" in body

    manifest = build_manifest(kept, collected_at="2026-06-20", source="https://ru.wikipedia.org")
    assert manifest["articles"] == 2
    assert manifest["titles"] == ["A", "B"]
    assert manifest["total_bytes"] == len("альфа текст".encode()) + len("бета текст".encode())
