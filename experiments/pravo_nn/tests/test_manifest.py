from experiments.pravo_nn.corpus_collector.assemble import build_manifest
from experiments.pravo_nn.corpus_collector.extract import Article


def test_manifest_counts_and_bytes_match_documents():
    big = "т" * 600  # over MIN_CODE_CHARS
    per_code = {
        "ГК РФ": [Article("ГК РФ", "Статья 1", big, "http://gk", "1994-11-30")],
        "УК РФ": [Article("УК РФ", "Статья 1", big, "http://uk", "1996-06-13")],
    }
    m = build_manifest(per_code, collected_at="2026-06-14", source="mirror")
    assert m["total_documents"] == 2
    assert m["source"] == "mirror"
    assert m["total_bytes"] == sum(d["bytes"] for d in m["documents"])
    assert {d["code"] for d in m["documents"]} == {"ГК РФ", "УК РФ"}
    assert all(len(d["md5"]) == 32 for d in m["documents"])


def test_manifest_flags_suspiciously_small_doc():
    per_code = {"ВзК РФ": [Article("ВзК РФ", "Статья 1", "крошечный", "http://vzk", "")]}
    m = build_manifest(per_code, collected_at="2026-06-14", source="mirror")
    doc = m["documents"][0]
    assert doc["suspiciously_small"] is True  # PDF/OCR canary fired
