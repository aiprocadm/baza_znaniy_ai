import json
from app.eval.dataset import (
    GoldenItem,
    CorpusSignature,
    load_golden,
    save_golden,
    write_signature,
    read_signature,
)


def test_goldenitem_roundtrip():
    item = GoldenItem(
        question="Что такое отпуск?",
        relevant_chunks=("7", "12"),
        reference_answer="Это перерыв.",
        expect_refusal=False,
        source="curated",
    )
    back = GoldenItem.from_jsonl_line(item.to_jsonl_line())
    assert back == item


def test_reads_plain_qapair_line_back_compat():
    # A line emitted by synthetic_qa.QAPair (only source_chunk_id, no relevant_chunk_ids)
    line = json.dumps(
        {
            "instruction": "Q?",
            "input": "",
            "output": "A [doc_chunk:5]",
            "meta": {"source_chunk_id": 5},
        },
        ensure_ascii=False,
    )
    item = GoldenItem.from_jsonl_line(line)
    assert item.relevant_chunks == ("5",)
    assert item.reference_answer == "A [doc_chunk:5]"
    assert item.expect_refusal is False and item.source == "auto"


def test_save_and_load_golden(tmp_path):
    items = [
        GoldenItem("Q1", ("1",), "A1"),
        GoldenItem("Q2", (), "", expect_refusal=True, source="curated"),
    ]
    path = tmp_path / "golden.jsonl"
    save_golden(path, items)
    assert load_golden(path) == items


def test_signature_sidecar_roundtrip(tmp_path):
    path = tmp_path / "golden.jsonl"
    sig = CorpusSignature(doc_count=3, max_chunk_id=42, embedder_name="ollama", dim=384)
    write_signature(path, sig)
    assert read_signature(path) == sig
    assert (tmp_path / "golden.sig.json").exists()


def test_read_signature_missing_returns_none(tmp_path):
    assert read_signature(tmp_path / "nope.jsonl") is None


def test_golden_item_round_trips_composite_keys():
    from app.eval.dataset import GoldenItem
    item = GoldenItem(
        question="Сколько стоит услуга?",
        relevant_chunks=("contract.md:3", "contract.md:4"),
        reference_answer="45000",
        source="curated",
    )
    line = item.to_jsonl_line()
    back = GoldenItem.from_jsonl_line(line)
    assert back == item
    assert back.relevant_chunks == ("contract.md:3", "contract.md:4")


def test_golden_item_reads_legacy_int_labels_as_strings():
    # Old QAPair / int-labelled lines must still load (stringified, won't match
    # composite hits, but must not crash).
    from app.eval.dataset import GoldenItem
    legacy = '{"instruction":"q","input":"","output":"a","meta":{"relevant_chunk_ids":[7,12]}}'
    item = GoldenItem.from_jsonl_line(legacy)
    assert item.relevant_chunks == ("7", "12")

    legacy_qapair = '{"instruction":"q","input":"","output":"a","meta":{"source_chunk_id":7}}'
    assert GoldenItem.from_jsonl_line(legacy_qapair).relevant_chunks == ("7",)
