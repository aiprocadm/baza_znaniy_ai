"""Pure-function tests for the mr-TyDi stage-1 builder (no ML deps)."""

from pathlib import Path

import pytest

import scripts.build_mrtydi_pairs as mrtydi
from scripts.build_mrtydi_pairs import record_to_texts, to_pairs


def test_to_pairs_emits_positive_then_negatives_with_binary_scores():
    rows = to_pairs("какой срок", "три года", ["ставка налога", "состав суда"])
    assert rows == [
        {"query": "какой срок", "text": "три года", "teacher_score": 1.0},
        {"query": "какой срок", "text": "ставка налога", "teacher_score": 0.0},
        {"query": "какой срок", "text": "состав суда", "teacher_score": 0.0},
    ]


def test_to_pairs_skips_blank_query_or_positive():
    assert to_pairs("", "pos", ["neg"]) == []
    assert to_pairs("q", "", ["neg"]) == []


def test_to_pairs_drops_blank_negatives_individually():
    rows = to_pairs("q", "pos", ["", "real neg", "   "])
    assert rows == [
        {"query": "q", "text": "pos", "teacher_score": 1.0},
        {"query": "q", "text": "real neg", "teacher_score": 0.0},
    ]


def test_record_to_texts_extracts_first_positive_and_caps_negatives():
    record = {
        "query": "вопрос",
        "positive_passages": [{"docid": "a", "text": "позитив", "title": "t"}],
        "negative_passages": [
            {"docid": "b", "text": "neg1", "title": "t"},
            {"docid": "c", "text": "neg2", "title": "t"},
            {"docid": "d", "text": "neg3", "title": "t"},
        ],
    }
    assert record_to_texts(record, max_negs=2) == ("вопрос", "позитив", ["neg1", "neg2"])


def test_record_to_texts_handles_missing_positive():
    record = {"query": "q", "positive_passages": [], "negative_passages": []}
    assert record_to_texts(record, max_negs=5) == ("q", "", [])


from scripts.build_mrtydi_pairs import take_first


def test_take_first_yields_at_most_limit_in_order():
    assert list(take_first(["a", "b", "c", "d"], 2)) == ["a", "b"]


def test_take_first_handles_fewer_than_limit():
    assert list(take_first(["a", "b"], 10)) == ["a", "b"]


def test_main_fails_loud_and_leaves_no_file_on_empty_stream(tmp_path: Path, monkeypatch):
    # An empty mr-TyDi stream (network/version issue) must NOT leave a 0-row file:
    # the turnkey runner would trust it as "already mined" and skip re-mining.
    out = tmp_path / "mrtydi_pairs.jsonl"
    monkeypatch.setattr(mrtydi, "iter_records", lambda limit, *, max_negs: iter(()))
    with pytest.raises(SystemExit, match="No mr-TyDi rows"):
        mrtydi.main(["--out", str(out)])
    assert not out.exists()  # poisoned empty file removed -> resume re-mines next run


def test_main_writes_rows_when_stream_has_records(tmp_path: Path, monkeypatch):
    out = tmp_path / "mrtydi_pairs.jsonl"
    monkeypatch.setattr(
        mrtydi,
        "iter_records",
        lambda limit, *, max_negs: iter([("вопрос", "позитив", ["негатив"])]),
    )
    mrtydi.main(["--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # 1 positive + 1 negative row


def test_main_leaves_no_final_file_when_stream_crashes_midway(tmp_path: Path, monkeypatch):
    # A kill/exception mid-stream must NOT leave a partial FINAL file: the runner
    # trusts out's existence as "already mined" and would skip re-mining, training
    # stage-1 on a truncated set. The partial only ever lives in the temp file.
    out = tmp_path / "mrtydi_pairs.jsonl"

    def _crashing_stream(limit, *, max_negs):
        yield ("вопрос", "позитив", ["негатив"])  # one record lands in the temp file
        raise RuntimeError("stream died mid-way")

    monkeypatch.setattr(mrtydi, "iter_records", _crashing_stream)
    with pytest.raises(RuntimeError, match="stream died"):
        mrtydi.main(["--out", str(out)])
    assert not out.exists()  # no partial final file -> runner re-mines next run
    assert not (tmp_path / "mrtydi_pairs.jsonl.tmp").exists()  # temp cleaned up too
