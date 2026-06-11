"""Data-plumbing tests for the distillation trainer. The training loop itself
is exercised by the manual gate run (plan Task 6), not by unit tests — it
needs torch+transformers which CI stubs out."""

import json

import pytest

from scripts.train_reranker import load_pairs, soft_label, split_by_query


def _write_pairs(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_pairs_reads_jsonl(tmp_path):
    p = tmp_path / "pairs.jsonl"
    _write_pairs(p, [{"query": "q", "text": "t", "teacher_score": 0.5}])
    assert load_pairs(p) == [{"query": "q", "text": "t", "teacher_score": 0.5}]


def test_load_pairs_rejects_empty(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_pairs(p)


def test_soft_label_passthrough_for_probabilities():
    assert soft_label(0.0) == 0.0
    assert soft_label(0.73) == 0.73
    assert soft_label(1.0) == 1.0


def test_soft_label_sigmoid_for_raw_logits():
    assert 0.95 < soft_label(4.0) < 1.0
    assert 0.0 < soft_label(-4.0) < 0.05


def test_split_by_query_is_query_disjoint_and_deterministic():
    rows = [{"query": f"q{i % 10}", "text": f"t{i}", "teacher_score": 0.1} for i in range(100)]
    train_a, val_a = split_by_query(rows, val_fraction=0.2, seed=42)
    train_b, val_b = split_by_query(rows, val_fraction=0.2, seed=42)
    assert (train_a, val_a) == (train_b, val_b)
    assert {r["query"] for r in train_a} & {r["query"] for r in val_a} == set()
    assert len(train_a) + len(val_a) == len(rows)
    assert val_a  # non-empty
