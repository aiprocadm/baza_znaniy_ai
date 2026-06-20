"""Data-plumbing tests for the distillation trainer. The training loop itself
is exercised by the manual gate run (plan Task 6), not by unit tests — it
needs torch+transformers which CI stubs out."""

import json

import pytest

from scripts.train_reranker import (
    enumerate_pairs,
    group_by_query,
    load_pairs,
    query_grouped_batches,
    select_device,
    soft_label,
    split_by_query,
)


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


# --- device selection (CPU now, GPU later without a code edit) --------------


def test_select_device_prefers_cuda_when_available():
    assert select_device(cuda_available=True) == "cuda"


def test_select_device_falls_back_to_cpu():
    assert select_device(cuda_available=False) == "cpu"


def test_select_device_override_wins_over_autodetect():
    # An explicit --device wins even if CUDA is present (e.g. forcing a CPU
    # repro on a GPU box).
    assert select_device(cuda_available=True, override="cpu") == "cpu"
    assert select_device(cuda_available=False, override="cuda") == "cuda"


# --- pairwise loss helpers (used only by --loss pairwise) -------------------


def test_group_by_query_groups_and_preserves_order():
    rows = [
        {"query": "a", "text": "a1", "teacher_score": 0.1},
        {"query": "b", "text": "b1", "teacher_score": 0.2},
        {"query": "a", "text": "a2", "teacher_score": 0.3},
    ]
    groups = group_by_query(rows)
    assert [[r["text"] for r in g] for g in groups] == [["a1", "a2"], ["b1"]]
    # every input row is preserved exactly once
    assert sum(len(g) for g in groups) == len(rows)


def test_enumerate_pairs_orders_positive_first():
    group = [
        {"teacher_score": 0.9},  # 0
        {"teacher_score": 0.1},  # 1
        {"teacher_score": 0.5},  # 2
    ]
    pairs = enumerate_pairs(group)
    # (i, j) means score_i > score_j; deterministic ascending order
    assert pairs == [(0, 1), (0, 2), (2, 1)]
    for i, j in pairs:
        assert group[i]["teacher_score"] > group[j]["teacher_score"]


def test_enumerate_pairs_no_pairs_when_all_equal():
    group = [{"teacher_score": 0.5}, {"teacher_score": 0.5}, {"teacher_score": 0.5}]
    assert enumerate_pairs(group) == []


def test_enumerate_pairs_margin_suppresses_near_ties():
    group = [{"teacher_score": 0.50}, {"teacher_score": 0.55}, {"teacher_score": 0.90}]
    # with a 0.1 margin only the 0.90 vs others gaps survive
    pairs = enumerate_pairs(group, margin=0.1)
    assert pairs == [(2, 0), (2, 1)]


def test_enumerate_pairs_uses_soft_label_for_raw_logits():
    # raw logits get sigmoid'd: 4.0 -> ~0.98 outranks -4.0 -> ~0.018
    group = [{"teacher_score": -4.0}, {"teacher_score": 4.0}]
    assert enumerate_pairs(group) == [(1, 0)]


def test_query_grouped_batches_deterministic_and_drops_unrankable():
    rows = (
        [{"query": "a", "teacher_score": s} for s in (0.1, 0.9)]
        + [{"query": "b", "teacher_score": s} for s in (0.5, 0.5)]  # no pairs -> dropped
        + [{"query": "c", "teacher_score": s} for s in (0.2, 0.8)]
    )
    b1 = query_grouped_batches(rows, seed=7)
    b2 = query_grouped_batches(rows, seed=7)
    assert b1 == b2  # deterministic
    queries = {g[0]["query"] for g in b1}
    assert queries == {"a", "c"}  # all-equal "b" dropped
    assert all(len(enumerate_pairs(g)) >= 1 for g in b1)


def test_init_from_defaults_to_base_model(monkeypatch, tmp_path):
    import scripts.train_reranker as tr

    captured = {}

    def fake_train(rows_train, rows_val, **kw):
        captured.update(kw)
        return {"val_pairs": 0, "val_pearson_vs_teacher": 0.0, "device": "cpu"}

    monkeypatch.setattr(tr, "train", fake_train)
    monkeypatch.setattr(
        tr,
        "load_pairs",
        lambda p: [
            {"query": "a", "text": "t", "teacher_score": 1.0},
            {"query": "b", "text": "u", "teacher_score": 0.0},
        ],
    )
    tr.main(["--pairs", "x.jsonl", "--out", str(tmp_path / "o"), "--epochs", "1"])
    assert captured["init_from"] == "cointegrated/rubert-tiny2"


def test_init_from_override_is_passed_through(monkeypatch, tmp_path):
    import scripts.train_reranker as tr

    captured = {}
    monkeypatch.setattr(
        tr,
        "train",
        lambda rt, rv, **kw: captured.update(kw)
        or {"val_pairs": 0, "val_pearson_vs_teacher": 0.0, "device": "cpu"},
    )
    monkeypatch.setattr(
        tr,
        "load_pairs",
        lambda p: [
            {"query": "a", "text": "t", "teacher_score": 1.0},
            {"query": "b", "text": "u", "teacher_score": 0.0},
        ],
    )
    tr.main(
        ["--pairs", "x.jsonl", "--out", str(tmp_path / "o"), "--init-from", "var/models/stage1"]
    )
    assert captured["init_from"] == "var/models/stage1"
