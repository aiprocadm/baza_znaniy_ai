"""Tests for the CPU-latency gate helpers (fake scorer — no model load)."""

import json

import pytest

from scripts.bench_reranker import group_queries, measure, percentile


def test_group_queries_collects_texts_per_query(tmp_path):
    rows = [
        {"query": "a", "text": "t1", "teacher_score": 0.1},
        {"query": "a", "text": "t2", "teacher_score": 0.2},
        {"query": "b", "text": "t3", "teacher_score": 0.3},
    ]
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert group_queries(p) == {"a": ["t1", "t2"], "b": ["t3"]}


def test_measure_calls_scorer_once_per_query_with_capped_candidates():
    calls: list[list[tuple[str, str]]] = []

    def fake_score(pairs):
        calls.append(list(pairs))
        return [0.0] * len(pairs)

    timings = measure(fake_score, [("q", ["t1", "t2", "t3"])], candidates=2)
    assert len(timings) == 1
    assert calls == [[("q", "t1"), ("q", "t2")]]
    assert timings[0] >= 0.0


def test_percentile_p95_and_median():
    timings = [float(v) for v in range(1, 101)]
    assert percentile(timings, 0.50) == pytest.approx(50.0, abs=1.0)
    assert percentile(timings, 0.95) == pytest.approx(95.0, abs=1.0)
