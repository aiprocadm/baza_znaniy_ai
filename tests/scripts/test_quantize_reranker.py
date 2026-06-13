"""Tests for the int8-quantization latency path -- pure helpers only.

No model load, no torch in the test body (mirrors test_bench_reranker.py).
"""

from scripts.quantize_reranker import select_queries, split_warmup, summarize


def test_select_queries_filters_by_candidate_count_and_caps():
    grouped = {
        "q1": ["a", "b", "c"],
        "q2": ["d"],  # too few
        "q3": ["e", "f"],
        "q4": ["g", "h", "i"],
    }
    selected = select_queries(grouped, candidates=2, limit=10)
    assert selected == [("q1", ["a", "b", "c"]), ("q3", ["e", "f"]), ("q4", ["g", "h", "i"])]


def test_select_queries_respects_limit():
    grouped = {f"q{i}": ["x", "y"] for i in range(5)}
    assert len(select_queries(grouped, candidates=2, limit=3)) == 3


def test_select_queries_empty_when_none_qualify():
    grouped = {"q1": ["only-one"]}
    assert select_queries(grouped, candidates=5, limit=10) == []


def test_split_warmup_separates_warm_and_timed():
    sample = [("q1", ["t"]), ("q2", ["t"]), ("q3", ["t"]), ("q4", ["t"])]
    warm, timed = split_warmup(sample, warmup=2)
    assert warm == [("q1", ["t"]), ("q2", ["t"])]
    assert timed == [("q3", ["t"]), ("q4", ["t"])]


def test_split_warmup_falls_back_to_full_sample_when_too_small():
    sample = [("q1", ["t"]), ("q2", ["t"])]
    warm, timed = split_warmup(sample, warmup=2)
    assert warm == sample
    assert timed == sample  # nothing left over -> reuse whole sample for timing


def test_split_warmup_zero_warmup_times_everything():
    sample = [("q1", ["t"]), ("q2", ["t"])]
    warm, timed = split_warmup(sample, warmup=0)
    assert warm == []
    assert timed == sample


def test_summarize_passes_when_p95_within_budget():
    timings = [10.0, 20.0, 30.0, 40.0, 50.0]
    p50, p95, passed = summarize(timings, budget_ms=200.0)
    assert p50 == 30.0
    assert p95 == 50.0
    assert passed is True


def test_summarize_fails_when_p95_over_budget():
    timings = [100.0, 250.0, 300.0]
    _, p95, passed = summarize(timings, budget_ms=200.0)
    assert p95 == 300.0
    assert passed is False
