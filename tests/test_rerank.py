from __future__ import annotations

from typing import Any

import pytest

from app.retriever.rerank import (
    CrossEncoderReranker,
    apply_rerank,
    get_rerank_top_k,
    is_rerank_enabled,
)


def test_is_rerank_enabled_supports_truthy_values() -> None:
    assert is_rerank_enabled({"RERANK_ENABLED": "true"}) is True
    assert is_rerank_enabled({"RERANK_ENABLED": "0"}, default=True) is False
    assert is_rerank_enabled({}, default=True) is True


def test_get_rerank_top_k_handles_invalid_values() -> None:
    assert get_rerank_top_k({"RERANK_TOP_K": "15"}) == 15
    assert get_rerank_top_k({"RERANK_TOP_K": ""}, default=7) == 7
    assert get_rerank_top_k({"RERANK_TOP_K": "abc"}, default=5) == 5
    assert get_rerank_top_k({"RERANK_TOPK": "3"}, default=2) == 3


def test_cross_encoder_reranker_updates_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_args: dict[str, Any] = {}

    class DummyModel:
        def predict(self, pairs: list[tuple[str, str]]):
            recorded_args["pairs"] = list(pairs)
            return [0.2, 0.9, -0.1]

    monkeypatch.setattr(
        "app.retriever.rerank.CrossEncoder",
        lambda model_name: DummyModel(),
    )

    reranker = CrossEncoderReranker()
    hits = [
        {"text": "one", "score": 0.5},
        {"text": "two", "score": 0.4},
        {"text": "three", "score": 0.3},
    ]

    result = reranker.rerank("query", hits, top_k=2)

    assert recorded_args["pairs"] == [
        ("query", "one"),
        ("query", "two"),
        ("query", "three"),
    ]
    assert [hit["score"] for hit in result] == [0.9, 0.2]


def test_apply_rerank_uses_reranker_when_enabled() -> None:
    class DummyReranker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def rerank(
            self,
            query: str,
            hits: list[dict[str, Any]],
            top_k: int,
        ) -> list[dict[str, Any]]:
            self.calls.append((query, top_k))
            return [dict(hits[1], score=42.0)]

    reranker = DummyReranker()
    hits = [
        {"text": "first", "score": 0.1},
        {"text": "second", "score": 0.2},
    ]

    result = apply_rerank("question", hits, top_k=1, enabled=True, reranker=reranker)

    assert reranker.calls == [("question", 1)]
    assert result == [{**hits[1], "score": 42.0}]


def test_apply_rerank_returns_slice_when_disabled() -> None:
    class DummyReranker:
        def rerank(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
            raise AssertionError("Should not be called when disabled")

    hits = [
        {"text": "first", "score": 0.1},
        {"text": "second", "score": 0.2},
        {"text": "third", "score": 0.3},
    ]

    result = apply_rerank("question", hits, top_k=2, enabled=False, reranker=DummyReranker())

    assert result == hits[:2]


def test_apply_rerank_handles_empty_hits() -> None:
    assert apply_rerank("q", [], top_k=5, enabled=True, reranker=None) == []
