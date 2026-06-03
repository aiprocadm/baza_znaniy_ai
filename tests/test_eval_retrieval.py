from app.eval.adapter import EvalHit
from app.eval.dataset import GoldenItem
from app.eval.retrieval_eval import evaluate


def _retriever(mapping):
    return lambda q, k: [EvalHit(cid, "t") for cid in mapping.get(q, [])][:k]


def test_evaluate_aggregates_over_items():
    items = [GoldenItem("q1", (7,)), GoldenItem("q2", (9,))]
    retriever = _retriever({"q1": [7, 1, 2], "q2": [1, 2, 3]})  # q1 hits, q2 misses
    result = evaluate(items, retriever)
    assert result["n"] == 2
    assert result["aggregate"]["hit@1"] == 0.5  # only q1 hits at rank 1
    assert result["aggregate"]["mrr@5"] == 0.5  # (1.0 + 0.0) / 2
    assert len(result["per_item"]) == 2


def test_evaluate_empty_items():
    assert evaluate([], _retriever({}))["aggregate"] == {}
