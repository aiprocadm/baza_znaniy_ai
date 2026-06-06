from app.eval.adapter import EvalHit
from app.eval.dataset import GoldenItem
from app.eval.retrieval_eval import evaluate


def _retriever(mapping):
    return lambda q, k: [EvalHit(cid, "t") for cid in mapping.get(q, [])][:k]


def test_evaluate_aggregates_over_items():
    items = [GoldenItem("q1", ("7",)), GoldenItem("q2", ("9",))]
    retriever = _retriever({"q1": ["7", "1", "2"], "q2": ["1", "2", "3"]})  # q1 hits, q2 misses
    result = evaluate(items, retriever)
    assert result["n"] == 2
    assert result["aggregate"]["hit@1"] == 0.5  # only q1 hits at rank 1
    assert result["aggregate"]["mrr@5"] == 0.5  # (1.0 + 0.0) / 2
    assert len(result["per_item"]) == 2


def test_evaluate_empty_items():
    assert evaluate([], _retriever({}))["aggregate"] == {}


def test_evaluate_threads_top_k_as_retrieval_depth():
    seen_k: list[int] = []

    def spy(q, k):
        seen_k.append(k)
        return [EvalHit("f.md:1", "t")]

    evaluate([GoldenItem("q", ("f.md:1",))], spy, top_k=7)
    assert seen_k == [7]


def test_evaluate_defaults_retrieval_depth_to_max_ks():
    seen_k: list[int] = []

    def spy(q, k):
        seen_k.append(k)
        return [EvalHit("f.md:1", "t")]

    evaluate([GoldenItem("q", ("f.md:1",))], spy)
    assert seen_k == [10]  # max(RETRIEVAL_KS) == 10


def test_evaluate_item_scores_on_chunk_keys():
    from app.eval.dataset import GoldenItem
    from app.eval.adapter import EvalHit
    from app.eval.retrieval_eval import evaluate_item

    item = GoldenItem("q", relevant_chunks=("f.md:2",), reference_answer="")

    def retriever(q, k):
        return [EvalHit("f.md:0", "a"), EvalHit("f.md:2", "b")][:k]

    scores = evaluate_item(item, retriever, max_k=5)
    assert scores["hit@5"] == 1.0
    assert scores["hit@1"] == 0.0
