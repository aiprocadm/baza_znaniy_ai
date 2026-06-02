from app.eval.metrics import hit_at_k, recall_at_k, mrr_at_k, score_item, aggregate, RETRIEVAL_KS


def test_hit_at_k_true_only_within_k():
    assert hit_at_k({7}, [3, 7, 1], 3) == 1.0
    assert hit_at_k({7}, [3, 1, 7], 2) == 0.0


def test_recall_at_k_fraction_of_relevant_found():
    assert recall_at_k({7, 9}, [7, 1, 2], 3) == 0.5
    assert recall_at_k({7, 9}, [7, 9, 2], 3) == 1.0


def test_mrr_at_k_uses_first_relevant_rank():
    assert mrr_at_k({9}, [1, 9, 3], 5) == 0.5
    assert mrr_at_k({9}, [1, 2, 3], 2) == 0.0


def test_empty_relevant_scores_zero():
    assert hit_at_k(set(), [1, 2], 3) == 0.0
    assert recall_at_k(set(), [1, 2], 3) == 0.0


def test_score_item_and_aggregate_keys():
    row = score_item({7}, [7, 1, 2])
    assert row["hit@1"] == 1.0 and row["mrr@3"] == 1.0
    assert set(row) == {f"{m}@{k}" for m in ("hit", "recall", "mrr") for k in RETRIEVAL_KS}
    agg = aggregate([score_item({7}, [7]), score_item({7}, [1, 7])])
    assert agg["hit@1"] == 0.5
