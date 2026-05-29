"""Tests for app.services.dpo_dataset — pure-logic DPO dataset builder."""

from __future__ import annotations


def test_module_imports() -> None:
    """Module imports without side effects."""
    from app.services import dpo_dataset

    assert dpo_dataset.__name__ == "app.services.dpo_dataset"


def test_reject_strategy_values() -> None:
    """The three canonical synthetic reject strategies are exposed."""
    from app.services.dpo_dataset import RejectStrategy

    values = {s.value for s in RejectStrategy}
    assert {"no_citation", "generic", "hallucination"}.issubset(values)


def test_dpo_pair_to_jsonl_line_top_level_keys() -> None:
    """to_jsonl_line() emits prompt / chosen / rejected at the top level."""
    import json

    from app.services.dpo_dataset import DPOPair, RejectStrategy

    pair = DPOPair(
        prompt="Что такое отпуск?",
        chosen="Это перерыв. [doc_chunk:7]",
        rejected="Это перерыв.",
        strategy=RejectStrategy.NO_CITATION,
        source="synthetic",
        source_chunk_id=7,
        feedback_ids=(),
    )
    line = pair.to_jsonl_line()
    assert line.endswith("\n")

    data = json.loads(line)
    assert data["prompt"] == "Что такое отпуск?"
    assert data["chosen"].endswith("[doc_chunk:7]")
    assert data["rejected"] == "Это перерыв."
    assert data["meta"]["strategy"] == "no_citation"
    assert data["meta"]["source"] == "synthetic"
    assert data["meta"]["source_chunk_id"] == 7
    assert data["meta"]["feedback_ids"] == []
