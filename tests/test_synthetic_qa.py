"""Tests for app.services.synthetic_qa — pure-logic Q&A generator."""

from __future__ import annotations

import pytest


def test_module_imports():
    """Module imports without side effects."""
    from app.services import synthetic_qa

    assert hasattr(synthetic_qa, "__name__")
    assert synthetic_qa.__name__ == "app.services.synthetic_qa"


def test_qa_pair_to_dict_uses_canonical_fields():
    from app.services.synthetic_qa import QAPair

    pair = QAPair(
        instruction="What is X?",
        input="Context paragraph.",
        output="X is the answer.",
        source_chunk_id=42,
    )

    data = pair.to_dict()
    assert data == {
        "instruction": "What is X?",
        "input": "Context paragraph.",
        "output": "X is the answer.",
        "meta": {"source_chunk_id": 42},
    }


def test_qa_pair_to_jsonl_line_is_single_line():
    from app.services.synthetic_qa import QAPair

    pair = QAPair(
        instruction="Q?",
        input="",
        output="A.",
        source_chunk_id=1,
    )

    line = pair.to_jsonl_line()

    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert "\\n" not in line  # No literal escaped newlines in output values


def test_qa_pair_from_jsonl_line_round_trip():
    from app.services.synthetic_qa import QAPair

    original = QAPair(
        instruction="Q with «русские» symbols?",
        input="ctx",
        output="A.",
        source_chunk_id=7,
    )

    parsed = QAPair.from_jsonl_line(original.to_jsonl_line())

    assert parsed == original


def test_length_filter_accepts_in_range_pair():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="What does the regulation say about Y?",
        input="",
        output="The regulation states that Y must be done following X procedure.",
        source_chunk_id=1,
    )

    assert length_ok(pair) is True


def test_length_filter_rejects_short_instruction():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="Why?",  # 4 chars < 10
        input="",
        output="A long enough answer goes here to pass that threshold.",
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_length_filter_rejects_long_instruction():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="x" * 201,  # 201 chars > 200
        input="",
        output="A long enough answer goes here to pass that threshold.",
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_length_filter_rejects_short_output():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="What is the rule about Z?",
        input="",
        output="Short.",  # 6 chars < 30
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_length_filter_rejects_long_output():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="What is the rule about Z?",
        input="",
        output="x" * 2001,  # 2001 chars > 2000
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_refusal_filter_accepts_normal_answer():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("The procedure requires two signatures.") is False


def test_refusal_filter_detects_english_refusal():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("I cannot answer this question.") is True
    assert is_refusal("Sorry, I can't help with that.") is True
    assert is_refusal("As an AI language model, I cannot...") is True


def test_refusal_filter_detects_russian_refusal():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("Извините, я не могу ответить на этот вопрос.") is True
    assert is_refusal("Я не имею возможности ответить.") is True
    assert is_refusal("Как языковая модель, я не могу") is True


def test_refusal_filter_is_case_insensitive():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("I CANNOT answer") is True
    assert is_refusal("извините, я Не Могу") is True


def test_refusal_filter_handles_empty_string():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("") is False
    assert is_refusal("   ") is False


def test_self_consistency_accepts_identical_text():
    from app.services.synthetic_qa import self_consistent

    text = "The annual leave is 28 calendar days per year."
    assert self_consistent(text, text) is True


def test_self_consistency_accepts_paraphrase():
    from app.services.synthetic_qa import self_consistent

    a = "Annual leave is twenty-eight calendar days each year for every employee."
    b = "Each employee is entitled to twenty-eight calendar days of annual leave per year."
    assert self_consistent(a, b) is True


def test_self_consistency_rejects_unrelated_text():
    from app.services.synthetic_qa import self_consistent

    a = "Annual leave is 28 calendar days per year."
    b = "The kitchen ventilation system needs monthly inspection."
    assert self_consistent(a, b) is False


def test_self_consistency_handles_empty_text():
    from app.services.synthetic_qa import self_consistent

    assert self_consistent("", "") is False
    assert self_consistent("Some text.", "") is False


def test_self_consistency_threshold_can_be_overridden():
    from app.services.synthetic_qa import self_consistent

    a = "alpha beta gamma delta"
    b = "alpha beta zeta theta"  # 2/6 unique tokens shared = 0.33 Jaccard

    assert self_consistent(a, b, threshold=0.5) is False
    assert self_consistent(a, b, threshold=0.3) is True
