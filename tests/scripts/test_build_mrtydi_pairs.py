"""Pure-function tests for the mr-TyDi stage-1 builder (no ML deps)."""

from scripts.build_mrtydi_pairs import record_to_texts, to_pairs


def test_to_pairs_emits_positive_then_negatives_with_binary_scores():
    rows = to_pairs("какой срок", "три года", ["ставка налога", "состав суда"])
    assert rows == [
        {"query": "какой срок", "text": "три года", "teacher_score": 1.0},
        {"query": "какой срок", "text": "ставка налога", "teacher_score": 0.0},
        {"query": "какой срок", "text": "состав суда", "teacher_score": 0.0},
    ]


def test_to_pairs_skips_blank_query_or_positive():
    assert to_pairs("", "pos", ["neg"]) == []
    assert to_pairs("q", "", ["neg"]) == []


def test_to_pairs_drops_blank_negatives_individually():
    rows = to_pairs("q", "pos", ["", "real neg", "   "])
    assert rows == [
        {"query": "q", "text": "pos", "teacher_score": 1.0},
        {"query": "q", "text": "real neg", "teacher_score": 0.0},
    ]


def test_record_to_texts_extracts_first_positive_and_caps_negatives():
    record = {
        "query": "вопрос",
        "positive_passages": [{"docid": "a", "text": "позитив", "title": "t"}],
        "negative_passages": [
            {"docid": "b", "text": "neg1", "title": "t"},
            {"docid": "c", "text": "neg2", "title": "t"},
            {"docid": "d", "text": "neg3", "title": "t"},
        ],
    }
    assert record_to_texts(record, max_negs=2) == ("вопрос", "позитив", ["neg1", "neg2"])


def test_record_to_texts_handles_missing_positive():
    record = {"query": "q", "positive_passages": [], "negative_passages": []}
    assert record_to_texts(record, max_negs=5) == ("q", "", [])
