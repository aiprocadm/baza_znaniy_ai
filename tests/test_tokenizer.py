import pytest

from app.rag.tokenizer import count_tokens, detokenize, tokenize, truncate_tokens


def test_tokenize_splits_by_character():
    text = "abc😊"
    assert tokenize(text) == ["a", "b", "c", "😊"]


@pytest.mark.parametrize(
    "tokens, expected",
    [
        (["a", "b", "c"], "abc"),
        ([], ""),
        (iter(["x", "y"]), "xy"),
    ],
)
def test_detokenize_reverses_tokenize(tokens, expected):
    assert detokenize(tokens) == expected


@pytest.mark.parametrize(
    "tokens, limit, expected",
    [
        (["a", "b", "c"], 0, []),
        (["a", "b", "c"], -1, []),
        (["a", "b", "c"], 2, ["a", "b"]),
        (["a", "b"], 5, ["a", "b"]),
    ],
)
def test_truncate_tokens_limits_output(tokens, limit, expected):
    assert truncate_tokens(tokens, limit) == expected


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "你好",
        "",
    ],
)
def test_count_tokens_matches_tokenize_length(text):
    assert count_tokens(text) == len(tokenize(text))
