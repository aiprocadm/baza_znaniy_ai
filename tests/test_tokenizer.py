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


def test_truncate_tokens_returns_new_list_for_tuples():
    tokens = ("a", "b")
    truncated = truncate_tokens(tokens, limit=5)

    assert isinstance(truncated, list)
    assert truncated == ["a", "b"]

    truncated.append("c")
    assert tokens == ("a", "b")


def test_truncate_tokens_and_count_tokens_whitespace_behavior():
    assert truncate_tokens((), limit=5) == []
    assert count_tokens("\n\n") == 2


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
