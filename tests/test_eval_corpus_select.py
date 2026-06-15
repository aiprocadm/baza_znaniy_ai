"""Unit tests for the public/private eval-corpus selector (pure, no I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.eval.corpus_select import (
    PUBLIC_GOLDEN,
    CorpusSelectionError,
    resolve_corpus,
)


def test_defaults_to_public_when_unset() -> None:
    paths = resolve_corpus({})
    assert paths.label == "public"
    assert paths.golden_path == PUBLIC_GOLDEN


def test_public_is_case_insensitive_and_trimmed() -> None:
    assert resolve_corpus({"KB_EVAL_CORPUS": "  PUBLIC "}).label == "public"


def test_private_resolves_to_env_paths() -> None:
    env = {
        "KB_EVAL_CORPUS": "private",
        "KB_EVAL_PRIVATE_DB": "var/data/kb_private.sqlite",
        "KB_EVAL_PRIVATE_GOLDEN": "var/data/eval/private/golden.jsonl",
    }
    paths = resolve_corpus(env)
    assert paths.label == "private"
    assert paths.db_path == Path("var/data/kb_private.sqlite")
    assert paths.golden_path == Path("var/data/eval/private/golden.jsonl")


def test_private_without_env_fails_loudly_naming_missing_vars() -> None:
    with pytest.raises(CorpusSelectionError) as exc:
        resolve_corpus({"KB_EVAL_CORPUS": "private"})
    msg = str(exc.value)
    assert "KB_EVAL_PRIVATE_DB" in msg and "KB_EVAL_PRIVATE_GOLDEN" in msg


def test_unknown_choice_is_rejected() -> None:
    with pytest.raises(CorpusSelectionError):
        resolve_corpus({"KB_EVAL_CORPUS": "staging"})
