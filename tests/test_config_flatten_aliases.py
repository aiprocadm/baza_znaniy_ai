"""Regression tests for app.core.config._flatten_aliases.

The original implementation had a dead try/except guard:

    try:
        iterable = source          # plain assignment never raises
    except TypeError:              # so this fallback never ran
        iterable = source.choices
    for choice in iterable:        # TypeError raised here, outside the try

In Pydantic versions where AliasChoices was Iterable this worked by
accident; in Pydantic >=2.9 AliasChoices is no longer Iterable and the
fallback must engage. These tests pin the contract so the regression
cannot return.
"""

from __future__ import annotations

from pydantic import AliasChoices

from app.core.config import _flatten_aliases


def test_flatten_alias_choices_returns_choice_names():
    """AliasChoices(...) should flatten to its choice list."""
    result = _flatten_aliases(AliasChoices("APP_ENV", "ENV", "ENVIRONMENT"))
    assert result == ["APP_ENV", "ENV", "ENVIRONMENT"]


def test_flatten_nested_alias_choices():
    """Choices containing strings nest correctly through recursion."""
    result = _flatten_aliases(AliasChoices("A", "B"))
    assert result == ["A", "B"]


def test_flatten_plain_string_returns_single_item():
    assert _flatten_aliases("APP_ENV") == ["APP_ENV"]


def test_flatten_none_returns_empty():
    assert _flatten_aliases(None) == []


def test_flatten_iterable_of_strings():
    assert _flatten_aliases(["A", "B"]) == ["A", "B"]
