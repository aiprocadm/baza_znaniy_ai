"""Tests for app.services.dpo_dataset — pure-logic DPO dataset builder."""

from __future__ import annotations


def test_module_imports() -> None:
    """Module imports without side effects."""
    from app.services import dpo_dataset

    assert dpo_dataset.__name__ == "app.services.dpo_dataset"
