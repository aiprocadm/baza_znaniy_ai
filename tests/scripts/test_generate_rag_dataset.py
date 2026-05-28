"""Smoke tests for scripts.generate_rag_dataset CLI."""

from __future__ import annotations


def test_cli_module_imports() -> None:
    import scripts.generate_rag_dataset as cli

    assert callable(cli.parse_args)
    assert callable(cli.main)


def test_parse_args_minimal() -> None:
    from pathlib import Path

    from scripts.generate_rag_dataset import parse_args

    ns = parse_args(
        [
            "--corpus",
            "var/data/kb.sqlite",
            "--seeds",
            "var/data/seeds.jsonl",
            "--output",
            "var/data/rag.jsonl",
            "--target-pairs",
            "100",
        ]
    )
    assert ns.corpus == Path("var/data/kb.sqlite")
    assert ns.seeds == Path("var/data/seeds.jsonl")
    assert ns.output == Path("var/data/rag.jsonl")
    assert ns.target_pairs == 100
    assert ns.top_k == 3  # default
