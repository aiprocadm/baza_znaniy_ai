"""Smoke tests for scripts.generate_dpo_pairs CLI."""

from __future__ import annotations


def test_cli_module_imports() -> None:
    import scripts.generate_dpo_pairs as cli

    assert callable(cli.parse_args)
    assert callable(cli.main)


def test_parse_args_minimal() -> None:
    from pathlib import Path

    from scripts.generate_dpo_pairs import parse_args

    ns = parse_args(
        [
            "--seeds",
            "var/data/seeds.jsonl",
            "--output",
            "var/data/dpo.jsonl",
            "--target-pairs",
            "100",
            "--yes",
        ]
    )
    assert ns.seeds == Path("var/data/seeds.jsonl")
    assert ns.output == Path("var/data/dpo.jsonl")
    assert ns.target_pairs == 100
    assert ns.yes is True
    assert ns.max_cost_usd == 1.0
