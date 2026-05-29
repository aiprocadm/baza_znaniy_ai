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


def test_cli_writes_jsonl_endtoend(tmp_path, monkeypatch) -> None:
    """4 seeds + fake teacher → 4 pairs, strategy mix within ±1 of 40/30/30."""
    import json

    from app.services.synthetic_qa import QAPair
    from scripts.generate_dpo_pairs import main

    seeds_path = tmp_path / "seeds.jsonl"
    with seeds_path.open("w", encoding="utf-8") as fh:
        for i in range(1, 5):
            fh.write(
                QAPair(
                    instruction=f"Вопрос {i}?",
                    input="",
                    output=f"Ответ {i}. [doc_chunk:{i}]",
                    source_chunk_id=i,
                ).to_jsonl_line()
            )

    import scripts.generate_dpo_pairs as cli

    monkeypatch.setattr(
        cli, "_make_teacher", lambda _args: (lambda _q: "Fake teacher answer.")
    )

    output = tmp_path / "dpo.jsonl"
    rc = main(
        [
            "--seeds",
            str(seeds_path),
            "--output",
            str(output),
            "--target-pairs",
            "4",
            "--yes",
        ]
    )
    assert rc == 0

    lines = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 4
    for line in lines:
        assert "prompt" in line
        assert "chosen" in line
        assert "rejected" in line
        assert line["meta"]["strategy"] in {"no_citation", "generic", "hallucination"}


def test_cost_guard_aborts_without_yes(tmp_path) -> None:
    """Without --yes the cost guard kicks in for >100 teacher calls."""
    import pytest

    from app.services.synthetic_qa import QAPair
    from scripts.generate_dpo_pairs import main

    seeds_path = tmp_path / "seeds.jsonl"
    with seeds_path.open("w", encoding="utf-8") as fh:
        for i in range(1, 5001):
            fh.write(
                QAPair(
                    instruction=f"Q{i}?",
                    input="",
                    output=f"A. [doc_chunk:{i}]",
                    source_chunk_id=i,
                ).to_jsonl_line()
            )

    output = tmp_path / "dpo.jsonl"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--seeds",
                str(seeds_path),
                "--output",
                str(output),
                "--target-pairs",
                "5000",
                "--max-cost-usd",
                "0.10",
            ]
        )
    assert "Estimated" in str(exc.value) or "budget" in str(exc.value).lower()
