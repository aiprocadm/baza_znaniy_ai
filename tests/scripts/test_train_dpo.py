"""Stub-backed tests for scripts.train_dpo."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_module_imports() -> None:
    import scripts.train_dpo as cli

    assert callable(cli.parse_args)
    assert callable(cli.main)


def test_parse_args_required_flags() -> None:
    from scripts.train_dpo import parse_args

    ns = parse_args(
        [
            "--base-model",
            "stub-base",
            "--train",
            "dpo.jsonl",
            "--sft-adapter",
            "adapters/sft",
            "--output",
            "adapters/dpo",
            "--prompt-mode",
            "rag",
            "--max-steps",
            "1",
        ]
    )
    assert ns.base_model == "stub-base"
    assert ns.train == Path("dpo.jsonl")
    assert ns.sft_adapter == Path("adapters/sft")
    assert ns.output == Path("adapters/dpo")
    assert ns.prompt_mode == "rag"
    assert ns.max_steps == 1


def test_train_dpo_smoke_against_stub(tmp_path) -> None:
    """End-to-end: write 2 DPO pairs, run main(), assert adapter saved."""
    import json

    import trl
    from scripts.train_dpo import main

    train_path = tmp_path / "dpo.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for i in range(2):
            fh.write(
                json.dumps(
                    {
                        "prompt": f"Q{i}?",
                        "chosen": f"A{i}. [doc_chunk:{i}]",
                        "rejected": f"A{i}.",
                        "meta": {"strategy": "no_citation"},
                    }
                )
                + "\n"
            )

    output = tmp_path / "adapters" / "dpo"
    trl.DPOTrainer.train_calls.clear()

    rc = main(
        [
            "--base-model",
            "stub-base",
            "--train",
            str(train_path),
            "--sft-adapter",
            str(tmp_path / "fake-sft"),
            "--output",
            str(output),
            "--prompt-mode",
            "rag",
            "--max-steps",
            "1",
        ]
    )
    assert rc == 0
    assert (output / "adapter_config.json").exists()
    assert len(trl.DPOTrainer.train_calls) == 1


def test_train_dpo_raises_systemexit_on_missing_dataset() -> None:
    from scripts.train_dpo import main

    with pytest.raises(SystemExit):
        main(
            [
                "--base-model",
                "stub-base",
                "--train",
                "does-not-exist.jsonl",
                "--sft-adapter",
                "fake-sft",
                "--output",
                "adapters/x",
            ]
        )
