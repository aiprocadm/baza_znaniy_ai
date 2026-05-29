"""@pytest.mark.integration: exercises real trl on a tiny model in CI."""

from __future__ import annotations

import json

import pytest


@pytest.mark.integration
def test_train_dpo_on_tiny_model(tmp_path) -> None:
    """Skipped unless real trl + a tiny HF model are available locally / in CI."""
    pytest.importorskip("trl")
    pytest.importorskip("transformers")

    from scripts.train_dpo import main

    train_path = tmp_path / "dpo.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for i in range(4):
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
    rc = main(
        [
            "--base-model",
            "sshleifer/tiny-gpt2",
            "--train",
            str(train_path),
            "--sft-adapter",
            str(tmp_path / "fake-sft"),
            "--output",
            str(output),
            "--prompt-mode",
            "generic",
            "--max-steps",
            "1",
            "--num-train-epochs",
            "1",
        ]
    )
    assert rc == 0
    assert (output / "adapter_config.json").exists() or any(output.iterdir())
