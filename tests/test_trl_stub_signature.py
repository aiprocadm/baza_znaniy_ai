"""Contract check: stub trl exposes the surface train_dpo.py uses."""

from __future__ import annotations


def test_stub_exposes_dpoconfig_with_expected_fields() -> None:
    import trl

    cfg = trl.DPOConfig(output_dir="x")
    for field_name in (
        "output_dir",
        "beta",
        "learning_rate",
        "per_device_train_batch_size",
        "num_train_epochs",
        "max_length",
        "max_prompt_length",
        "logging_steps",
        "save_steps",
    ):
        assert hasattr(cfg, field_name), f"DPOConfig missing field {field_name!r}"


def test_stub_dpotrainer_records_train_call() -> None:
    import trl

    trl.DPOTrainer.train_calls.clear()
    trainer = trl.DPOTrainer(
        model=object(),
        args=trl.DPOConfig(output_dir="x", beta=0.1),
        train_dataset=[1, 2, 3],
    )
    trainer.train()
    trainer.save_model("./var/test-dpo-stub-out")
    assert len(trl.DPOTrainer.train_calls) == 1
    assert trl.DPOTrainer.train_calls[0]["beta"] == 0.1
    assert trl.DPOTrainer.train_calls[0]["dataset_size"] == 3
