"""Regression tests for the train_lora adapter-artifact finalisation.

These guard the lora-smoke CI workflow against a recurrence of the
``HFValidationError`` failure: ``PeftModel.from_pretrained(adapter_dir)``
expects to find ``adapter_model.safetensors`` in the directory, while
the KB.AI runtime expects ``adapter.safetensors``. Both must coexist
after a successful train run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts._lora_artifacts import finalise_adapter_artifacts


def test_finalise_keeps_both_filenames_when_peft_standard_present(tmp_path: Path) -> None:
    """The PEFT standard file is preserved and copied to the KB.AI name."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    peft_standard = adapter_dir / "adapter_model.safetensors"
    peft_standard.write_bytes(b"\x00\x01\x02 fake-weights")

    finalise_adapter_artifacts(adapter_dir)

    assert peft_standard.is_file(), "PEFT standard filename must be retained"
    assert (adapter_dir / "adapter.safetensors").is_file(), "KB.AI runtime filename must exist"
    assert (
        peft_standard.read_bytes() == (adapter_dir / "adapter.safetensors").read_bytes()
    ), "both files must have identical contents"


def test_finalise_promotes_unusual_safetensors_to_both_names(tmp_path: Path) -> None:
    """A non-standard *.safetensors filename is mirrored under both canonical names."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    unusual = adapter_dir / "fancy_adapter.safetensors"
    unusual.write_bytes(b"weights")

    finalise_adapter_artifacts(adapter_dir)

    assert (adapter_dir / "adapter.safetensors").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()


def test_finalise_raises_when_no_safetensors_produced(tmp_path: Path) -> None:
    """If save_pretrained produced no .safetensors file, fail loudly."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No safetensors adapter produced"):
        finalise_adapter_artifacts(adapter_dir)
