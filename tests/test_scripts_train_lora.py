import argparse
import types
from pathlib import Path

import pytest

from scripts import train_lora


class DummyArgs(argparse.Namespace):
    pass


def build_args(tmp_path: Path) -> DummyArgs:
    return DummyArgs(
        dataset=tmp_path / "data.csv",
        base_model="distil",
        output_dir=tmp_path / "out",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=128,
    )


def test_validate_args_missing_dataset(tmp_path):
    args = build_args(tmp_path)
    with pytest.raises(FileNotFoundError):
        train_lora.validate_args(args)


def test_validate_args_wrong_suffix(tmp_path):
    args = build_args(tmp_path)
    args.dataset = tmp_path / "data.txt"
    args.dataset.write_text("dummy", encoding="utf-8")
    with pytest.raises(ValueError):
        train_lora.validate_args(args)


def test_convert_adapter_to_ggml_with_module(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    output_dir = tmp_path / "ggml"
    called = {}

    module = types.SimpleNamespace()

    def fake_convert(**kwargs):
        called.update(kwargs)

    module.convert_lora_to_ggml = fake_convert
    monkeypatch.setattr(train_lora, "load_llama_cpp", lambda: module)

    result_path = train_lora.convert_adapter_to_ggml(adapter_dir, output_dir, "llama-base", filename="out.ggml")

    assert result_path == output_dir / "out.ggml"
    assert called["base_model"] == "llama-base"
    assert called["adapter_path"] == str(adapter_dir)
    assert called["output_path"] == str(result_path)


def test_convert_adapter_to_ggml_cli_fallback(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    output_dir = tmp_path / "ggml"

    def fake_load():
        raise ModuleNotFoundError("llama_cpp not installed")

    monkeypatch.setattr(train_lora, "load_llama_cpp", fake_load)

    recorded = {}

    def fake_run(cmd, check):
        recorded["cmd"] = cmd
        recorded["check"] = check

    monkeypatch.setattr(train_lora.subprocess, "run", fake_run)

    result_path = train_lora.convert_adapter_to_ggml(adapter_dir, output_dir, "llama-base")

    assert result_path == output_dir / "adapter.ggml"
    assert recorded["check"] is True
    assert "llama_cpp.convert_lora" in recorded["cmd"]
