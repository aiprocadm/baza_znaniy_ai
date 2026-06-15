import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig
from experiments.pravo_nn.mini_gpt.model import GPT
from experiments.pravo_nn.mini_gpt.train import get_device, save_checkpoint


def test_get_device_returns_known_value():
    assert get_device() in {"cpu", "cuda"}


def test_save_checkpoint_writes_expected_contract(tmp_path):
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, cfg, tokenizer_dir="data/tokenizer", step=10, val_loss=1.5, path=path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ckpt) == {"model_state_dict", "config", "tokenizer", "step", "val_loss"}
    assert ckpt["config"] == {
        "vocab_size": 64, "block_size": 16, "n_layer": 2,
        "n_head": 2, "n_embd": 32, "dropout": 0.0,
    }
    assert ckpt["step"] == 10 and ckpt["val_loss"] == 1.5
    assert ckpt["tokenizer"] == "data/tokenizer"
