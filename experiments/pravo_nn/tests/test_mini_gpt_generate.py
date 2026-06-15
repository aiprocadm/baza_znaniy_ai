import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig
from experiments.pravo_nn.mini_gpt.model import GPT
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer
from experiments.pravo_nn.mini_gpt.train import save_checkpoint
from experiments.pravo_nn.mini_gpt.generate import load_checkpoint, generate_text

SAMPLE = "Статья 1. Основные начала.\nСтатья 2. Регулируемые отношения.\n" * 10


def _make_ckpt(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    tok_dir = tmp_path / "tokenizer"
    tok.save(tok_dir)
    cfg = GPTConfig(
        vocab_size=len(tok.vocab), block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0
    )
    model = GPT(cfg)
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, cfg, tokenizer_dir=str(tok_dir), step=1, val_loss=9.9, path=ckpt_path)
    return ckpt_path, model, tok


def test_load_checkpoint_restores_identical_weights(tmp_path):
    ckpt_path, model, _ = _make_ckpt(tmp_path)
    loaded, meta = load_checkpoint(ckpt_path, device="cpu")
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), loaded.state_dict().items()
    ):
        assert k1 == k2 and torch.equal(v1, v2)
    assert meta["step"] == 1 and meta["val_loss"] == 9.9


def test_generate_text_returns_string_starting_with_prompt(tmp_path):
    ckpt_path, _, _ = _make_ckpt(tmp_path)
    out = generate_text(ckpt_path, prompt="Статья", max_new_tokens=10, device="cpu")
    assert isinstance(out, str)
    assert out.startswith("Статья")
    assert len(out) > len("Статья")  # something was generated
