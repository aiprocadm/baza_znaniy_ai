import pytest
import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig
from experiments.pravo_nn.mini_gpt.data import encode_corpus_split
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer
from experiments.pravo_nn.mini_gpt.train import train

SAMPLE = "Статья 1. Основные начала регулирования отношений в обществе.\n" * 80
# Tiny model so the whole resume cycle runs in well under a second.
TINY = GPTConfig(vocab_size=0, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)


def _setup(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    tok.save(tmp_path / "tokenizer")
    encode_corpus_split(
        SAMPLE, tok,
        train_path=tmp_path / "train.bin",
        val_path=tmp_path / "val.bin",
        val_frac=0.1,
    )


def _ckpt(tmp_path):
    return tmp_path / "checkpoints" / "ckpt.pt"


def test_checkpoint_carries_optimizer_and_real_val_loss(tmp_path):
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=3, batch_size=2,
          warmup=1, log_interval=1, ckpt_interval=2, eval_interval=2, eval_iters=2)
    ckpt = torch.load(_ckpt(tmp_path), map_location="cpu", weights_only=False)
    assert ckpt["step"] == 3
    assert "optimizer_state_dict" in ckpt
    assert isinstance(ckpt["val_loss"], float)


def test_resume_continues_step_counter(tmp_path):
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=3, batch_size=2, warmup=1)
    train(preset=TINY, data_dir=tmp_path, max_steps=2, batch_size=2, warmup=1,
          resume_from=_ckpt(tmp_path))
    ckpt = torch.load(_ckpt(tmp_path), map_location="cpu", weights_only=False)
    assert ckpt["step"] == 5  # 3 + 2, not reset to 2


def test_resume_works_without_optimizer_state(tmp_path):
    """Backward compat: the original #1 ckpt_v1 has no optimizer_state_dict."""
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=2, batch_size=2, warmup=1)
    p = _ckpt(tmp_path)
    c = torch.load(p, map_location="cpu", weights_only=False)
    c.pop("optimizer_state_dict")
    torch.save(c, p)
    train(preset=TINY, data_dir=tmp_path, max_steps=1, batch_size=2, warmup=1, resume_from=p)  # must not raise


def test_resume_rejects_vocab_mismatch(tmp_path):
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=2, batch_size=2, warmup=1)
    # retrain the tokenizer to a different vocab, overwriting the dir
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=320)
    tok.save(tmp_path / "tokenizer")
    with pytest.raises(ValueError):
        train(preset=TINY, data_dir=tmp_path, max_steps=1, batch_size=2, warmup=1,
              resume_from=_ckpt(tmp_path))
