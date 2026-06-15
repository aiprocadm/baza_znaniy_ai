"""Training loop for the mini-GPT + the checkpoint writer.

`save_checkpoint` writes the spec's #1<->#2 contract: model weights, the exact
config (so #2 rebuilds the architecture), the tokenizer directory, step, and
val_loss. Device is auto-detected and logged; checkpoints are written
periodically so an interrupted overnight run still leaves a usable base."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig, CPU_OVERNIGHT
from experiments.pravo_nn.mini_gpt.data import get_batch, load_bin
from experiments.pravo_nn.mini_gpt.model import GPT
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

LOGGER = logging.getLogger(__name__)
_DATA = Path(__file__).resolve().parent.parent / "data"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(model, cfg: GPTConfig, *, tokenizer_dir: str, step: int, val_loss: float, path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "tokenizer": tokenizer_dir,
            "step": step,
            "val_loss": val_loss,
        },
        out,
    )


def _lr_at(step: int, *, base_lr: float, warmup: int, total: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def train(
    *,
    preset: GPTConfig = CPU_OVERNIGHT,
    data_dir: Path = _DATA,
    max_steps: int = 5000,
    batch_size: int = 32,
    base_lr: float = 3e-4,
    warmup: int = 100,
    log_interval: int = 250,
    ckpt_interval: int = 500,
) -> Path:
    device = get_device()
    tok = BPETokenizer.load(data_dir / "tokenizer")
    cfg = GPTConfig(
        vocab_size=len(tok.vocab) + len(tok.special_tokens),
        block_size=preset.block_size,
        n_layer=preset.n_layer,
        n_head=preset.n_head,
        n_embd=preset.n_embd,
        dropout=preset.dropout,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info("device=%s params=%.2fM block=%d vocab=%d", device, n_params / 1e6, cfg.block_size, cfg.vocab_size)

    data = load_bin(data_dir / "train.bin")
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    ckpt_path = data_dir / "checkpoints" / "ckpt.pt"
    last_loss = float("inf")
    for step in range(max_steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, base_lr=base_lr, warmup=warmup, total=max_steps, min_lr=base_lr / 10)
        x, y = get_batch(data, block_size=cfg.block_size, batch_size=batch_size, device=device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if step % log_interval == 0:
            LOGGER.info("step %d/%d loss %.4f", step, max_steps, last_loss)
        if step > 0 and step % ckpt_interval == 0:
            save_checkpoint(model, cfg, tokenizer_dir="data/tokenizer", step=step, val_loss=last_loss, path=ckpt_path)
    save_checkpoint(model, cfg, tokenizer_dir="data/tokenizer", step=max_steps, val_loss=last_loss, path=ckpt_path)
    LOGGER.info("done; final loss %.4f -> %s", last_loss, ckpt_path)
    return ckpt_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="mini_gpt.train")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)
    train(max_steps=args.max_steps, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
