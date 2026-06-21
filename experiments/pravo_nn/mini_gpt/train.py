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


def save_checkpoint(model, cfg: GPTConfig, *, tokenizer_dir: str, step: int, val_loss: float, path, optimizer=None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "tokenizer": tokenizer_dir,
        "step": step,
        "val_loss": val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, out)


def _lr_at(step: int, *, base_lr: float, warmup: int, total: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def estimate_loss(model, data, *, block_size: int, batch_size: int, device: str, eval_iters: int = 20) -> float:
    model.train(False)  # inference mode (disables dropout) for a clean val measurement
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size=block_size, batch_size=batch_size, device=device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train(True)  # back to training mode
    return sum(losses) / len(losses)


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
    eval_interval: int = 500,
    eval_iters: int = 20,
    resume_from: Path | None = None,
) -> Path:
    device = get_device()
    tok = BPETokenizer.load(data_dir / "tokenizer")
    vocab_size = len(tok.vocab) + len(tok.special_tokens)
    ckpt_path = data_dir / "checkpoints" / "ckpt.pt"

    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if ckpt["config"]["vocab_size"] != vocab_size:
            raise ValueError(
                f"tokenizer vocab {vocab_size} != checkpoint vocab {ckpt['config']['vocab_size']}; "
                "warm-start needs the SAME tokenizer (reuse data/tokenizer, do not retrain it)"
            )
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        LOGGER.info("resumed from %s at step %d", resume_from, start_step)
    else:
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=preset.block_size,
            n_layer=preset.n_layer,
            n_head=preset.n_head,
            n_embd=preset.n_embd,
            dropout=preset.dropout,
        )
        model = GPT(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        "device=%s params=%.2fM block=%d vocab=%d start_step=%d",
        device, n_params / 1e6, cfg.block_size, cfg.vocab_size, start_step,
    )

    data = load_bin(data_dir / "train.bin")
    val_path = data_dir / "val.bin"
    val_data = load_bin(val_path) if val_path.exists() else None

    last_loss = float("inf")
    last_val = float("inf")
    for local in range(max_steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(local, base_lr=base_lr, warmup=warmup, total=max_steps, min_lr=base_lr / 10)
        x, y = get_batch(data, block_size=cfg.block_size, batch_size=batch_size, device=device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        abs_step = start_step + local
        if local % log_interval == 0:
            LOGGER.info("step %d (local %d/%d) loss %.4f", abs_step, local, max_steps, last_loss)
        if val_data is not None and local > 0 and local % eval_interval == 0:
            last_val = estimate_loss(model, val_data, block_size=cfg.block_size, batch_size=batch_size, device=device, eval_iters=eval_iters)
            LOGGER.info("step %d val_loss %.4f", abs_step, last_val)
        if local > 0 and local % ckpt_interval == 0:
            save_checkpoint(
                model, cfg, tokenizer_dir="data/tokenizer", step=abs_step,
                val_loss=(last_val if val_data is not None else last_loss),
                path=ckpt_path, optimizer=opt,
            )

    final_val = (
        estimate_loss(model, val_data, block_size=cfg.block_size, batch_size=batch_size, device=device, eval_iters=eval_iters)
        if val_data is not None else last_loss
    )
    save_checkpoint(
        model, cfg, tokenizer_dir="data/tokenizer", step=start_step + max_steps,
        val_loss=final_val, path=ckpt_path, optimizer=opt,
    )
    LOGGER.info("done; final train-loss %.4f val-loss %.4f -> %s", last_loss, final_val, ckpt_path)
    return ckpt_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="mini_gpt.train")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-interval", type=int, default=500)
    default_ckpt = str(_DATA / "checkpoints" / "ckpt.pt")
    p.add_argument("--resume", nargs="?", const=default_ckpt, default=None,
                   help="resume (warm-start) from a checkpoint; bare flag uses the default ckpt path")
    args = p.parse_args(argv)
    train(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        resume_from=Path(args.resume) if args.resume else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
