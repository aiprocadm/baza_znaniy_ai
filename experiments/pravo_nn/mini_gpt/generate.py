"""Checkpoint reader + text generation. `load_checkpoint` reconstructs the exact
architecture from the saved config and loads the weights (the #1<->#2 contract
reader). `generate_text` ties tokenizer + model into prompt -> sampled text."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig
from experiments.pravo_nn.mini_gpt.model import GPT
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

LOGGER = logging.getLogger(__name__)
_DATA = Path(__file__).resolve().parent.parent / "data"


def load_checkpoint(path, *, device: str = "cpu"):
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).train(False)  # inference mode (equivalent to .eval())
    return model, ckpt


def _resolve_tokenizer_dir(ckpt: dict, ckpt_path: Path) -> Path:
    raw = Path(ckpt["tokenizer"])
    if raw.is_absolute() and raw.exists():
        return raw
    # try the package data dir, then next to the checkpoint, then the raw path
    for cand in (_DATA / raw.name, ckpt_path.parent / raw.name, raw):
        if cand.exists():
            return cand
    return raw


def generate_text(
    ckpt_path,
    *,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
    device: str = "cpu",
) -> str:
    path = Path(ckpt_path)
    model, ckpt = load_checkpoint(path, device=device)
    tok = BPETokenizer.load(_resolve_tokenizer_dir(ckpt, path))
    ids = tok.encode(prompt) or [0]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="mini_gpt.generate")
    p.add_argument("--prompt", default="Статья 1.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args(argv)
    text = generate_text(
        _DATA / "checkpoints" / "ckpt.pt",
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
