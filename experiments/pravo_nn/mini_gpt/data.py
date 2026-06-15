"""Corpus <-> on-disk token array. `encode_corpus` tokenizes once into a uint16
.bin (vocab < 65536 fits uint16); `load_bin` memmaps it; `get_batch` draws random
contiguous windows. `y` is `x` shifted by one token (next-token prediction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer


def encode_corpus(text: str, tokenizer: BPETokenizer, out_path) -> int:
    ids = tokenizer.encode(text)
    arr = np.array(ids, dtype=np.uint16)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out)
    return len(ids)


def load_bin(path) -> np.memmap:
    return np.memmap(Path(path), dtype=np.uint16, mode="r")


def get_batch(
    data,
    *,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
):
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
