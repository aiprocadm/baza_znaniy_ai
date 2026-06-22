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
    # uint16 silently wraps ids >= 65536; guard so an oversized vocab fails loud
    # rather than corrupting the training bin invisibly.
    if ids and max(ids) > 65535:
        raise ValueError(f"token id {max(ids)} exceeds uint16; vocab too large for .bin")
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
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


def encode_corpus_split(
    text: str,
    tokenizer: BPETokenizer,
    *,
    train_path,
    val_path,
    val_frac: float = 0.05,
) -> tuple[int, int]:
    """Encode once, then reserve the LAST `val_frac` of tokens as a held-out
    val.bin. Returns (n_train, n_val)."""
    ids = tokenizer.encode(text)
    if ids and max(ids) > 65535:
        raise ValueError(f"token id {max(ids)} exceeds uint16; vocab too large for .bin")
    arr = np.array(ids, dtype=np.uint16)
    n_val = int(len(arr) * val_frac)
    split = len(arr) - n_val
    train_arr, val_arr = arr[:split], arr[split:]
    for out_path, chunk in ((train_path, train_arr), (val_path, val_arr)):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        chunk.tofile(out)
    return len(train_arr), len(val_arr)
