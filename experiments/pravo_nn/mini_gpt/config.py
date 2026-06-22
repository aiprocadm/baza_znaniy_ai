"""Hyperparameters for the mini-GPT. `GPTConfig` is the single source of model
shape; it is round-tripped through the checkpoint so sub-project #2 can rebuild
the exact architecture. `CPU_OVERNIGHT` is the ~10M-param preset sized for a
CPU/overnight budget; a GPU run just scales these numbers up."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 8000  # overwritten at train time to the tokenizer's real size
    block_size: int = 256  # context length (tokens)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


# ~10M parameters: fits an overnight CPU budget on the 12 MB corpus.
CPU_OVERNIGHT = GPTConfig(
    vocab_size=8000, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1
)
