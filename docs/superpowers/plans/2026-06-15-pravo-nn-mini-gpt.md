# Mini-GPT on the Legal Corpus — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a small decoder-only transformer **from scratch** on the 12 MB Russian legal corpus produced by sub-project #0, producing a documented checkpoint that sub-project #2 (LoRA) can load.

**Architecture:** New import-safe package `experiments/pravo_nn/mini_gpt/` (NOT the production `app/` tree), mirroring the `corpus_collector/` layout. Pipeline: `corpus.txt → tokenizer (own byte-level BPE) → train.bin (memmap token ids) → GPT training loop → ckpt.pt → generation`. Device-agnostic (cpu/cuda auto-detect); the default preset is sized for an overnight CPU run.

**Tech Stack:** Python 3.13 (`py -3.13`), PyTorch 2.12.0+cpu (already on user site-packages — no venv), numpy (torch dep), stdlib (`json`, `re`, `dataclasses`). Tokenizer + model implemented from scratch — no HuggingFace `transformers`/`tokenizers` dependency. pytest with heavy training mocked to a few steps.

**Reference spec:** [docs/superpowers/specs/2026-06-15-pravo-nn-mini-gpt-design.md](../specs/2026-06-15-pravo-nn-mini-gpt-design.md)

**Conventions (from CLAUDE.md / repo memory):**
- Run tests with `py -3.13 -m pytest experiments/pravo_nn/tests` (bare `py -3` resolves to a 3.14 with no pytest).
- Lives under `experiments/` so CI's `app/**` path-scoped gates are untouched and the anti-roadmap (own-LLM rejected as a *product*) is not violated — this is research.
- Conventional Commits; commit after every green step.

---

## File Structure

```
experiments/pravo_nn/
  mini_gpt/
    __init__.py
    config.py        # GPTConfig dataclass + CPU-overnight preset (~10M params)
    tokenizer.py     # byte-level BPE: train / encode / decode / save / load
    model.py         # GPT from scratch: GPTConfig -> nn.Module (+ generate)
    data.py          # encode corpus -> train.bin (memmap); random (x,y) batches
    train.py         # AdamW + cosine LR; device auto-detect; save_checkpoint
    generate.py      # load_checkpoint (reader) + generate_text (sampling)
  tests/             # (existing dir from #0 — add new test files here)
    test_mini_gpt_config.py
    test_mini_gpt_tokenizer.py
    test_mini_gpt_data.py
    test_mini_gpt_model.py
    test_mini_gpt_checkpoint.py
    test_mini_gpt_generate.py
  data/              # tokenizer/, train.bin, checkpoints/ all gitignored
```

Responsibilities: `config` is the single source of hyperparameters. `tokenizer` maps bytes ⇄ token ids and knows nothing about the model. `model` maps ids → logits and knows nothing about files. `data` turns corpus text into an on-disk token array and batches. `train` owns the optimization loop and the checkpoint *writer*. `generate` owns the checkpoint *reader* and sampling. The two cross-module contracts are the `Tokenizer` (encode/decode/save/load), the `GPTConfig`/`GPT` pair, and the `ckpt.pt` dict.

---

## Task 1: Package scaffold + gitignore for model artifacts

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/__init__.py` (empty)
- Modify: `experiments/pravo_nn/.gitignore`

- [ ] **Step 1: Create the package marker**

Create `experiments/pravo_nn/mini_gpt/__init__.py` as an empty file.

- [ ] **Step 2: Add model artifacts to `.gitignore`**

Append to `experiments/pravo_nn/.gitignore`:

```gitignore
# mini-GPT generated artifacts (tokenizer, token bins, checkpoints).
data/tokenizer/
data/train.bin
data/val.bin
data/checkpoints/
```

- [ ] **Step 3: Verify torch + numpy import on the ML interpreter**

Run: `py -3.13 -c "import torch, numpy; print(torch.__version__, numpy.__version__)"`
Expected: prints `2.12.0+cpu <numpy-version>`, exit 0.

- [ ] **Step 4: Verify the package imports**

Run: `py -3.13 -c "import experiments.pravo_nn.mini_gpt"`
Expected: no output, exit 0.

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/__init__.py experiments/pravo_nn/.gitignore
git commit -m "chore(pravo-nn): scaffold mini_gpt package + ignore model artifacts"
```

---

## Task 2: `config.py` — GPTConfig + CPU-overnight preset

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/config.py`
- Test: `experiments/pravo_nn/tests/test_mini_gpt_config.py`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import replace

from experiments.pravo_nn.mini_gpt.config import GPTConfig, CPU_OVERNIGHT


def test_default_config_is_consistent():
    cfg = GPTConfig()
    # n_embd must be divisible by n_head (heads split the embedding evenly)
    assert cfg.n_embd % cfg.n_head == 0
    assert cfg.block_size > 0
    assert cfg.vocab_size > 256  # at least the 256 byte base + merges


def test_preset_is_a_gptconfig_and_overridable():
    assert isinstance(CPU_OVERNIGHT, GPTConfig)
    assert CPU_OVERNIGHT.n_embd % CPU_OVERNIGHT.n_head == 0
    smaller = replace(CPU_OVERNIGHT, n_layer=2)
    assert smaller.n_layer == 2
    assert smaller.n_embd == CPU_OVERNIGHT.n_embd  # other fields preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_config.py -v`
Expected: FAIL — `ModuleNotFoundError` / `cannot import name 'GPTConfig'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Hyperparameters for the mini-GPT. `GPTConfig` is the single source of model
shape; it is round-tripped through the checkpoint so sub-project #2 can rebuild
the exact architecture. `CPU_OVERNIGHT` is the ~10M-param preset sized for a
CPU/overnight budget; a GPU run just scales these numbers up."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 8000  # overwritten at train time to the tokenizer's real size
    block_size: int = 256   # context length (tokens)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


# ~10M parameters: fits an overnight CPU budget on the 12 MB corpus.
CPU_OVERNIGHT = GPTConfig(
    vocab_size=8000, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/config.py experiments/pravo_nn/tests/test_mini_gpt_config.py
git commit -m "feat(pravo-nn): GPTConfig + CPU-overnight preset"
```

---

## Task 3: `tokenizer.py` — byte-level BPE (train / encode / decode / save / load)

The BPE is trained on **pre-split chunks with frequency counts** (each merge pass iterates unique chunks, not the 12 MB byte stream) so training is tractable in pure Python and cacheable to disk. `encode` is byte-level so it never fails on unseen characters; `decode(encode(x)) == x` is the canary the whole project depends on.

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/tokenizer.py`
- Test: `experiments/pravo_nn/tests/test_mini_gpt_tokenizer.py`

- [ ] **Step 1: Write the failing test**

```python
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

SAMPLE = (
    "Статья 1. Основные начала гражданского законодательства.\n"
    "Статья 2. Отношения, регулируемые гражданским законодательством.\n"
    "Гражданское законодательство основывается на признании равенства."
)


def test_roundtrip_is_lossless():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_roundtrip_on_unseen_text():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    other = "Совершенно новый текст 42 — с пунктуацией!"
    assert tok.decode(tok.encode(other)) == other  # byte-level: never OOV


def test_training_reduces_token_count_vs_raw_bytes():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    n_tokens = len(tok.encode(SAMPLE))
    n_bytes = len(SAMPLE.encode("utf-8"))
    assert n_tokens < n_bytes  # merges must compress something


def test_vocab_size_is_respected():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    assert len(tok.vocab) == 300


def test_special_token_is_atomic():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300, special_tokens=["<|endoftext|>"])
    ids = tok.encode("привет<|endoftext|>мир", allowed_special=True)
    eot_id = tok.special_tokens["<|endoftext|>"]
    assert ids.count(eot_id) == 1
    assert tok.decode(ids) == "привет<|endoftext|>мир"


def test_save_load_round_trips(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300, special_tokens=["<|endoftext|>"])
    tok.save(tmp_path)
    reloaded = BPETokenizer.load(tmp_path)
    assert reloaded.encode(SAMPLE) == tok.encode(SAMPLE)
    assert reloaded.special_tokens == tok.special_tokens
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_tokenizer.py -v`
Expected: FAIL — `cannot import name 'BPETokenizer'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Byte-level BPE tokenizer, implemented from scratch (minbpe-style, but trained
over frequency-counted chunks for speed). Base vocab is the 256 bytes; merges add
ids 256.. . Encoding is byte-level, so any input round-trips losslessly.

Files written by `save`: `vocab.json` (id -> hex of the token's bytes),
`merges.txt` (one `a b` int pair per line, in merge order), and
`special_tokens.json`."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Split into maximal runs of whitespace OR non-whitespace; keeps spaces attached
# to their run so the BPE can learn space-prefixed legal tokens (" Кодекса").
_SPLIT_RE = re.compile(r"\s+|\S+")


def _merge_seq(seq: tuple[int, ...], pair: tuple[int, int], idx: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}

    def train(
        self,
        text: str,
        *,
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        # Frequency-counted chunks -> each is a tuple of byte ids.
        chunk_counts = Counter(_SPLIT_RE.findall(text))
        words: dict[tuple[int, ...], int] = {}
        for chunk, cnt in chunk_counts.items():
            key = tuple(chunk.encode("utf-8"))
            words[key] = words.get(key, 0) + cnt

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for seq, cnt in words.items():
                for pair in zip(seq, seq[1:]):
                    stats[pair] = stats.get(pair, 0) + cnt
            if not stats:
                break
            best = max(stats, key=lambda p: stats[p])
            idx = 256 + i
            merges[best] = idx
            vocab[idx] = vocab[best[0]] + vocab[best[1]]
            words = {_merge_seq(seq, best, idx): cnt for seq, cnt in words.items()}

        self.merges = merges
        self.vocab = vocab
        self.special_tokens = {}
        for st in special_tokens or []:
            self.special_tokens[st] = len(self.vocab) + len(self.special_tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = list(_merge_seq(tuple(ids), pair, self.merges[pair]))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _SPLIT_RE.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, *, allowed_special: bool = False) -> list[int]:
        if not allowed_special or not self.special_tokens:
            return self._encode_ordinary(text)
        # Split out special tokens, encode the gaps ordinarily.
        pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
        out: list[int] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def decode(self, ids: list[int]) -> str:
        inv_special = {v: k for k, v in self.special_tokens.items()}
        parts: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in inv_special:
                parts.append(inv_special[i].encode("utf-8"))
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, out_dir) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "vocab.json").write_text(
            json.dumps({str(i): b.hex() for i, b in self.vocab.items()}),
            encoding="utf-8",
        )
        lines = [f"{a} {b}" for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])]
        (out / "merges.txt").write_text("\n".join(lines), encoding="utf-8")
        (out / "special_tokens.json").write_text(
            json.dumps(self.special_tokens, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, in_dir) -> "BPETokenizer":
        src = Path(in_dir)
        tok = cls()
        raw_vocab = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
        tok.vocab = {int(i): bytes.fromhex(h) for i, h in raw_vocab.items()}
        merges: dict[tuple[int, int], int] = {}
        text = (src / "merges.txt").read_text(encoding="utf-8")
        for rank, line in enumerate(filter(None, text.splitlines())):
            a, b = (int(x) for x in line.split())
            merges[(a, b)] = 256 + rank
        tok.merges = merges
        tok.special_tokens = json.loads((src / "special_tokens.json").read_text(encoding="utf-8"))
        return tok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_tokenizer.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/tokenizer.py experiments/pravo_nn/tests/test_mini_gpt_tokenizer.py
git commit -m "feat(pravo-nn): from-scratch byte-level BPE tokenizer"
```

---

## Task 4: `data.py` — encode corpus to a memmap + batch sampler

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/data.py`
- Test: `experiments/pravo_nn/tests/test_mini_gpt_data.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import torch

from experiments.pravo_nn.mini_gpt.data import encode_corpus, load_bin, get_batch
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

SAMPLE = "Статья 1. Основные начала.\nСтатья 2. Регулируемые отношения.\n" * 20


def _tok():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    return tok


def test_encode_corpus_writes_uint16_bin(tmp_path):
    tok = _tok()
    out = tmp_path / "train.bin"
    n = encode_corpus(SAMPLE, tok, out)
    assert out.exists()
    arr = load_bin(out)
    assert arr.dtype == np.uint16
    assert len(arr) == n


def test_get_batch_shapes_and_shift(tmp_path):
    tok = _tok()
    out = tmp_path / "train.bin"
    encode_corpus(SAMPLE, tok, out)
    data = load_bin(out)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.int64
    # within each sampled window, y is x shifted by one position
    x1, y1 = get_batch(
        data, block_size=8, batch_size=4, device="cpu",
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(x1[:, 1:], y1[:, :-1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_data.py -v`
Expected: FAIL — `cannot import name 'encode_corpus'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

Note: `get_batch`'s tunables are keyword-only (the `*`) so call sites read clearly.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_data.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/data.py experiments/pravo_nn/tests/test_mini_gpt_data.py
git commit -m "feat(pravo-nn): corpus tokenization to memmap + batch sampler"
```

---

## Task 5: `model.py` — GPT from scratch (forward + overfit probe + generate)

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/model.py`
- Test: `experiments/pravo_nn/tests/test_mini_gpt_model.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig
from experiments.pravo_nn.mini_gpt.model import GPT


def _tiny_cfg():
    return GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_forward_returns_logits_and_loss_shapes():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(x, targets=x)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_one_optimizer_step_reduces_loss_on_tiny_batch():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, loss0 = model(x, targets=y)
    for _ in range(50):
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
    _, loss1 = model(x, targets=y)
    assert loss1.item() < loss0.item()  # overfit probe: backward path works


def test_generate_appends_exactly_n_in_vocab_tokens():
    cfg = _tiny_cfg()
    model = GPT(cfg)
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (1, 6)  # 1 prompt token + 5 new
    assert int(out.max()) < cfg.vocab_size and int(out.min()) >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_model.py -v`
Expected: FAIL — `cannot import name 'GPT'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Decoder-only transformer (nanoGPT-style), built from scratch: causal
multi-head self-attention, pre-LN, GELU MLP, tied input/output embeddings.
Architecture is fully determined by `GPTConfig`."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from experiments.pravo_nn.mini_gpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head = C // self.n_head
        q = q.view(B, T, self.n_head, head).transpose(1, 2)
        k = k.view(B, T, self.n_head, head).transpose(1, 2)
        v = v.view(B, T, self.n_head, head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/model.py experiments/pravo_nn/tests/test_mini_gpt_model.py
git commit -m "feat(pravo-nn): GPT model from scratch (forward + generate)"
```

---

## Task 6: `train.py` — training loop + checkpoint writer

`train.py` owns the optimization loop and the checkpoint **writer** (`save_checkpoint`). The matching **reader** lives in `generate.py` (Task 7). The checkpoint dict is the stable #1↔#2 interface from the spec.

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/train.py`
- Test: `experiments/pravo_nn/tests/test_mini_gpt_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_checkpoint.py -v`
Expected: FAIL — `cannot import name 'get_device'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_checkpoint.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/train.py experiments/pravo_nn/tests/test_mini_gpt_checkpoint.py
git commit -m "feat(pravo-nn): training loop + checkpoint writer"
```

---

## Task 7: `generate.py` — checkpoint reader + sampling

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/generate.py`
- Test: `experiments/pravo_nn/tests/test_mini_gpt_generate.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_generate.py -v`
Expected: FAIL — `cannot import name 'load_checkpoint'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_generate.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/generate.py experiments/pravo_nn/tests/test_mini_gpt_generate.py
git commit -m "feat(pravo-nn): checkpoint reader + text generation"
```

---

## Task 8: Full suite green + real tokenizer train + short training smoke + README

This task wires the real corpus through the pipeline end to end. The full overnight training run is launched by the user afterward (per repo memory: long CPU runs use a detached process + Monitor, not a harness background task that dies ~10 min in).

**Files:**
- Create: `experiments/pravo_nn/mini_gpt/README.md`

- [ ] **Step 1: Run the whole mini_gpt test suite**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests -k mini_gpt -v`
Expected: all mini_gpt tests PASS (config, tokenizer, data, model, checkpoint, generate).

- [ ] **Step 2: Train the real tokenizer on the corpus (one-time, cached)**

Run:
```bash
py -3.13 -c "from pathlib import Path; from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer; d=Path('experiments/pravo_nn/data'); t=BPETokenizer(); t.train(Path(d,'corpus/corpus.txt').read_text(encoding='utf-8'), vocab_size=8000, special_tokens=['<|endoftext|>']); t.save(d/'tokenizer'); print('vocab', len(t.vocab)+len(t.special_tokens))"
```
Expected: prints `vocab 8001`; creates `data/tokenizer/{vocab.json,merges.txt,special_tokens.json}`. This may take several minutes (pure-Python BPE) — it is a one-time step.

- [ ] **Step 3: Encode the corpus to `train.bin`**

Run:
```bash
py -3.13 -c "from pathlib import Path; from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer; from experiments.pravo_nn.mini_gpt.data import encode_corpus; d=Path('experiments/pravo_nn/data'); t=BPETokenizer.load(d/'tokenizer'); n=encode_corpus(Path(d,'corpus/corpus.txt').read_text(encoding='utf-8'), t, d/'train.bin'); print('tokens', n)"
```
Expected: prints a token count (millions); creates `data/train.bin`. The token count should be well under the 12.1M bytes — confirms BPE compression.

- [ ] **Step 4: Smoke-train for a few steps to confirm the real loop runs and loss drops**

Run:
```bash
py -3.13 -m experiments.pravo_nn.mini_gpt.train --max-steps 50 --batch-size 8
```
Expected: logs `device=cpu params=~10M ...`, loss printed at step 0, decreasing by step 50; writes `data/checkpoints/ckpt.pt`. (Smoke test, not the real run — 50 steps won't produce good text.)

- [ ] **Step 5: Sample from the smoke checkpoint to confirm end-to-end generation**

Run:
```bash
py -3.13 -m experiments.pravo_nn.mini_gpt.generate --prompt "Статья 1." --max-new-tokens 60
```
Expected: prints text starting with `Статья 1.` followed by generated tokens (near-gibberish after only 50 steps — the point is the pipeline runs end to end and decodes valid UTF-8).

- [ ] **Step 6: Write the README**

Create `experiments/pravo_nn/mini_gpt/README.md`:

```markdown
# mini-GPT (sub-project #1)

Decoder-only transformer trained from scratch on the legal corpus from
sub-project #0. Design: `docs/superpowers/specs/2026-06-15-pravo-nn-mini-gpt-design.md`.

## Pipeline

    py -3.13 -m pytest experiments/pravo_nn/tests -k mini_gpt   # tests

    # 1. train tokenizer (one-time, cached) -> data/tokenizer/
    # 2. encode corpus -> data/train.bin
    #    (see plan Task 8 steps 2-3 for the exact one-liners)

    # 3. train (real run is hours on CPU; launch detached, monitor it)
    py -3.13 -m experiments.pravo_nn.mini_gpt.train --max-steps 5000

    # 4. generate
    py -3.13 -m experiments.pravo_nn.mini_gpt.generate --prompt "Статья 1." --max-new-tokens 200

## Checkpoint contract (interface to #2 LoRA)

`data/checkpoints/ckpt.pt` = `{model_state_dict, config, tokenizer, step, val_loss}`.
Sub-project #2 rebuilds `GPT(GPTConfig(**ckpt["config"]))`, loads the weights,
freezes them, and attaches LoRA adapters.

## Known ceiling

CPU + 12 MB corpus -> recognizably Russian *legal-register* text, not
legally-correct norms. Raising quality needs more data / a GPU run (config
scales up without code changes).

## Status

> Fill in after the first real training run: final loss, params, tokens,
> wall-clock, and a sample of generated text.
```

- [ ] **Step 7: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/README.md
git commit -m "docs(pravo-nn): mini-GPT pipeline README + smoke-run notes"
```

- [ ] **Step 8: (User-run, outside the plan) launch the real overnight training**

Per repo memory (`detached-long-runs`): launch via `Start-Process` detached + a persistent Monitor with a dead-man rule, NOT a harness background task (those die ~10 min into CPU LLM runs). After it finishes, fill in the README "Status" section — the checkpoint is then ready for sub-project #2.

---

## Definition of Done (from spec)

- [ ] Custom BPE tokenizer trains on `corpus.txt` and round-trips losslessly.
- [ ] GPT trains (loss decreases) and a checkpoint is saved.
- [ ] Generation script produces recognizable legal text from a prompt.
- [ ] Checkpoint format documented as the #1↔#2 interface.
- [ ] All tests green (`py -3.13 -m pytest experiments/pravo_nn/tests -k mini_gpt`).
- [ ] No code touches the production tree (`app/`).

## Self-Review notes (author checklist, already applied)

- **Spec coverage:** scaffold/location + artifact gitignore (Task 1), GPTConfig + preset (Task 2), from-scratch byte-level BPE with roundtrip/save-load/special-token (Task 3), corpus→memmap + batches with x/y shift (Task 4), GPT from scratch with forward-shape + overfit-probe + generate (Task 5), training loop + checkpoint writer contract (Task 6), checkpoint reader + sampling (Task 7), full-suite + real tokenizer/encode/smoke-train/generate + README + DoD (Task 8). Every spec section maps to a task.
- **Type/interface consistency:** `BPETokenizer.train(text, *, vocab_size, special_tokens)`, `.encode(text, *, allowed_special)`, `.decode(ids)`, `.save(dir)`, `.load(dir)` used identically in Tasks 3,4,7,8. `GPTConfig(vocab_size, block_size, n_layer, n_head, n_embd, dropout)` fields match across config/model/train/generate and the checkpoint `config` dict assertion. `GPT(cfg).forward(idx, targets)`/`.generate(idx, max_new_tokens, *, temperature, top_k)` consistent in Tasks 5,6,7. `save_checkpoint(model, cfg, *, tokenizer_dir, step, val_loss, path)` (Task 6) matches its calls in Tasks 6,7. `get_batch(data, *, block_size, batch_size, device, generator)` consistent in Tasks 4,6. Checkpoint dict keys `{model_state_dict, config, tokenizer, step, val_loss}` identical in writer (Task 6) and reader (Task 7).
- **Known one-time/slow steps:** pure-Python BPE training and corpus encoding (Task 8 steps 2–3) are cached; the real multi-hour training (Task 8 step 8) is user-launched detached, not in pytest — mirroring #0's mocked-network discipline.
```