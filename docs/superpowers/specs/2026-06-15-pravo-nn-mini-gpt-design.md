# Mini-GPT on the Legal Corpus — Design (sub-project #1)

**Status:** approved (brainstorm 2026-06-15)
**Predecessor:** sub-project #0 — corpus collector (DONE, PR #592). Produces
`experiments/pravo_nn/data/corpus/corpus.txt` — ~12 MB clean Russian legal text,
6141 articles across 23 RF codes.
**Successor:** sub-project #2 — LoRA fine-tune, which loads this model's checkpoint.

## Goal

Train a small decoder-only transformer **from scratch** on the legal corpus that
(a) serves as a **base checkpoint for the #2 LoRA fine-tune**, and (b) aims to be
a **usable Russian-legal text generator**.

### Realistic Definition of Done (and the honest ceiling)

On **CPU-only, ~hours/overnight budget** with a **12 MB** corpus, the realistic
ceiling is a **~10M-parameter** model that produces *recognizably Russian legal*
text (correct register, `Статья N. …` structure, legal vocabulary), **not**
legally-correct norms. This limitation is recorded here deliberately — same
lesson as the reranker sub-project (tiny corpus caps quality; the fix is more
data / a GPU box, not more CPU tuning).

DoD:
1. A **custom BPE tokenizer** is trained on `corpus.txt` (own implementation, no
   external tokenizer dependency).
2. The GPT **trains** (loss decreases) and a **checkpoint is saved**.
3. A **generation script** produces recognizable legal text from a prompt.
4. The **checkpoint format is documented** so sub-project #2 (LoRA) can load it.
5. All code lives under `experiments/pravo_nn/` and **does not touch `app/`**
   (research, not product — the anti-roadmap rejects own-LLM *as a product*).

## Non-goals

- Not a product feature; not wired into `/api/kb/*` or `/api/v1/*`.
- Not legally-correct generation; not a replacement for the RAG MVP.
- No distributed training, no GPU-cluster scaling. Device-agnostic (cpu/cuda
  auto-detect) so a future GPU run is a config change, not a rewrite.

## Tech stack

- Python 3.13 (`py -3.13`), PyTorch (torch 2.12.0+cpu already installed on the
  user site-packages per repo memory — no venv).
- Model + tokenizer implemented **from scratch** (nanoGPT-style), stdlib + torch
  only. No HuggingFace `transformers`/`tokenizers` dependency for the core path.
- pytest for the test suite; heavy real training is **not** run inside pytest
  (only a few-step mock probe), mirroring how #0 mocked the network.

## Architecture

New package `experiments/pravo_nn/mini_gpt/`, structured like
`corpus_collector/`:

```
experiments/pravo_nn/mini_gpt/
  __init__.py
  config.py        # GPTConfig dataclass + one "CPU-overnight" preset (~10M)
  tokenizer.py     # byte-level BPE: train / encode / decode / save / load
  model.py         # decoder-only transformer from scratch (GPTConfig -> nn.Module)
  data.py          # encode corpus -> train.bin (memmap); random (x,y) batches
  train.py         # AdamW + cosine LR; device auto-detect; periodic ckpt + loss log
  generate.py      # load ckpt + tokenizer; sample with temperature + top-k
tests/
  test_tokenizer.py
  test_model.py
  test_data.py
  test_checkpoint.py
  test_generate.py
```

Responsibilities (one clear purpose each):
- `tokenizer` — bytes ⇄ token ids. Knows nothing about the model.
- `model` — ids → logits. Knows nothing about data layout or files.
- `data` — corpus text → on-disk token array → batches. Knows the tokenizer
  interface only.
- `train` — owns the optimization loop and the checkpoint *writer*.
- `generate` — owns the checkpoint *reader* and sampling.
- `config` — the single source of hyperparameters.

### Default preset (CPU-overnight, ~10M params)

`n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.1`,
BPE `vocab_size≈8000`. AdamW, cosine LR with warmup. All overridable via
`config.py` / CLI flags so a GPU run just scales the numbers up.

## Data flow

```
corpus.txt
  -> tokenizer.train()        -> vocab.json + merges.txt
  -> data.encode_corpus()     -> train.bin (+ optional val.bin split)
  -> train.py                 -> ckpt.pt   (weights + config + tokenizer ref)
  -> generate.py              -> sampled legal text
```

Tokenizer training and corpus encoding are **one-time, cached** steps (like #0's
raw-cache): re-running training reuses the existing `vocab/merges` and `train.bin`
unless asked to rebuild.

## Checkpoint contract (the #1 ↔ #2 interface)

`ckpt.pt` (torch.save of a dict) holds:

```python
{
  "model_state_dict": <state_dict>,
  "config": <GPTConfig as dict>,      # rebuild the architecture exactly
  "tokenizer": {"vocab": "vocab.json", "merges": "merges.txt"},  # relative paths
  "step": int,
  "val_loss": float,
}
```

This is the **stable interface** sub-project #2 depends on: LoRA loads this dict,
reconstructs the model from `config`, loads `model_state_dict`, **freezes** the
base weights, and attaches adapters. Agreeing on it now avoids re-training the
base when #2 starts.

## Error handling & guardrails

- **Tokenizer roundtrip is the canary:** if `decode(encode(x)) != x` for sample
  text, training is meaningless — asserted in tests and checked at encode time.
- **Overfit probe:** a test runs a few optimizer steps on one tiny batch and
  asserts loss goes *down* — catches a broken backward path before an overnight
  run is wasted.
- **No silent device fallback surprises:** `train.py` logs the chosen device and
  the param count at startup, so a CPU run is never mistaken for a GPU one.
- **Checkpoints are periodic**, not only at the end — an interrupted overnight
  run still leaves a usable base.

## Testing strategy (TDD)

- `tokenizer`: roundtrip equality, determinism, special-token handling, save/load.
- `model`: forward returns logits shaped `(B, T, vocab_size)`; one AdamW step
  decreases loss on a tiny synthetic batch.
- `data`: batches have shape `(B, block_size)`; `y` is `x` shifted by one.
- `checkpoint`: save → load reconstructs identical weights and config.
- `generate`: returns exactly N new in-vocabulary tokens for a given prompt.

Real multi-hour training is run manually (a `train` CLI invocation), not in
pytest; tests use a few-step mock to keep the suite fast.

## Risks

- **Quality ceiling** (primary): 12 MB + CPU may yield only weakly-coherent text.
  Mitigation: scope is explicitly "recognizable legal register", and the device
  -agnostic design lets a GPU run raise the ceiling without a rewrite.
- **Pure-Python BPE training speed** over 12 MB: mitigated by caching the trained
  merges to disk and only training once.
- **CPU throughput**: the ~10M preset is sized to fit an overnight CPU budget;
  if too slow, the preset shrinks (fewer layers / smaller block) via config.

## Out-of-scope follow-ups (later sub-projects)

- #2 LoRA fine-tune (loads this checkpoint).
- Any RAG-ingest of generated text. Not now.
