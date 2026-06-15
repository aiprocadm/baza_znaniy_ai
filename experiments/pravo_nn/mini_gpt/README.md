# mini-GPT (sub-project #1)

Decoder-only transformer trained from scratch on the legal corpus from
sub-project #0. Design: `docs/superpowers/specs/2026-06-15-pravo-nn-mini-gpt-design.md`.
Plan: `docs/superpowers/plans/2026-06-15-pravo-nn-mini-gpt.md`.

Everything here is implemented from scratch in PyTorch (no HuggingFace
`transformers`/`tokenizers`): a byte-level BPE tokenizer, a nanoGPT-style
transformer, a training loop, and a generation script.

## Pipeline

    py -3.13 -m pytest experiments/pravo_nn/tests -k mini_gpt   # 20 tests

    # 1. train tokenizer (one-time, cached) -> data/tokenizer/
    # 2. encode corpus -> data/train.bin
    #    (one-liners in plan Task 8 steps 2-3; set vocab_size there)

    # 3. train (real run is HOURS on CPU; launch detached, monitor it)
    py -3.13 -m experiments.pravo_nn.mini_gpt.train --max-steps 5000 --batch-size 8

    # 4. generate
    py -3.13 -m experiments.pravo_nn.mini_gpt.generate --prompt "Статья 1." --max-new-tokens 200

## Checkpoint contract (interface to #2 LoRA)

`data/checkpoints/ckpt.pt` = `{model_state_dict, config, tokenizer, step, val_loss}`.
Sub-project #2 rebuilds `GPT(GPTConfig(**ckpt["config"]))`, loads the weights,
freezes them, and attaches LoRA adapters. The tokenizer directory it points at is
self-describing (it carries its own split pattern in `tokenizer_config.json`), so
the base model + tokenizer round-trip even across machines.

## Known ceiling

CPU + 12 MB corpus -> recognizably Russian *legal-register* text, not
legally-correct norms. Raising quality needs more data / a GPU run. The model is
device-agnostic (`get_device()` auto-detects cuda), so a GPU run is a config
change, not a rewrite.

## Status (first integration run, 2026-06-15)

Pipeline verified end-to-end on the real corpus:

- **Tokenizer:** byte-level BPE, **vocab 4097** (4096 merges + `<|endoftext|>`),
  trained on `corpus.txt` in ~5.5 min (pure Python, one-time cached).
  *Note:* vocab 4096 was chosen over the spec's 8000 — on a 6.5M-char corpus a
  smaller vocab keeps per-token occurrence counts healthy for a small model.
  Raise it for a longer / GPU run; it is a `vocab_size` argument, not a code change.
- **Corpus encoding:** 6,555,619 chars / 12,092,964 bytes -> **2,066,171 tokens**
  (**5.85 bytes/token** — solid BPE compression).
- **Model:** **12.32M parameters** (n_layer=6, n_head=6, n_embd=384,
  block_size=256), device=cpu.
- **Smoke train (50 steps, batch 8):** loss **8.4953 -> 5.3298** (start ≈
  ln(4097)=8.32, i.e. near-random, then dropping — the backward path works).
- **Generation:** decodes valid Russian UTF-8 and is prompt-anchored. After only
  50 steps the sample is not yet coherent (expected) — coherence needs the full run.
- **Throughput on this CPU:** ~3.7 s/step → ~5000 steps ≈ ~5 h. The full training
  run is therefore launched **detached** (not inside this tooling): use
  `Start-Process` + a polling monitor, per repo memory `detached-long-runs`
  (harness background tasks die ~10 min into long CPU runs).

### Next: the real training run (handoff)

Launch the ~5000-step run detached, then fill in the final loss + a real
generated sample below. After that the checkpoint is ready for sub-project #2 (LoRA).

- Final loss: _TBD after full run_
- Sample (`--prompt "Статья 1." --max-new-tokens 200`): _TBD after full run_
