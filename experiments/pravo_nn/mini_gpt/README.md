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

## Status (full training run COMPLETE, 2026-06-15)

End-to-end on the real corpus, trained to completion:

- **Tokenizer:** byte-level BPE, **vocab 4097** (4096 merges + `<|endoftext|>`),
  trained on `corpus.txt` in ~5.5 min (pure Python, one-time cached).
  *Note:* vocab 4096 was chosen over the spec's 8000 — on a 6.5M-char corpus a
  smaller vocab keeps per-token occurrence counts healthy for a small model.
  Raise it for a longer / GPU run; it is a `vocab_size` argument, not a code change.
- **Corpus encoding:** 6,555,619 chars / 12,092,964 bytes -> **2,066,171 tokens**
  (**5.85 bytes/token** — solid BPE compression).
- **Model:** **12.32M parameters** (n_layer=6, n_head=6, n_embd=384,
  block_size=256), device=cpu.
- **Training:** 2000 steps, batch 8, AdamW + cosine LR. **loss 8.52 → 4.60**
  (final 4.5995). ~2 epochs over the corpus. The early descent stalled near the
  unigram entropy (~5.2) for ~0.3 epoch, then resumed past step ~450 — expected
  for a from-scratch LM accumulating context examples.
- **Throughput:** highly variable on this shared CPU — **~7 s/step idle, up to
  ~56 s/step under load** (Task Manager / browser / the editor compete for the
  16-core Meteor Lake). Wall time dominated by machine contention, not the model.
  Two operational gotchas hit and fixed during the run: (1) **never launch more
  than one torch process** — three concurrent runs oversubscribed the cores and
  every step thrashed to a near-halt; (2) `Start-Process -RedirectStandardError`
  **buffers** — log via a `logging.FileHandler` (flushes per record) instead, or
  progress looks frozen.

### Generated samples (final checkpoint, loss 4.60)

`--prompt "Статья 1." --max-new-tokens 200 --temperature 0.8 --top-k 40`:

> Статья 1. в на ( в в быть на на на в от в ( на или с в вы по ( на или о если в
> до или в быть при или и - с с в на с и в по в в и по вы без в не В на если быть
> до из Российской с и от в для на об и срок по на о и не и в в если или на суд в
> его об не на при на с в быть по для и суд при в на по срок к или в в

`--prompt "Статья 105."`:

> Статья 105. Федерации вми не - либо статьи и в о срок к в в в в (не быть об в из
> Российской на на и и и В их соответствии в к на в в ( вы также с его права
> настоящего по не и в быть или и вы и и на его Российской вы ( в на в о также на
> в и в от к в Российской Федерации в о по или и в до на об в а для срок в на
> если права или со на об также быть с до на и в к

**Honest quality read:** the model produces recognizable Russian **legal
register** — correct domain vocabulary and real multi-word fragments
("Российской Федерации", "в соответствии", "права настоящего", "срок", "суд",
"статьи") — but **no coherent grammar**. This is the documented ceiling for
loss ~4.6: it learned *what legal text is made of*, not *how to compose a norm*
(that needs loss ~3, i.e. many more epochs — realistically a GPU run). For the
goal of sub-project #1 (prove from-scratch training + a base for #2 LoRA) this
is a success: the pipeline works end to end and the checkpoint is a valid base.

### Next

The checkpoint `data/checkpoints/ckpt.pt` is ready for **sub-project #2 (LoRA)**.
To push raw mini-GPT quality higher: train many more epochs (lower loss toward ~3)
on a GPU / idle machine — `train.py` is device-agnostic, so it's a config change,
not a rewrite.
