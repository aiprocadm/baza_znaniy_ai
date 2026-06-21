# Mini-GPT v2 — Continued Training on a Mixed Law+Wikipedia Corpus (warm-start)

**Status:** approved (brainstorm 2026-06-20)
**Predecessor:** sub-project #1 — mini-GPT (DONE+TRAINED, PR #593). Produced
`experiments/pravo_nn/data/checkpoints/ckpt.pt` — a 12.32M-param decoder-only
transformer, vocab 4097, trained 2000 steps to **train-loss 4.60** on the 23-code
legal corpus (`data/corpus/corpus.txt`, ~6.5M chars / 2.07M tokens).
**Relation to sub-project #2 (LoRA):** this run *improves the same base
checkpoint*. The #1↔#2 checkpoint contract (`model_state_dict`, `config`,
`tokenizer`, `step`, `val_loss`) is preserved; we only **add** keys.

## Goal

Continue training the existing mini-GPT to push it from "recognizable legal
*register* without grammar" (the documented loss-4.60 ceiling) toward **coherent
Russian grammar**, by **adding general-Russian text from Wikipedia** to the
training data — addressing the real bottleneck recorded in repo memory: **data,
not CPU**. The legal corpus is a narrow normative register with few ordinary
connected sentences; mixing in live encyclopedic prose supplies the grammar and
broad vocabulary the model never saw.

### Why warm-start (not retrain from scratch)

The user's intent is literally *"continue training."* We **reuse the existing
vocab-4097 tokenizer** so the model shape (embedding table) still matches
`ckpt.pt`, and **resume from the checkpoint** rather than discarding the loss-4.60
progress. Byte-level BPE never fails on unseen text — general Russian merely
tokenizes slightly less efficiently (more tokens/char) than legal text it was
tuned on. That efficiency loss is accepted in exchange for keeping the checkpoint
and a far cheaper CPU run. (Retraining the tokenizer to vocab 8000 + training
from scratch was considered and **rejected**: it throws away the checkpoint and
costs days on CPU.)

### Realistic Definition of Done (and the honest ceiling)

CPU-only, the loss-~3 grammar threshold is still **out of reach** — that needs a
GPU run, and this design does not change that. Success here is judged by
**direction, not an absolute number**:

1. A bounded clean Russian Wikipedia sample (**~12 MB**, sized to match the legal
   corpus so a 50/50 mix keeps *all* the law) is collected, idempotently and
   offline-testably, via the Wikimedia API (plain-text extracts, no markup).
2. A `corpus_mixed.txt` (~50/50 law/wiki by bytes) is assembled with a provenance
   manifest.
3. `train.py` can **resume** from `ckpt.pt` (weights **and** optimizer state) and
   report a **real held-out val-loss** — neither exists today.
4. A warm-start run on the mixed corpus completes and writes an improved
   checkpoint **without breaking the #2 LoRA contract**.
5. Qualitative read: generated samples show **more connected Russian** than the
   word-salad of the loss-4.60 samples (judged by eye + val-loss trend), even if
   absolute loss stays near 4.6 due to the distribution shift.

All code stays under `experiments/pravo_nn/` and **does not touch `app/`**
(research, not product — the anti-roadmap rejects own-LLM *as a product*).

## Non-goals

- Not a product feature; not wired into `/api/kb/*` or `/api/v1/*`.
- Not retraining the tokenizer; not changing model architecture or vocab.
- Not ingesting all of Wikipedia — a bounded ~12 MB sample only. The mixed corpus
  roughly doubles to ~24 MB; CPU cost is bounded by `max_steps`, not corpus size.
- No GPU-cluster scaling. Code stays device-agnostic so a future GPU run is a
  config change, not a rewrite.

## Tech stack

- Python 3.13 (`py -3.13`), PyTorch (already on user site-packages, no venv).
- Wikipedia fetch uses **stdlib `urllib` only** (mirrors `corpus_collector/fetch.py`),
  injectable `opener` for offline tests. No new third-party dependency.
- pytest for the suite; heavy real training is **not** run inside pytest (only a
  few-step resume probe), mirroring #0/#1.

## Architecture

Two new sibling packages under `experiments/pravo_nn/`, plus surgical edits to the
existing `mini_gpt/` training/data modules.

```
experiments/pravo_nn/
  wiki_collector/            # NEW — mirrors corpus_collector/ design
    __init__.py
    config.py               # WikiConfig: target_bytes (~12MB), batch size, lang, UA, endpoint
    fetch.py                # random-article plaintext via Wikimedia API (injectable opener, cache, backoff)
    clean.py                # strip "== headings ==", drop stubs, normalize whitespace
    assemble.py             # accumulate to target_bytes, dedupe by title -> wiki.txt + manifest
    cli.py                  # `py -3.13 -m experiments.pravo_nn.wiki_collector.cli`
  corpus_mix/               # NEW — small, single-purpose
    __init__.py
    assemble.py             # corpus.txt + wiki.txt -> corpus_mixed.txt (~50/50), provenance manifest
  mini_gpt/
    data.py                 # EDIT: encode mixed corpus; carve val.bin (held-out tail)
    train.py                # EDIT: resume_from, optimizer state, val-loss eval
  data/
    wiki/wiki.txt           # NEW collected sample (+ manifest.json)
    corpus_mixed.txt        # NEW assembled mix (+ manifest.json)
    train.bin               # RE-ENCODED from corpus_mixed.txt
    train_legal.bin         # NEW backup of the legal-only bin (reproducible #1 artifact)
    val.bin                 # NEW held-out tail for val-loss
    checkpoints/ckpt.pt     # warm-started in place (BACKED UP first to ckpt_v1.pt)
```

### Component 1 — `wiki_collector` (network + clean + assemble)

- **fetch.py:** `GET https://ru.wikipedia.org/w/api.php` with
  `action=query&format=json&prop=extracts&explaintext=1&exsectionformat=plain&
  generator=random&grnnamespace=0&grnlimit=N`. `explaintext=1` returns markup-free
  article text. Loop batches until `target_bytes` is reached. Per-batch raw JSON is
  cached to disk (`data/wiki/raw/batch-NNNN.json`) so re-runs are free and tests
  inject the `opener`. Polite: descriptive `User-Agent` header, small `sleep`
  between batches, linear backoff retry — same shape as `corpus_collector/fetch.py`.
  - **Determinism note:** `generator=random` is non-deterministic by nature. Tests
    inject a fixed opener returning fixtures; the live CLI just caches whatever it
    drew, and the manifest records article titles for provenance/reproducibility.
- **clean.py:** strip residual `== Section ==` headings, collapse blank runs,
  drop articles below a min length (disambiguation/stub pages add noise, not
  grammar). Pure function, fully unit-tested.
- **assemble.py:** concatenate cleaned articles, dedupe by title, stop at
  `target_bytes`; emit `wiki.txt` + `manifest.json` (article count, bytes, titles).

### Component 2 — `corpus_mix.assemble`

Reads `corpus.txt` (law, ~12.1 MB) and `wiki.txt` (wiki, ~12 MB), targets
**~50/50 by bytes** (truncate the larger source to the smaller's size,
parameterizable — sizing wiki to the law corpus means no law is discarded), writes
`corpus_mixed.txt` + a provenance manifest (bytes per source, ratio, source
file hashes). Concatenation order: law then wiki (the random batcher already
samples wiki shuffled; `get_batch` draws random windows, so global order doesn't
bias training).

### Component 3 — `mini_gpt/data.py` (re-encode + val split)

- Encode `corpus_mixed.txt` with the **existing** tokenizer → new `train.bin`.
  **Back up** the current legal-only bin to `train_legal.bin` first.
- New `split_bin(...)` / encode path reserves the **last ~5%** of tokens as
  `val.bin`. `get_batch` is unchanged; a thin `load_bin` already memmaps either
  file. Val batches are drawn from `val.bin`, train batches from `train.bin`.

### Component 4 — `mini_gpt/train.py` (resume + optimizer + val-loss)

- `save_checkpoint`: **add** `optimizer_state_dict` (additive key). #2 LoRA rebuilds
  from `config` + `model_state_dict` and ignores extra keys — to be verified the
  loader is not `strict`-keyed on the dict's top level.
- `train(resume_from: Path | None = None, ...)`:
  - If `resume_from` set: load ckpt, rebuild `GPT(GPTConfig(**ckpt["config"]))`,
    `load_state_dict` weights, `optimizer.load_state_dict` **if present** (backward
    compatible — the existing ckpt_v1 has none → weights-only resume), set
    `start_step = ckpt["step"]`.
  - **Hard guard:** assert `len(tok.vocab)+len(tok.special_tokens) == ckpt config
    vocab_size`; mismatch raises a clear error (prevents silently corrupting a
    warm-start by pointing at a re-trained tokenizer).
  - **LR schedule:** *warm restart* — a fresh short-warmup cosine over this run's
    `max_steps` (local step), while logged/saved `step` is `start_step + local`.
- **Val-loss:** `estimate_loss(model, val_data, ...)` averages cross-entropy over a
  few `val.bin` batches under `torch.no_grad()`, every `eval_interval`. Logged and
  written into the checkpoint's `val_loss` (now a *real* held-out number, not the
  last train-loss as in #1).
- CLI: `--resume [PATH]` (default `data/checkpoints/ckpt.pt`), `--from-scratch`,
  `--max-steps`, `--batch-size`, `--eval-interval`.

### Component 5 — Tests (TDD)

- **wiki fetch:** parses an API-JSON fixture; respects `target_bytes` cap; dedupes
  by title; fully offline via injected `opener`; retry/backoff on transient error.
- **wiki clean:** strips `== headings ==`, drops sub-min-length stubs, normalizes
  whitespace.
- **corpus_mix:** output is ~50/50 within tolerance; manifest byte counts match.
- **data val-split:** `val.bin` is the held-out tail; train/val don't overlap.
- **train resume:** restores `step`, weights, and optimizer state; a 2-step resume
  probe shows loss continues near the checkpoint loss (does **not** jump back to
  ~8.5 cold-init).
- **backward compat:** a checkpoint **without** `optimizer_state_dict` resumes
  (weights-only) without error.
- **#2 contract:** a warm-started checkpoint still loads via the #2-style rebuild
  (config + model_state_dict), extra keys ignored.

## Operational plan (from repo memory — these have bitten before)

- Run everything with **`py -3.13`** (bare `py -3` is a 3.14 with no torch/pytest).
- **Exactly one torch process at a time** — concurrent runs thrash the 16-core CPU
  to a near-halt (hit during the #1 run).
- Long warm-start run launched **detached** (`Start-Process`) with logging via a
  `logging.FileHandler` (flushes per record; `-RedirectStandardError` buffers and
  looks frozen). Monitor with a persistent dead-man rule (harness bg tasks die
  ~10 min into CPU LLM runs).
- **Back up `ckpt.pt` → `ckpt_v1.pt` before the warm-start run** so a bad run can't
  destroy the reproducible loss-4.60 #1 base.

## Risks & honest notes

- **Distribution shift bump:** warm-starting on a 50%-new corpus will bump loss up
  briefly while the model adapts — expected adaptation, not regression. Judge by
  the val-loss *trend* and sample quality.
- **CPU ceiling unchanged:** loss ~3 (true grammar) remains a GPU goal. The honest
  win here is *qualitatively* more connected Russian from real sentences, possibly
  at a similar absolute loss.
- **Tokenizer is legal-tuned:** general Russian → more tokens/char → effectively
  shorter context coverage per window. Accepted tradeoff of warm-start path A.
- **Licensing:** Wikipedia text is CC BY-SA. Fine for a local research model;
  provenance (titles, source) is recorded in the wiki manifest.
