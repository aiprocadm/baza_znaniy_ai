# Runbook: mini-GPT v2 — warm-start on mixed law+Wikipedia corpus

**Spec:** ../specs/2026-06-20-mini-gpt-v2-mixed-corpus-warmstart-design.md.
**Plan:** ../plans/2026-06-20-mini-gpt-v2-mixed-corpus-warmstart.md.
**Code:** PRs #610 (warm-start: wiki collector, corpus mix, val split, resume +
optimizer state + real val-loss) and #611 (`run_continue` entrypoint). All merged.

**Goal.** Continue training the loss-4.60 legal-only mini-GPT (sub-project #1) by
mixing ~6 MB of general Russian Wikipedia text ~50/50 with the legal corpus and
warm-starting from `ckpt_v1.pt`, to test the documented hypothesis that **grammar
was data-bottlenecked** — the legal-only corpus is one narrow register.

## Setup (as executed, all CPU — no GPU on this box)

- **Model:** 12.32M params, `block=256`, `vocab=4097` (the #1 BPE tokenizer reused
  unchanged, so weights warm-start directly — the #1↔#2 checkpoint contract holds).
- **Corpus:** `corpus_mixed.txt` = 12.06 MB, 50/50 by bytes
  (`law_bytes 6 002 092` / `wiki_bytes 6 003 135`; see `data/corpus_mixed.manifest.json`).
  Wikipedia half: 6.04 MB via the Wikimedia random-article API (`data/wiki/manifest.json`).
- **Split:** `encode_corpus_split(..., val_frac=0.05)` → `train.bin` / `val.bin`
  (held-out tail; val-loss is a real measurement, not the train loss).
- **Backup:** `ckpt_v1.pt` (the loss-4.60 legal-only base) kept — never overwritten.

## Loss trajectory (val_loss, by absolute step)

| step | val_loss | run | note |
|---|---|---|---|
| 2000 | ~4.60 (train) | #1 legal-only base | `ckpt_v1.pt` |
| 2000→6000 | 5.18 → **4.27** | warm-start (mixed) | initial distribution-shift bump to 5.85, then steady descent |
| 6000→8000 | → **4.10** | continue | |
| 8000→10000 | → **3.91** | continue | |
| 10000→12001 | → **3.72** | continue (2026-06-23) | current `ckpt.pt` |

**Reading.** Warm-starting onto the broader distribution produced the textbook
bump-then-recover: val_loss first *rose* (legal-only weights hitting unfamiliar
Wikipedia Russian) then fell well past the old 4.60 floor. The descent is **not
plateaued** at 12k steps — the last 2000 steps still bought −0.25 — so more steps
would lower loss further, with diminishing returns.

## Generation samples (current `ckpt.pt`, step 12001, temp 0.7–0.8, top_k 40)

```
Статья 1. → "...не установлено орган за ... на срок от двух месяца до трех лет
            либо лишением на срок до одного до двух лет. 2. То же деяние,
            совершенное ... по неосторожности либо ..."
Россия — это → "...в Сортском в 1976 году к Кадше и Белар ... 2 сентября ... года
            ... (1985 г.) ..."
В 1812 году → "...1965 года ... «На в Иосковн Прг» ... в Колкее городно ...
            (1985 г.) ..."
```

**What this shows — register transfer, not grammar.** The model now *switches
register by prompt*: a `Статья N.` prompt yields statute shape (sanction clauses,
the real Criminal-Code idiom «То же деяние, совершенное … по неосторожности», term
ranges «на срок от … до … лет»); a general prompt yields encyclopedia shape (dates,
place-name-shaped tokens, parenthetical years «(1985 г.)», «quoted titles»). The
Wikipedia half clearly added a second register the legal-only #1 never had.

**But fine-grained grammar still did not emerge.** In both registers the words are
largely invented ("организовой", "повлекния", "Сортском", "Иосковн") and
case/agreement is broken. The model captures *macro-structure and register* but not
*micro-grammar* (word formation, agreement).

## Verdict — hypothesis refined, not confirmed

- **Partial success.** Mixing general-language data measurably helped (val_loss
  4.60 → 3.72) and taught a second register — qualitatively unlike #1, which only
  ever produced legal word-salad. The "more/broader data helps" half of the
  hypothesis holds.
- **The grammar goal is NOT met at this scale.** Adding 6 MB of clean general
  Russian did not produce clean grammar. This *refines* the original
  "bottleneck = DATA, not CPU/params" claim: at **12.3M params / 12 MB corpus** the
  model is now **also capacity-limited**, not purely data-starved. True grammar
  (word formation + agreement) is below this architecture's reach.
- **Cheap lever (more CPU steps) has low marginal value toward the actual goal.**
  Continuing lowers loss but will not break the invented-word ceiling — that needs
  more parameters and/or much more corpus, i.e. a different scale tier, not more
  hours. Banked as an honest negative-ish result: the v2 pipeline (collector, mix,
  val split, warm-start/resume with optimizer state) is reproducible and reusable
  for any future larger-scale attempt.

## Reproduce

```powershell
# 1. collect wiki (live API; raw batches cached under data/wiki/raw/)
py -3.13 -m experiments.pravo_nn.wiki_collector.cli --target-bytes 12000000
# 2. mix 50/50 + 3. re-encode with val split (see plan Task 8 Steps 4-5 one-liners)
# 4. warm-start from the loss-4.60 base, detached (one torch process, FileHandler log)
Start-Process -FilePath py -ArgumentList '-3.13','-m','experiments.pravo_nn.mini_gpt.run_warmstart' -WorkingDirectory (Get-Location) -WindowStyle Hidden
# continue an interrupted/finished run (resumes absolute step from ckpt.pt):
py -3.13 -m experiments.pravo_nn.mini_gpt.run_continue 2000
# 5. judge by eye
py -3.13 -m experiments.pravo_nn.mini_gpt.generate --prompt "Статья 1." --max-new-tokens 160 --temperature 0.8 --top-k 40
```

Ops rules honored (repo memory): `py -3.13` only; exactly one torch process; long
runs detached via `Start-Process`; logging via `FileHandler` (flushes per record,
so a killed run keeps its tail); `ckpt_v1.pt` backed up before any warm-start.
