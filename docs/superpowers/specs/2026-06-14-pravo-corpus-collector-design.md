# pravo.gov.ru corpus collector — design

**Date:** 2026-06-14
**Status:** approved (brainstorm), pending implementation plan
**Author:** brainstorm session (KB AI Dev)

## Why this exists (context)

The user wants to build their own neural network. The work was decomposed into
three independent sub-projects, each with its own spec → plan → implementation
cycle:

```
[Sub-project 0: pravo.gov.ru corpus collector]   ← THIS SPEC (built first)
      │ output: clean Russian legal corpus + manifest  (the contract boundary)
      ├──► [Sub-project 1: educational mini-GPT from scratch]   (own spec, later)
      ├──► [Sub-project 2: LoRA fine-tune of a large LLM]       (own spec, later)
      └──► [2b: load corpus into KB.AI RAG]                     (optional)
```

The collector is the shared dependency of all three. It knows nothing about
models; the models know nothing about scraping. The only coupling is the
`corpus/` directory plus `manifest.json`.

Goal-level decisions already settled during brainstorm:

- **Sub-project 1 goal** is *understanding* — a hand-written PyTorch transformer
  (nanoGPT-style), own BPE tokenizer (~4–8k), ~10–30M params (debugged on a tiny
  config first), trained to learn the *register* of Russian legal language.
  Factual correctness is explicitly **not** a goal of sub-project 1 — a small
  from-scratch generative model will produce plausible-sounding but invented
  legal text. Facts are the job of RAG (sub-project 2b) or LoRA (sub-project 2).
- **Corpus choice:** Russian text, specifically the **codes of the Russian
  Federation** (ГК, УК, НК, ТК, КоАП, ЖК, СК, etc. — ~20 documents). Rationale:
  large, coherent, canonically "legal" register, enumerable by hand, predictable
  size (tens of MB), and reusable by all three consumers.
- **Legal basis:** official documents of state bodies are excluded from
  copyright (ч.6 ст.1259 ГК РФ), and pravo.gov.ru publishes its data under an
  explicit open-data license (free to copy, modify, redistribute, including
  commercial use). The corpus is therefore legally clean.

## Scope of THIS spec

Only sub-project 0 — the collector. Sub-projects 1 and 2 get their own specs.

In scope:

- Fetching ~20 RF codes from pravo.gov.ru (or an agreed text source).
- Cleaning raw documents into normalized UTF-8 Russian text.
- Emitting a documented corpus artifact (`corpus.jsonl` + `corpus.txt`) and a
  `manifest.json` with provenance.
- Unit tests with the network mocked.

Out of scope: any model code, tokenizer training, training loops, RAG ingestion.

## Where it lives

Nothing here enters the production tree (`app/`). CLAUDE.md is strict about the
prod tree and ROADMAP.md marks own-LLM as rejected, so this is explicitly an
experiment:

```
experiments/pravo_nn/            # underscore: import-safe Python package
  __init__.py
  corpus_collector/
    __init__.py
    config.py            # list of codes (name + identifier) + chosen source
    fetch.py             # network layer: rate-limit, cache to data/raw/, idempotent
    extract.py           # raw -> clean UTF-8 text
    assemble.py          # clean -> data/corpus/ + data/manifest.json
    cli.py               # `py -3 -m experiments.pravo_nn.corpus_collector.cli collect`
  tests/
    test_extract.py      # fixture (one short code excerpt) -> expected clean text
    test_manifest.py     # manifest integrity
    test_idempotent.py   # re-run does not change corpus
    fixtures/            # committed sample raw document(s)
  data/                  # raw/ (gitignored), clean/, corpus/, manifest.json
  README.md
```

This sits under `experiments/`, so CI's path-scoped prod gates (`app/**`) are not
touched, and it does not violate the anti-roadmap (clearly research, not a
product feature).

> Naming: the directory is `pravo_nn` (underscore) so it is importable as the
> Python package `experiments.pravo_nn`. The git branch label uses a hyphen
> (`pravo-nn`) — branch names allow it, import paths do not.

## Architecture

Data flow:

```
config (list of codes)
  -> fetch.py    -> data/raw/    (raw API/page responses; on-disk cache; idempotent)
  -> extract.py  -> data/clean/  (clean UTF-8 text: headers/footers/page-numbers
                                  removed, whitespace normalized, article structure kept)
  -> assemble.py -> data/corpus/ corpus.jsonl  ({code, article, text, source_url, date})
                    data/corpus/ corpus.txt    (concatenated plain text for training)
                    data/manifest.json         (provenance: source, doc count, bytes,
                                                md5 per doc, collection date)
```

`raw/` is deliberately separated from `clean/`: the network is slow and fragile,
so caching raw bytes makes re-runs instant and reproducible, and cleaning can be
re-done arbitrarily without re-downloading.

### Step 0 — mandatory spike (decision gate)

Before building the full pipeline, take 2–3 codes and empirically determine
**which source gives the cleanest text**:

- API `publication.pravo.gov.ru` — confirmed to exist, returns JSON, read-only,
  open-data license. **Risk:** official publication is "electronic images of
  control copies", i.e. often **PDF scans** → may require PDF text extraction or
  OCR (tesseract-rus). This is the single biggest unknown in the project.
- ИПС "Законодательство России" `pravo.gov.ru/ips/` — serves acts as HTML/text.
- Third-party mirror `data.apicrafter.ru/packages/pubpravogovru` — already-parsed
  structured text.

Spike output: a short note "source = X, because Y", committed to the README.
Only then are `fetch.py`/`extract.py` built against the chosen source. The
"source of text" is an intentionally swappable detail (the agreed hybrid
approach: API for the document list/metadata, text from whichever source is
cleanest).

Rationale for spike-first: if the text turns out to be image-PDF, extraction cost
rises by an order of magnitude (OCR). Cheaper to learn that on 2 documents than
on 20.

## Output contract (the boundary consumers depend on)

`data/corpus/corpus.jsonl` — one JSON object per article/section:

```json
{"code": "ГК РФ", "article": "Статья 1", "text": "...", "source_url": "http://...", "date": "1994-11-30"}
```

`data/corpus/corpus.txt` — the same text concatenated (article separators), for
the mini-GPT training loop.

`data/manifest.json`:

```json
{
  "collected_at": "2026-06-14",
  "source": "<chosen in spike>",
  "documents": [
    {"code": "ГК РФ", "source_url": "http://...", "articles": 1551, "bytes": 1234567, "md5": "..."}
  ],
  "total_documents": 20,
  "total_bytes": 41234567
}
```

## Error handling

- Network: retry with backoff, respect rate limits; on persistent failure, log
  which code failed and continue (partial corpus is still usable), recording the
  gap in the manifest. No silent success — a failed fetch is visible in output.
- Extraction: if a document yields suspiciously little text (e.g. < threshold of
  expected size), flag it in the manifest rather than silently emitting garbage —
  this is the canary for an image-PDF that needs OCR.
- Idempotency: re-running with cached `raw/` must not change `corpus/` bytes.

## Testing

- `test_extract.py`: committed fixture (one short code excerpt in the spike's
  chosen raw format) → asserted clean text output.
- `test_manifest.py`: manifest schema + integrity (counts/bytes match files).
- `test_idempotent.py`: second assemble run over the same `clean/` is a no-op.
- Network is **mocked** — tests never hit the internet.
- Run: `py -3.13 -m pytest experiments/pravo_nn/tests` (per repo memory: pin
  `py -3.13`, the env where deps live; bare `py -3` resolves to a 3.14 with no
  pytest).

## Definition of done (sub-project 0)

- One command `py -3 -m experiments.pravo_nn.corpus_collector.cli collect`
  reproducibly assembles ~20 RF codes into clean Russian text (tens of MB) with a
  `manifest.json`.
- Tests green.
- Text source documented in README per the spike outcome.

## Open questions / risks

- **PDF vs text source** — resolved by the Step 0 spike; until then the extract
  cost is unknown.
- **Code "current version" vs "as published"** — pravo.gov.ru official
  publication is point-in-time acts; the ИПС/consolidated source gives the
  current consolidated code. For learning *register* either works; the spike note
  should record which semantics the chosen source has, since it matters for
  sub-projects 2/2b (facts).
```
