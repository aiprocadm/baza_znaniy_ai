# pravo.gov.ru corpus collector (sub-project 0)

Assembles ~20 Russian Federation codes into a clean Russian-text corpus for the
mini-GPT / LoRA / RAG sub-projects. See the design spec:
`docs/superpowers/specs/2026-06-14-pravo-corpus-collector-design.md`.

## Run

```
py -3.13 -m experiments.pravo_nn.corpus_collector.cli collect
py -3.13 -m pytest experiments/pravo_nn/tests
```

## Text source

source = **B. ИПС "Законодательство России"** (`pravo.gov.ru/proxy/ips/`).
The article body of a code is served at:

```
http://pravo.gov.ru/proxy/ips/?doc_itself=&nd=<ND_ID>&page=1&rdk=0
```

where `<ND_ID>` is the document's internal registry number (e.g. ГК РФ ч.1 =
`102033239`). **The id is NOT derivable from the slug** — Task 8 needs an
explicit `slug -> nd` map (the `/codex/` catalog and the ИПС search resolve it).

because, of the three candidates, B is the only one that returns clean
**selectable HTML** for the codes (verified: 587 KB, `text/html`, not a PDF, 300+
`Статья N` markers in a single fetch). Source A (`publication.pravo.gov.ru`)
publishes acts as image-PDF scans → would need OCR; source C
(`data.apicrafter.ru`) is metadata-only (200 rows, last updated 2021) behind
registration. See `corpus_collector/spike.py` for the probe that produced this.

Semantics: **current consolidated** — the served text is the in-force redaction
with inline amendment annotations (`(В редакции Федерального закона от …)`),
not a frozen point-in-time as-published act. The portal serves `windows-1251`
over an outdated TLS chain.
