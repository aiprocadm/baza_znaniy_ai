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
`102033239`). **The id is NOT derivable from the slug** — so `config.CODES`
carries an explicit `slug -> nd` map. All 23 ids were harvested from the ИПС
search and **verified live** (each fetched, `<title>` matched the code, healthy
article count) before being committed.

because, of the three candidates, B is the only one that returns clean
**selectable HTML** for the codes (verified: 587 KB, `text/html`, not a PDF, 300+
`Статья N` markers in a single fetch). Source A (`publication.pravo.gov.ru`)
publishes acts as image-PDF scans → would need OCR; source C
(`data.apicrafter.ru`) is metadata-only (200 rows, last updated 2021) behind
registration. See `corpus_collector/spike.py` for the probe that produced this.

Semantics: **current consolidated** — the served text is the in-force redaction
with inline amendment annotations (`(В редакции Федерального закона от …)`),
not a frozen point-in-time as-published act.

The body is served as `windows-1251` (the collector decodes `cp1251`). The
endpoint is plain `http://`, so no TLS is involved — the cert-chain error seen
via some HTTP clients only happens when a client force-upgrades to `https://`.

## Result (collected 2026-06-14)

One `collect` run over all 23 code documents produced:

- **23 documents**, **6141 articles**, **~10.8 MB** of clean UTF-8 Russian legal
  text (`corpus.txt` ≈ 11.6 MB with article separators).
- Small-doc canary (`suspiciously_small`): **none flagged** — every code
  extracted as real text, no image-PDF/OCR fallback needed.

Definition of done (sub-project 0) is met: the single `collect` command above
reproducibly assembles the corpus + `manifest.json`; the package suite
(`py -3.13 -m pytest experiments/pravo_nn/tests`) is green; the text source is
documented here; nothing touches the production tree (`app/`).

The corpus, raw cache, and manifest are generated artifacts (gitignored) — rerun
`collect` to rebuild them (cached `data/raw/` makes reruns instant).

## Known limitations

- **Trailing footer in the last article.** Text after the final «Статья N»
  marker (signature block, «Президент Российской Федерации», city/date) is
  included in the last article's `text`, since splitting is marker-based with no
  end sentinel. One article per document carries this tail. Acceptable noise for
  register-learning / RAG; revisit with a footer sentinel if a cleaner cut is
  needed.
- **Rare capitalized cross-references.** A «Статья N …» reference that happens to
  start its own line (after whitespace normalization) can still be mis-split into
  a fragment “article”. In-text (mid-line) references are correctly ignored.
