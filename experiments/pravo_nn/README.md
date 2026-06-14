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

> **DECIDED IN THE SPIKE (Task 3) — fill this in:** source = ___, because ___.
> Semantics: ___ (point-in-time as-published vs current consolidated).
