# Contributing to KB.AI

Thanks for considering a contribution. KB.AI is a side-project maintained
best-effort by one person, so please read the "What we will NOT accept"
section before opening a large PR.

## Dev setup (5 minutes)

```bash
git clone https://github.com/aiprocadm/baza_znaniy_ai.git
cd baza_znaniy_ai
python3.12 -m pip install -e .[dev]
pytest -q
```

For the MVP UI dev server (no Qdrant, no llama.cpp):
```bash
python -m uvicorn scripts.dev_server_mvp:app --reload --port 8001
```

## Conventions

- **Commit messages:** [Conventional Commits](https://www.conventionalcommits.org/) — `feat(area): ...`, `fix(area): ...`, `docs(area): ...`, `refactor(area): ...`, `chore(area): ...`.
- **Code style:** `ruff` + `black` (configured in `pyproject.toml`). Run `make format` before pushing.
- **Tests:** TDD when feasible. New features need at least one happy-path test and one edge-case test. `pytest -q` must pass.
- **PR size:** keep under ~400 added LoC. Larger refactors — open an Issue first to discuss scope.
- **Type hints:** required on new public APIs; encouraged everywhere.

## What we WILL accept

- Bug fixes with a regression test.
- New OpenAI-compatible LLM provider presets (add to `app/services/kb_llm.py:KNOWN_PRESETS`).
- New parsers for upload formats (extend `app/ingest/`).
- i18n translations (add `data/www/i18n/<lang>.json`).
- Documentation improvements.

## What we will NOT accept (without prior discussion)

- Multi-tenant / SaaS / billing features — explicitly out of scope (`ROADMAP.md`).
- Slack / Teams / Telegram bot integrations — fragmentation overhead.
- Mobile apps — responsive web covers it.
- Agentic / tool-use features — KB use-case is RAG, not autonomy.
- New abstractions ("framework" PRs without concrete user need).
- Forks of the embedder/reranker stack to add caching/parallelism without a benchmark showing >2x improvement on a real corpus.

If you're unsure, open an Issue first.

## Running the full suite

```bash
make lint   # ruff + black --check
make test   # pytest -q
```

CI runs both on every PR.
