# AGENTS.md

This file provides guidance to OpenAI Codex when working with code in this repository.

## Source-of-truth entrypoints (read this first)

- **Production / CI / containers:** `uvicorn app.api.main:app` → `app/core/app.py:create_app`. Full stack (both API surfaces, multi-tenant deps).
- **MVP-only dev server (light deps, no Qdrant):** `uvicorn scripts.dev_server_mvp:app --reload --port 8001`.

## Two-path API design — do NOT merge

The repo intentionally ships **two parallel HTTP surfaces**:

- `/api/kb/*` — single-tenant MVP, single `KB_API_KEY` env, SQLite state. Source: `app/api/kb_mvp/` (package; split from the former single-file kb_mvp.py).
- `/api/v1/*` — multi-tenant with JWT/RBAC, Postgres + Qdrant. Source: `app/api/v1/*.py`.

Full rationale and the "when to revisit" criteria are in `docs/architecture.md`. **Unifying them is a known anti-pattern** — it forces MVP installs to carry ~2 GB of multi-tenant deps. If you spot the duplication and feel a refactor urge, read `docs/architecture.md` first.

## Two frontends — different jobs

- `frontend/` — React/Vite/TS app. The `package.json` calls itself **`operations-console`** (admin/diagnostic UI). This is where TS code, Vitest tests, and Tailwind config live.
- `data/www/` — the built-in **end-user MVP UI** served directly by FastAPI (vanilla HTML + i18n JSON in `data/www/i18n/`). Citation viewer (PDF.js) lives here.

When a request mentions "the UI", confirm which: end-user chat → `data/www/`; admin/ops → `frontend/`. Editing the wrong one is a common false-success failure mode.

## Runtime conventions on this machine (Windows)

- Python is invoked via the `py -3` launcher. **There is no `.venv`** — dependencies live on the user site-packages. Do not create a venv, do not run `python3.12` (Linux path). Use `py -3 -m pytest ...`, `py -3 -m pip ...` etc.
- The POSIX `Makefile` targets (`make test`, `make lint`) are for CI and Linux contributors. On Windows, run the underlying commands directly.

## Common commands

```powershell
py -3 -m pytest -q                           # full Python suite
py -3 -m pytest tests/test_api_v1_search.py  # single file
py -3 -m pytest -k "search_filters"          # by keyword
py -3 -m pytest -m "not requires_postgres"   # skip Postgres-marked tests
py -3 -m ruff check .                        # lint
py -3 -m black --check .                     # style
py -3 -m ruff check . --fix; py -3 -m black .   # autoformat (equiv. `make format`)
py -3 -m mypy app                            # type-check the active runtime tree
py -3 -m alembic upgrade head                # migrations
py -3 -m scripts.kb_cli health               # ops CLI: also backup/restore/reindex
docker compose -f compose.yml up -d --build  # full stack (README quick start)
```

`mypy app` carries a **pre-existing baseline (244 errors across 48 files)** — judge a change by *new* errors on touched lines, not the total count. (A single-file check like `mypy app/foo.py` reports a *different, smaller* total — it only follows that file's imports; don't compare the two.) Config: `[tool.mypy]` in `pyproject.toml`.

Frontend (in `frontend/`):

```powershell
npm install
npm run dev       # Vite dev server
npm run test      # Vitest with coverage
npm run lint      # eslint --max-warnings=0
```

CI on `.github/workflows/ci.yml` runs **path-scoped** jobs — changes to `app/**` skip the frontend job, etc. Mirror this when working locally: do not run unrelated suites.

## tests/stubs — a frequent gotcha

`tests/stubs/` replaces heavy third-party deps (Qdrant client, sentence_transformers, llama-cpp, etc.) when pytest collects. When a stub-backed test fails with `AttributeError` on a method that obviously exists in the real library, **suspect stub drift first** — the real prod code has moved ahead of the stub. Update the stub before assuming a code bug.

## Vector store layering

- **Protocol + factory + canonical filter contract:** `app/retriever/vector_store.py` (`VectorStore` Protocol, `SearchFilters` dataclass, `get_vector_store()`).
- **Backends:** `app/retriever/qdrant.py`, `app/retriever/faiss.py` — both must satisfy the Protocol and accept `SearchFilters`.
- **Service wrapper with in-memory fallback:** `app/services/vectorstore.py`. The fallback is a substring scan, **not a real vector search** — if Qdrant/FAISS fails at runtime, search silently degrades to grep (and records a degradation report via `app/observability/retrieval_health.py`). The `SearchFilters` filter-contract unification is **complete** (Sprints 1–2, #553–#556); the four files above carry zero `@pytest.mark.skip`. Any remaining repo skips belong to other subsystems (postgres/docker/LoRA/Linux-only).
- Two **back-compat shims** still exist (`app/qdrant_client.py`, `app/models/qdrant_client.py`) — treat as deprecated; do not add new code that imports them.

## Embedder gotcha

`KB_EMBEDDINGS_BACKEND` defaults to a hashing embedder when no real backend is configured (`app/services/kb_embeddings.py`). Hashing embeddings give **near-random semantic matches** — fine for unit tests, terrible for any real RAG eval. If a user reports "search returns nonsense", check this first.

**Switching backends requires a reindex:** hashing and Ollama/OpenAI embeddings have different vector dimensions. Changing `KB_EMBEDDINGS_BACKEND` without running `kb-cli reindex --embedder <new>` leaves the index incoherent — old vectors are no longer comparable to new query vectors. Symptom: search returns plausible-looking but unrelated chunks.

LLM provider auto-priority: DeepSeek > Groq > OpenRouter > OpenAI > Ollama, defined in `app/services/kb_llm.py`. New OpenAI-compatible providers go into `KNOWN_PRESETS` in that file.

## Plans and specs

Multi-step work follows the structured-plan workflow: a markdown file under `docs/superpowers/plans/YYYY-MM-DD-<slug>.md` with checkbox TDD steps. Design rationale lives next door in `docs/superpowers/specs/`. To resume in-flight work, read the latest plan there before touching code.

- **Plan checkboxes are NOT updated on completion** — every plan stays all `- [ ]` even when fully shipped (all 10 plans: 733 unchecked, 0 checked, yet every one is merged). **Determine real status from git history / merged PRs, not the boxes.** Sanity-check with `git log --oneline --all | grep -i <slug>` before assuming work remains.
- PRs often ship in named series (PR1/PR2/PR2b…) tracked in `docs/architecture.md`; check it to see which siblings already merged.

## Conventions (delta from CONTRIBUTING.md)

CONTRIBUTING.md covers Conventional Commits, TDD, `ruff + black + pytest` discipline, and PR size (~400 LoC max). Two project-specific notes for AI agents:

- **Anti-roadmap is binding.** `ROADMAP.md` enumerates explicitly rejected features (multi-tenant SaaS, Slack/Teams bots, agentic tool-use, etc.). Do not propose those without a concrete user request to point at.
- **Prefer `@pytest.mark.integration` (or a similar marker) over `@pytest.mark.skip`** when a test needs a real Postgres/Qdrant/Docker/LLM. Plain `skip` makes coverage gaps invisible; markers let CI selectively run them.
