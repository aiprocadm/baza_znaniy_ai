# Architecture decisions

## Two-path API design (intentional, not legacy)

The codebase exposes two parallel HTTP API surfaces:

- **`/api/kb/*`** (MVP) — single-tenant, auth via single `KB_API_KEY` env var,
  state in one SQLite file (`KB_MVP_DB_PATH`), primarily uses OpenAI-compatible LLM providers (DeepSeek/Groq/Ollama/etc.), with fallback to the app's `state.llm_provider` (llama.cpp when mounted in production) and extractive answers as last resort. Source: `app/api/kb_mvp.py`. Lightweight dependency tree
  (FastAPI + httpx). Designed for ≤10 person SMB self-hosted deployments.

- **`/api/v1/*`** (mature) — multi-tenant with JWT auth + RBAC,
  separate SQL database (PostgreSQL in production, SQLite in dev) for tenant metadata, Qdrant for vectors, supports llama.cpp +
  LoRA adapters. Source: `app/api/v1/*.py`. Heavy dependency tree (sqlmodel,
  qdrant-client, sentence-transformers, llama-cpp-python). Designed for
  multi-tenant SaaS or large internal deployments.

### Why two paths

The MVP path was added later for two reasons:

1. **Easier installation for SMB customers.** A `docker compose up` that needs
   nothing more than Python and SQLite ships in minutes; the mature stack
   requires Qdrant, model downloads, and several GB of dependencies.
2. **Simpler hardening surface.** Single-tenant, single-key auth is easier to
   audit for compliance than full RBAC. For one-off self-hosted deployments
   this matches what customers want.

### Do NOT merge them

A future contributor (human or AI agent) may notice the duplication of
concepts (documents, search, chat) and propose unifying. **This is wrong** at
the current stage:

- The MVP and mature paths target different customer profiles (single-tenant
  install vs SaaS).
- Merging would force MVP installs to carry full multi-tenant dependencies
  (~2 GB of model downloads, Qdrant, PostgreSQL drivers).
- The vision document (`docs/superpowers/specs/2026-05-22-project-vision-design.md`)
  explicitly defers the multi-tenant decision until after the first 5 paying customers.

### When to revisit

After 5 paying customers, evaluate:
- Are customers happy with the single-tenant MVP install?
- Has anyone explicitly asked for multi-tenant SaaS?
- Would unifying simplify maintenance enough to outweigh the install-weight cost?

Until then, treat the two paths as separate products in the same repo.

## Source-of-truth entrypoints

- **Production:** `uvicorn app.api.main:app` (loads `app/core/app.py:create_app`,
  full multi-tenant stack).
- **MVP-only dev:** `uvicorn scripts.dev_server_mvp:app` (loads only the
  `/api/kb/*` router, lightweight deps).
- **Legacy:** `backend/app/*` is deprecated, kept only for compatibility tests
  in CI job `legacy-compatibility-tests`.
