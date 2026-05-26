# Roadmap

**KB.AI is a side-project. It does not intend to become a startup.**

This document is **anti-roadmap-first**: it explicitly lists what we will NOT
build, because at 10-20 hours/week the most important decision is what to say
no to.

## Currently shipped (v1.0.0)

- MVP RAG pipeline: ingest → chunk → embed → search → answer with citations
- 6 LLM providers (DeepSeek, Groq, OpenRouter, OpenAI, Ollama, custom)
- 3 embedding backends (hashing, Ollama, OpenAI-compat API)
- Cross-encoder reranker (BGE multilingual)
- PDF citation viewer with text-search highlight (PDF.js)
- Multi-turn dialogues with SQLite history
- SSE streaming responses
- API key auth + DoS protection
- Audit log + admin endpoint
- i18n-ready UI (RU)
- kb-cli (backup/restore/reindex/health)
- One-click Linux/macOS install.sh

## Deferred (we will build this IF a real user asks)

These are valid feature requests but **none of them are guaranteed**.
We will revisit each only when a specific GitHub Issue describes the use case.

- **GigaChat / YandexGPT native integration.** Vision Phase 2; valuable for RU/CIS compliance customers. Deferred until: 1+ Issue from someone who can't use the current OpenAI-compat path.
- **LoRA Auto-Train UI.** Vision Phase 2; valuable for domain-specific customization. Deferred until: 1+ Issue from someone who wants to fine-tune on their own corpus and finds the existing `scripts/train_lora.py` too low-level.
- **Compliance Mode (per-country).** Env-flag scaffolding ships in v1.0; actual filtering of LLM providers deferred until first compliance-driven request. See `KB_COMPLIANCE_MODE` in `.env.example`.
- **Multi-tenant SaaS.** Deferred until 5+ paying customers exist for single-tenant version. Until then, run one installation per team.
- **Hybrid sparse+dense search.** Useful for large corpora; deferred until SQLite store starts showing query-time pain (>50ms p95 on a real corpus).
- **Document-level RBAC.** Single shared `KB_API_KEY` is the MVP choice. Deferred until a multi-user installation needs per-document permissions.

## Will NOT build (anti-roadmap)

These have been considered and rejected. Don't open an Issue for these without
a really specific use case.

- Slack / Teams / Telegram bot integrations — fragmentation overhead, can be done as a thin client by users.
- Mobile apps — responsive web covers 95% of usage.
- Real-time collaboration / multi-cursor editing.
- Agentic features / tool use / autonomous agents — wrong abstraction for KB use case.
- Workspaces / spaces / nested permissions.
- Vector-DB-as-a-service — Qdrant and Pinecone do this better.
- Cloud-hosted demo with persistent storage — operationally expensive, security-sensitive.
- "Migrate to Postgres for everything" — SQLite is the right scale for single-tenant; Postgres is opt-in for chat history only.

## Roadmap re-evaluation triggers

This document is re-read and (possibly) updated when:

- A new GitHub Issue requests a deferred item with concrete use case.
- 30 days have passed since v1.0.0 with 0 issues and 0 stars (project considered tilted toward "internal-only" outcome).
- A specific commercial inquiry arrives (`discovery-call` label).

## Contact

Feature requests → [GitHub Issues](https://github.com/aiprocadm/baza_znaniy_ai/issues). For discovery / commercial inquiries, tag the issue `discovery-call`.
