# Release checklist — KB.AI v1.0.0

Run through this before `git tag` and `GitHub release`. Manual smoke
catches things automated tests can't.

## 1. Code state

- [ ] On branch `feat/kb-mvp-corporate-rag` (or main after merge).
- [ ] Working tree clean: `git status --short` shows nothing.
- [ ] All commits squash-friendly (or already squashed) for the release PR.

## 2. Tests + lint

- [ ] `py -m pytest -q` → all green
- [ ] `ruff check .` → clean
- [ ] `black --check .` → clean

## 3. Manual smoke (clean Ubuntu VM or container)

Spin up a clean Ubuntu 22.04+ environment.

- [ ] `git clone https://github.com/aiprocadm/baza_znaniy_ai.git`
- [ ] `cd baza_znaniy_ai && bash install.sh`
- [ ] Server starts: `python3 -m uvicorn scripts.dev_server_mvp:app --port 8001`
- [ ] Open `http://localhost:8001/` — UI loads, no debug pills visible.
- [ ] Open `http://localhost:8001/?debug=1` — debug pills visible.
- [ ] Upload a PDF (any short text PDF).
- [ ] Wait for indexing → document appears in list.
- [ ] Ask a question in «Вопрос-ответ» tab.
- [ ] Answer renders with citations.
- [ ] Click a citation → PDF.js modal opens, page rendered, snippet highlighted.
- [ ] Close modal, reload page — state preserved.

## 4. kb-cli smoke

- [ ] `kb-cli --help` — shows 4 subcommands.
- [ ] `kb-cli backup /tmp/backup.tar.gz` — succeeds, manifest valid.
- [ ] `kb-cli restore /tmp/backup.tar.gz --data-dir /tmp/restored --yes` — succeeds.
- [ ] `kb-cli health --base-url http://localhost:8001` — exit code 0, prints OK.

## 5. Docs

- [ ] README.md renders cleanly on GitHub (check after push).
- [ ] All 3 screenshots present in `docs/screenshots/` and visible in README.
- [ ] CI badge shows passing.
- [ ] LICENSE, CONTRIBUTING, SECURITY, ROADMAP all link from README.

## 6. Tag + release

- [ ] `git tag -a v1.0.0 -m "Release v1.0.0"`
- [ ] `git push origin v1.0.0`
- [ ] Create GitHub Release with changelog (see release-notes template below).
- [ ] Verify release page renders, ZIP downloads work.

## Release notes template

```markdown
# v1.0.0 — first stable release

KB.AI is now usable as a self-hosted RAG over corporate documents.

## Highlights

- PDF citation viewer with text-search highlight (PDF.js)
- 6 LLM providers + 3 embedding backends + cross-encoder reranker
- Multi-turn dialogues + SSE streaming
- kb-cli for backup/restore/reindex/health
- One-click Linux/macOS install.sh

## What's NOT in this release (deferred to Phase 2)

- GigaChat / YandexGPT native integration
- LoRA Auto-Train UI
- Compliance Mode actual implementation (env-flag scaffold only)
- Multi-tenant SaaS

See ROADMAP.md for the full anti-roadmap.

## Install

bash install.sh # Linux/macOS
docker compose up -d --build # Docker (any OS)
```
