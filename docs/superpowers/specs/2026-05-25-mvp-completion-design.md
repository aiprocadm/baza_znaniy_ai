# MVP Completion — Design

**Date:** 2026-05-25
**Branch:** `feat/kb-mvp-corporate-rag`
**Status:** Validated design. Source of truth для имплементации.
**Subordinate to:** [`2026-05-22-project-vision-design.md`](2026-05-22-project-vision-design.md)
**Related plans:** [`2026-05-22-foundation-cleanup-audit-i18n.md`](../plans/2026-05-22-foundation-cleanup-audit-i18n.md), [`2026-05-22-pdf-citation-viewer.md`](../plans/2026-05-22-pdf-citation-viewer.md) — оба завершены до этого документа.

---

## 1. Background

Vision-документ (2026-05-22) определил 6-месячный roadmap до первых платящих клиентов. За 3 дня (до 2026-05-25) реализовано **4 из 7** пунктов Phase 1: audit log, i18n, tech debt, PDF citation viewer. Осталось три: UI brand polish (1.1), one-click installer (1.3), backup/restore CLI (1.4).

Этот документ закрывает MVP **в контексте отказа от агрессивной SaaS-коммерциализации**. Автор после обсуждения выбрал dual-track «internal-tool + OSS publication» с сохранением optionality для будущего pivot, но без активной sales-работы в ближайшие 3 месяца.

Это **не отмена** vision'а. Vision Section 7.4 KILL #3 явно описывает «open-source и забыть про деньги» как легитимный исход. Этот документ — путь к этому исходу с сохранением возможности вернуться в SaaS-трек по pull-сигналу.

## 2. Goals / Non-goals

### Goals

1. **Internal-use стабильность.** `kb-cli backup/restore/reindex/health` — защита данных, миграция эмбеддеров без потери индекса, cron-monitoring.
2. **OSS-publication readiness.** Apache-2.0, README понятный outsider'у за 3-5 минут, install.sh для Linux/macOS, screenshots, SECURITY.md, CONTRIBUTING.md, GitHub Actions CI badge.
3. **Optionality для будущего pivot.** UI prod-mode без debug-pills (через `?debug=1`); `KB_COMPLIANCE_MODE` env-flag как заготовка (без implementation); `ROADMAP.md` с явным «deferred Phase 2».

### Non-goals

- ❌ **GigaChat / YandexGPT native** — Phase 2 vision'а. Отложено до pilot-запроса.
- ❌ **LoRA Auto-train UI** — Phase 2. Не продаём, не нужно сейчас.
- ❌ **Multi-tenant SaaS** — vision Section 6, deferred until ≥5 paying customers.
- ❌ **Compliance Mode implementation** — только env-flag + docs. Реальная фильтрация ждёт первого RU-клиента с явным compliance-запросом.
- ❌ **Live demo host** — self-host only по решению автора.
- ❌ **Heavy rebrand** (новое имя, новый логотип) — lightweight UI polish only.
- ❌ **Sales materials** (landing, pricing, one-pager) — vision Section 5.5, до Phase 2.
- ❌ **Discord / Slack / community channels** — psychological-trap mitigation.
- ❌ **SLA / bug bounty** — SECURITY.md явно отрицает.

## 3. Commerce-readiness criterion

После завершения этого MVP проект **НЕ становится commerce-ready**. Это сознательный выбор.

| Веха | Триггер | Источник |
|---|---|---|
| Можно поставить перед outsider'ом | ✓ После этого MVP | Этот дизайн |
| Можно начинать discovery calls | ✓ После этого MVP + warm network mapped (3ч) | Vision Section 5.1-5.2 |
| Готов к pilot agreement | ≥4 discovery calls с pain≥7 | Vision Section 5.3 |
| Готов к trial pricing | 3-5 pilots × 90 дней, 60-day review дал buy-сигналы | Vision Section 5.4 |
| Production pricing | 5+ платящих + case studies (≥6 мес работы) | Vision Section 5.4 |

**Самый ранний обоснованный gate** — discovery в течение Q3-2026 (если автор пойдёт этим путём; vision'ом не обязан).

**Самый честный ответ** при текущем выборе — **никогда** в SMB-SaaS-смысле. Коммерция возможна как pull от issues/contributors → consulting проекты (₽5-15K/час). Это vision Section 7.6 KILL #3 option (b).

Конкретный сигнал «можно начинать discovery»: **1+ GitHub issue с формулировкой «как настроить под нашу компанию?»** в течение 30 дней после Хабр-публикации.

## 4. Architecture

### 4.1 High-level layout

```
.
├── LICENSE                              ← NEW (Apache-2.0)
├── README.md                            ← REWRITE (outsider-first)
├── ROADMAP.md                           ← NEW (deferred Phase 2)
├── CONTRIBUTING.md                      ← NEW
├── SECURITY.md                          ← NEW
├── install.sh                           ← NEW (one-click для Linux/macOS)
├── .github/workflows/ci.yml             ← EXISTS (verify covers kb-cli; add badge in README)
├── docs/screenshots/
│   ├── chat-with-citations.png          ← NEW
│   ├── pdf-viewer-modal.png             ← NEW
│   └── upload-flow.png                  ← NEW
├── docs/release_checklist.md            ← NEW (manual smoke перед tag)
├── docs/legacy_README.md                ← NEW (архив текущего README)
├── scripts/
│   ├── kb_cli.py                        ← NEW (Typer entrypoint)
│   └── cli/
│       ├── __init__.py                  ← NEW
│       ├── backup.py                    ← NEW
│       ├── restore.py                   ← NEW
│       ├── reindex.py                   ← NEW
│       └── health.py                    ← NEW
├── data/www/index.html                  ← MODIFY (debug-pills behind ?debug=1)
├── app/api/kb_mvp.py                    ← MODIFY (extend /health)
├── tests/
│   ├── test_kb_cli_backup.py            ← NEW
│   ├── test_kb_cli_restore.py           ← NEW
│   ├── test_kb_cli_reindex.py           ← NEW
│   ├── test_kb_cli_health.py            ← NEW
│   ├── test_install_sh_smoke.py         ← NEW
│   ├── test_readme_outsider.py          ← NEW
│   └── test_ui_debug_hide.py            ← NEW
├── pyproject.toml                       ← MODIFY (+[project.scripts] kb-cli)
├── .env.example                         ← MODIFY (+KB_COMPLIANCE_MODE)
└── .gitignore                           ← MODIFY (+var/data/backups/)
```

### 4.2 Architectural decisions

1. **`kb-cli` как entry-point** через `pyproject.toml` → `[project.scripts]`. После `pip install .` команда доступна в PATH. OSS-norm.

2. **Typer вместо Click.** Type-hints native (соответствует Pydantic-style codebase). Если не в `requirements-runtime.txt` — добавить (одна строка).

3. **Backup format: `tar.gz` с `manifest.json`.** Содержимое: `kb_mvp.sqlite`, `kb_files/*.pdf`, `manifest.json`. Manifest schema:
   ```json
   {
     "version": "1.0",
     "created_at": "2026-05-25T12:00:00Z",
     "kb_mvp_db_path": "var/data/kb_mvp.sqlite",
     "file_count": 42,
     "total_bytes": 31457280,
     "embedder_used": "ollama:nomic-embed-text"
   }
   ```

4. **Restore — две стратегии: `replace` (default), `merge`.** `replace` сносит существующий стейт и заменяет; обязательный `--yes` или Y/n-prompt если БД не пустая. `merge` дедуплицирует по hash содержимого. `replace` всегда делает `cp -r kb_files kb_files.bak-<ts>` перед сносом.

5. **Reindex — atomic generator-based.** Не загружает все чанки в память. Создаёт `kb_chunks_new`, после успеха `BEGIN; DELETE FROM kb_chunks; INSERT FROM kb_chunks_new; COMMIT;`. Поддерживает `--from-document-id N` для resume.

6. **Health-check — расширение `/api/kb/health`.** Новые поля: `db_size_bytes`, `documents_count`, `chunks_count`, `disk_free_bytes`, `last_indexed_at`, `compliance_mode`, `compliance_implemented: false`. `kb-cli health` форматирует human-friendly + exit codes 0/1/2 для cron.

7. **UI debug-pill hiding — via `URLSearchParams.has('debug')`.** Если `?debug=1` отсутствует → `auth-pill`, `rerank-chip`, `providers-chip` получают `style.display = "none"`. Server-side не изменяется.

8. **`KB_COMPLIANCE_MODE` — только env-flag + docstring.** В `.env.example`:
   ```env
   # Compliance Mode (Phase 2 заготовка — не имплементирована):
   # KB_COMPLIANCE_MODE=ru_strict|kz_strict|by_strict|cis_universal
   # /health эхо-возвращает значение; фильтрация провайдеров — TODO.
   ```

9. **install.sh — Linux/macOS only.** Windows-пользователи имеют `py -m uvicorn` уже работающий. install.sh для outsider'ов на сервере. Не заменяет docker-compose — это лёгкий путь для тех, кто не хочет Docker.

10. **CI — только lint + tests, без deploy.** GitHub Actions matrix `python-version: ['3.12']`, runner `ubuntu-latest`. Triggers: `on: pull_request`. Badge в README.

## 5. Build sequence

Каждый Sprint имеет **abort point** — частичное завершение всё равно даёт ценность.

### Sprint 1: Internal-use solidity (~12ч)

| # | Component | Часов | Acceptance criteria |
|---|---|---|---|
| 1.1 | `kb-cli backup <out.tar.gz>` | 3 | `tar -tzf` показывает kb_mvp.sqlite + kb_files/ + manifest.json; round-trip test (backup→restore→diff=0); manifest все 6 полей |
| 1.2 | `kb-cli restore <archive.tar.gz> [--mode replace\|merge]` | 3 | `replace` требует `--yes` или Y/n если БД не пустая; `merge` дедупит по hash; backup перед `replace`; manifest-version + embedder-mismatch warning |
| 1.3 | `kb-cli reindex --embedder <name>` | 4 | Atomic via `kb_chunks_new`; progress-bar (tqdm); `--from-document-id N` для resume; `DELETE FROM kb_chunks WHERE embedder=?` старого после успеха |
| 1.4 | `kb-cli health` + extended `/api/kb/health` | 1 | HTTP-call, human-friendly format; новые поля (см. 4.2.6); exit codes 0/1/2 |
| 1.5 | `[project.scripts]` + `--help` | 1 | `pip install -e .` создаёт `kb-cli`; `--help` показывает 4 subcommand'а |

**Abort point:** после 1.1+1.2+1.4 (~7ч) — данные защищены, можно жить дальше.

### Sprint 2: Presentable state (~10ч)

| # | Component | Часов | Acceptance criteria |
|---|---|---|---|
| 2.1 | UI debug-pill hide за `?debug=1` | 2 | Default `/` → нет pills; `/?debug=1` → старое поведение; `test_ui_debug_hide.py` |
| 2.2 | README.md rewrite + archive old | 5 | Текущий README → `docs/legacy_README.md` (link из нового); новый README: первые 3 строки — ЧТО+КОМУ; quickstart ≤5 шагов; «Не для вас если...»; screenshots inline; H2-структура проверяется тестом |
| 2.3 | docs/screenshots/ (3 PNG) | 3 | chat-with-citations.png, pdf-viewer-modal.png, upload-flow.png; ≤500KB через `pngquant` |

**Abort point:** после 2.1+2.2 (~7ч) — UI чистый, README понятный.

### Sprint 3: OSS-ready (~8ч)

| # | Component | Часов | Acceptance criteria |
|---|---|---|---|
| 3.1 | LICENSE (Apache-2.0) | 0.5 | Каноничный текст; source-файлы без header-комментов |
| 3.2 | CONTRIBUTING.md | 1 | Dev-setup ≤5 строк; conventions (Conv. Commits, ruff/black, pytest); «Что НЕ принимаем» |
| 3.3 | SECURITY.md | 0.5 | aiproc.adm@gmail.com; no bug bounty; 30-day grace; known limitations |
| 3.4 | install.sh | 2 | Python 3.12+ check; idempotent `pip install .[runtime]`; copy `.env.example→.env`; финальный echo URL; `--dry-run` flag |
| 3.5 | CI badge + verify ci.yml covers kb-cli | 1 | Existing `.github/workflows/ci.yml` job `python-ci` уже делает lint+pytest на PR. Тест-файлы `tests/test_kb_cli_*.py` автоматически попадают в `pytest -q`. Добавить badge `![CI](...)` в README; smoke-проверка что новый Typer-dep не ломает `python-ci`. |
| 3.6 | ROADMAP.md | 1 | «KB.AI is a side-project; не startup»; Phase 2 vision'а deferred; compliance mode + env-flag mention |
| 3.7 | .env.example + KB_COMPLIANCE_MODE | 0.5 | Секция «# Compliance Mode (Phase 2 заготовка)» + /health эхо |
| 3.8 | release_checklist.md + .gitignore + tag v1.0.0 | 1 | `docs/release_checklist.md` — manual smoke шаги (clean Ubuntu VM, install.sh, upload PDF, ask, click citation); `.gitignore` добавляет `var/data/backups/`; CHANGELOG в GitHub Release; тег v1.0.0 на финальном коммите |

**Abort point:** после 3.1+3.2+3.6 (~3ч) — минимум OSS-compliance есть.

### Sprint 4 (опционально): Publication (~4ч)

| # | Component | Часов | Acceptance criteria |
|---|---|---|---|
| 4.1 | Хабр / Reddit r/LocalLLaMA пост | 3 | Заголовок «Self-hosted RAG для русских корпоративных документов с PDF-citation»; ссылка на v1.0.0 release |
| 4.2 | Post-publication monitoring | 1 | GitHub Issues labels: `feature-request`, `bug`, `discovery-call`; saved search для `discovery-call` |

### Total budget

| Sprint | Часов |
|---|---|
| 1 — Internal solidity | 12 |
| 2 — Presentable | 10 |
| 3 — OSS-ready | 7.5 |
| 4 — Publication (opt) | 4 |
| **Итого без Sprint 4** | **29.5** |
| **Итого с Sprint 4** | **33.5** |

Соответствует диапазону 30-40ч из Approach C ✓.

## 6. Testing strategy

| Слой | Что покрываем | Чем |
|---|---|---|
| kb-cli backup/restore | Round-trip integrity, manifest schema, replace vs merge dedup | `tests/test_kb_cli_backup.py`, `tests/test_kb_cli_restore.py` |
| kb-cli reindex | Atomicity, resume, embedder-dim mismatch | `tests/test_kb_cli_reindex.py` (фейковый embedder) |
| kb-cli health | HTTP-call parsing, exit codes 0/1/2, JSON schema | `tests/test_kb_cli_health.py` + FastAPI TestClient |
| install.sh | Shellcheck clean, `--dry-run`, Python-version detection | `tests/test_install_sh_smoke.py` (skipif Windows) |
| README structure | H2-секции, screenshots referenced, CI badge present | `tests/test_readme_outsider.py` (regex/markdown-it) |
| UI debug-pill hide | Default → нет `data-debug-pill`; `?debug=1` → видны | `tests/test_ui_debug_hide.py` (structural HTML) |
| Regression | Все 100+ существующих тестов остаются зелёными | `pytest -q` через CI |
| Manual smoke | E2E: install.sh на чистой Ubuntu VM → upload → ask → click citation | `docs/release_checklist.md` перед `git tag v1.0.0` |

**Не покрываем:** Browser E2E PDF.js (есть unit + manual), performance benchmarks (premature), multi-tenant (out of scope).

## 7. Risks + mitigations

| Риск | Вероятность | Impact | Mitigation |
|---|---|---|---|
| OSS-publish создаст maintenance-burden | High | High | SECURITY.md явно «no SLA»; ROADMAP.md «we defer most requests»; не отвечать на Issues 30 дней — норма; *не* добавлять Discord |
| Backup/restore — silent data loss | Low | Critical | Round-trip test; `restore --mode replace` всегда `cp -r kb_files kb_files.bak-<ts>`; `--dry-run` flag |
| Reindex прерывается → corrupt state | Medium | High | Atomic via `kb_chunks_new`; resume через `--from-document-id N` |
| install.sh ломается на дистрибутивах | Medium | Medium | Тест в Ubuntu CI; README — «tested on Ubuntu 22.04/24.04, Debian 12; иначе docker compose» |
| Compliance Mode env-flag забудут implement | Medium | Low | ROADMAP.md деферd-статус; `/health` отдаёт `{compliance_mode: null, compliance_implemented: false}` |
| README rewrite сломает SEO/links | Low | Low | Старый README → `docs/legacy_README.md` (link в новом) |
| CI жжёт GitHub Actions minutes | Low | Low | Public repo → unlimited на ubuntu-latest |
| Psychological trap «теперь это бизнес» | High | Medium | ROADMAP.md первая строка: «KB.AI is a side-project; не намерен становиться стартапом» |

## 8. Rollout

```
Day 0     Spec approved → /superpowers:writing-plans → impl plan
─────────────────────────────────────────────────────────────────────
Sprint 1  Internal solidity (12ч, растяжимо на 1-2 нед)
          opt: tag v0.9-internal
─────────────────────────────────────────────────────────────────────
Sprint 2  Presentable (10ч, 1-2 нед)
          можно показать другу — fishing for early feedback
─────────────────────────────────────────────────────────────────────
Sprint 3  OSS-ready (8ч, 1 нед)
          git tag v1.0.0 + GitHub Release
─────────────────────────────────────────────────────────────────────
Sprint 4  Publication (opt, 4ч)
          Хабр пост; ждём 30 дней; смотрим signals
─────────────────────────────────────────────────────────────────────
+30 дней  Checkpoint:
          - 0-5 stars, 0 issues → проект тихий, всё ок
          - 50+ stars, 10+ issues → внимание, переоцениваем (но НЕ обязаны в SaaS)
          - 1+ issue «как настроить под нас?» → обоснованная причина discovery
─────────────────────────────────────────────────────────────────────
```

## 9. Что произойдёт после Sprint 3

1. Надёжный internal-tool (`kb-cli backup` в crontab защищает данные).
2. Публикуемый OSS-проект (LICENSE + README + CI badge).
3. Открытая дверь в SaaS — НЕ открытая активно, а доступная.
4. Ясный gate для commerce: если кто-то конкретно попросит — это сигнал.
5. Право **закрыть проект** через 6 месяцев без чувства «недоделал».

Это **finished MVP**, не «потенциальная вершина», а honest finish line.

## 10. Open questions

Нет открытых вопросов на момент финального дизайна. Все стратегические и архитектурные выборы зафиксированы.

## 11. Document lifecycle

- Этот документ subordinate к vision'у. При конфликте — vision wins, обновляем этот.
- Переоценивается через 30 дней после `git tag v1.0.0` на основании publication-signals.
- Если результаты публикации триггерят pivot — пишется новый spec, этот замораживается.
