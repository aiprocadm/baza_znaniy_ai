# KB.AI Project Vision — Design

**Date:** 2026-05-22
**Author scope:** solo, side-project (10-20 h/week), planning to go full-time if traction
**Geographic scope:** Russia + CIS (Kazakhstan, Belarus, Uzbekistan, Armenia, Moldova, etc.)
**Status:** Strategic vision document. Source of truth for product direction decisions.

> Этот документ не описывает технические фичи. Он описывает **куда идёт проект, кто платит, что строим в первую очередь и когда останавливаемся**. Все технические спецификации (вроде `2026-05-22-llm-providers-revision-design.md`) подчиняются этому документу.

---

## 1. Context and motivation

KB.AI — корпоративный RAG-продукт (Knowledge Base AI), построенный на базе FastAPI + Qdrant + llama.cpp + Docling, с поддержкой LoRA fine-tuning. На момент этого документа в коде есть:

- Multi-format ingestion (PDF/DOCX/PPTX/XLSX/MD/HTML/TXT) через Docling
- Векторный поиск (Qdrant + FAISS)
- Cross-encoder reranker (BGE multilingual)
- 6 OpenAI-совместимых LLM-провайдеров (DeepSeek, Groq, OpenRouter, OpenAI, Ollama, custom)
- Локальный llama.cpp + полный LoRA-пайплайн (QLoRA, GGUF conversion, hot-swap)
- Multi-turn диалоги, SSE streaming, API key auth, rate limiting
- Operations Console UI + MVP UI
- Docker compose стек, Alembic миграции, Prometheus metrics

**Технология готова.** Проблема — отсутствие ясного market focus и пути к монетизации. Этот документ закрывает этот пробел.

## 2. Founder constraints (важно для всего что ниже)

- **Время:** 10-20 ч/неделю в ближайшие 6-12 месяцев (side-project)
- **Готовность go full-time:** есть, при условии MRR ≥ ₽300K/мес и 3+ платящих
- **Сеть:** legal / consulting / finance в РФ и СНГ
- **Использование:** проект используется в реальной работе (dog-fooded) для двух задач:
  - Внутренняя база знаний для сотрудников (регламенты, процедуры)
  - Сборка информации из проектных документов / НПА / контрактов

Эти ограничения **жёстко** определяют всё что ниже. Нельзя строить SaaS на 1000 клиентов при 15ч/нед — мы строим под другую реальность.

## 3. Positioning

### 3.1 Value proposition (формула)

> **KB.AI** — self-hosted Russian-native AI-помощник по корпоративным документам для компаний в РФ и СНГ, которым **нельзя или невыгодно** использовать Notion AI / Microsoft 365 Copilot / ChatGPT.

**Russian-native** в этом контексте означает: русский язык как primary UX, документы и запросы преимущественно на русском, поддержка русско-английского смешанного контента, понимание русских юридических и деловых формулировок. UI остаётся **i18n-ready** для будущего перевода на казахский / белорусский / узбекский без переписывания, но локализация не делается до явного запроса от первого CIS-клиента (см. anti-roadmap).

### 3.2 Кто наш клиент (positioning by exclusion)

Под "нельзя или невыгодно" попадают четыре категории SMB в РФ/СНГ (50-1000 сотрудников):

1. **Compliance-чувствительные:** 152-ФЗ (РФ), Закон о персональных данных (KZ), ЗоЗПД (BY), data sovereignty (UZ). Данные не могут уйти в зарубежное облако.
2. **После санкций:** российские компании, у которых отрезаны прямые подписки на западные SaaS.
3. **Cost-conscious SMB:** не платят за Notion/MS 365 сейчас. Per-seat подписка ₽15-25K × N пользователей выглядит дороже одноразовой install ₽300-500K.
4. **Со сложными документами:** договоры, регламенты, технические спецификации, юр.документы, где **Docling layout-aware parsing** даёт значимо лучшее качество чем стандартные парсеры.

### 3.3 Anti-positioning (кто НЕ наш клиент)

- Компании, которые уже платят за Notion AI / MS Copilot и довольны качеством — их мы не переубедим
- Hyperscale-tech компании с большим IT-бюджетом и предпочтением cloud (они купят Glean)
- Микро-бизнес < 20 человек (TCO self-hosted = одно лицо собственника пилит сервер, дорого по их меркам)

### 3.4 Tagline-кандидаты для A/B-тестирования

- "Self-hosted ChatGPT для ваших документов. Без облака, под 152-ФЗ и СНГ-compliance."
- "AI-поиск по корпоративным регламентам и контрактам. У вас на сервере."
- "Корпоративный AI без зависимости от OpenAI и западных облаков."

## 4. Three differentiators (wedge)

В широкой "internal KB" нише против Notion AI / MS Copilot / Onyx / AnythingLLM нужно три **конкретных** дифференциатора, которые мы маркетируем громко и в которые вкладываемся.

### 4.1 LoRA per-tenant fine-tuning — главный moat

Никто из массовых SMB-конкурентов этого не делает. У нас можно за вечер обучить LoRA-адаптер на:
- Корпоративном жаргоне клиента (внутренние сокращения, продукты, проектные кодовые имена)
- Стиле формулировок их документов (генерация в том же стиле)
- Доменной лексике (legal terms, technical standards, мед.термины)

**Что нужно дополнительно построить:**
- Auto-LoRA UI: загрузил корпус → нажал кнопку → через 2-4 часа активен адаптер (сейчас manual: `train_lora.py` + convert + hot-load)
- Tenant-isolated registry адаптеров (расширить существующий `LORA_REGISTRY_DIR`)
- "До/после" метрики качества — наглядно показать клиенту улучшение

**Pitch:** *"AI, который говорит на языке вашей компании. Дообучаем модель на ВАШИХ договорах, регламентах, технической документации — за один вечер."*

### 4.2 Docling layout-aware parsing — техническое превосходство

Конкуренты используют PyPDF2/Tika/pypdfium2 — плоский текст без понимания структуры. Docling даёт корректные таблицы, нумерованные пункты, footnotes, captions, reading order. Это критично для корпоративных документов.

**Что нужно дополнительно:**
- Side-by-side демонстрация ("PyPDF2 vs Docling" на корпусе клиента) — материал для landing page и discovery
- Опциональный OCR для отсканированных документов (Docling умеет, сейчас `DOCLING_OCR_ENABLED=false`)
- Сохранение section-структуры в metadata для поиска "только в Главе 3"

**Pitch:** *"Понимает структуру документа: таблицы, нумерованные пункты, сноски. Не теряет ничего."*

### 4.3 Self-hosted-first + RU/CIS LLM-провайдеры — compliance moat

Notion AI / MS Copilot — cloud-only. AnythingLLM, Onyx — self-hosted, но не интегрированы с GigaChat / YandexGPT. Мы — единственное пересечение.

**Что нужно дополнительно:**
- Завершить GigaChat и YandexGPT интеграцию (см. `2026-05-22-llm-providers-revision-design.md`, **сокращённый scope**)
- **Compliance Mode** с per-country switching:
  - `KB_COMPLIANCE_MODE=ru_strict` — local llama.cpp + GigaChat + YandexGPT, запрещены западные
  - `KB_COMPLIANCE_MODE=kz_strict` — local llama.cpp + Ollama, запрещены даже GigaChat (их облако в РФ)
  - `KB_COMPLIANCE_MODE=by_strict` — аналогично KZ
  - `KB_COMPLIANCE_MODE=cis_universal` — только local llama.cpp + Ollama, никаких внешних API
  - Audit log попыток включения запрещённых провайдеров
- Whitepaper "Compliance с KB.AI" — официальный документ для CISO клиента

**Pitch:** *"Данные остаются у вас. Self-hosted с RU/CIS-провайдерами (GigaChat, YandexGPT) или полностью offline (Llama, Qwen, Saiga)."*

### 4.4 Table stakes (поддерживаем, не вкладываемся специально)

Multi-format ingestion ✓, reranker ✓, SSE streaming ✓, multi-turn dialog ✓, vector search ✓, API auth ✓. Конкуренты тут на похожем уровне — это не дифференциация.

### 4.5 Anti-roadmap (что НЕ делать)

При 15ч/нед эти "очевидные" фичи — ловушки:

- ❌ Slack/Teams/Telegram bot интеграция (огромный fragmentation overhead, не приносит revenue)
- ❌ Мобильное приложение (responsive web покрывает 95%)
- ❌ Real-time collaboration / multi-cursor
- ❌ Agentic features / tool use / autonomous agents (для KB-юзкейса overkill)
- ❌ Workspaces / spaces / nested permissions (до 10 платящих клиентов — единое API key)
- ❌ Векторная база как сервис (Qdrant и Pinecone делают это лучше)
- ❌ Локализация UI на 5+ языков **до** первого CIS-клиента, который её попросит
- ❌ Расширение LLM-провайдеров до 10 (изначальный план, теперь YAGNI — нужно 2-3)

## 5. Path to first 3-5 paying customers

Никакого cold outreach. Никаких рекламных бюджетов. Только тёплая сеть в legal/consulting/finance, которая часто имеет CIS-cross-border контакты.

### 5.1 Map warm network (неделя 1, 3 часа)

Составить список **15 A-контактов** (близкие, готовые дать 30 минут) + 30-50 B/C-контактов в legal/consulting/finance РФ и СНГ. Пометить:
- A = высокая вероятность ответа
- B = средняя
- C = только через интро через A или B

### 5.2 Discovery calls (недели 2-8, 8-12 встреч, ~30 часов)

**Не продажа.** Не показывать продукт. Не показывать цены. Слушать.

4 группы вопросов:
1. Текущая практика поиска информации в документах
2. Frustration moments за последний месяц
3. Что пробовали (ChatGPT? Notion?)
4. Минимум фич для серьёзного рассмотрения

Финал каждой встречи: запрос referral + email list для early access.

Уточнение для CIS-scope: добавить вопрос "работаете ли с CIS-клиентами / контрагентами?" — выявляет тех, кто видит боль шире РФ.

**Kill criterion #1 (месяц 4):** <4 discovery с pain ≥ 7 ИЛИ <1 pilot agreement → переключаемся на Direction A (legal-tech vertical).

### 5.3 Pilot stage (месяцы 3-5, 3-5 пилотов)

Из discovery выбрать 3 компании с высоким pain + готовых к pilot. Условия:
- Бесплатно на 90 дней, self-hosted установка
- В обмен: weekly 30-минутная feedback-встреча, право использовать как case study, ответ "за сколько купите?"
- Письменное pilot agreement (1 страница): обязательства, условия выхода, NDA

90-дневный цикл: install (1 день) → upload корпуса (1-3 дня) → observe usage → доработка 2-3 болей → 60-day review → 80-day pricing conversation.

**Цель:** 3 pilot → 2 paying.

### 5.4 First paying customers (месяцы 5-7)

**Trial pricing** (первые 3-5 клиентов):
- ₽30K / месяц при 12-мес контракте (₽360K/год)
- ИЛИ ₽250K разово + ₽5K/мес support за perpetual license

**Production pricing** (после 5 платящих + case studies):
- ₽60-80K / месяц
- ИЛИ ₽600K разово + ₽10K/мес

**Цель к концу 6 месяца:** 2 платящих × ₽30K = ₽60K MRR. Это сигнал PMF, не "успех".

### 5.5 Materials (недели 1-3, параллельно с discovery)

- **Landing page** одностраничник на Tilda/Webflow (4 часа): tagline, 3 wedge-фичи, форма "ранний доступ"
- **3-минутное демо-видео** (OBS/Loom): загрузка → 3 вопроса → 3 ответа с цитатами
- **One-pager PDF** для decision-maker'ов: проблема, решение, 3 wedge, цена, контакт
- **Один публичный case study** — ваш собственный использование KB.AI

## 6. Product roadmap (6 месяцев)

Каждая фича оплачена либо ответом на discovery, либо болью pilot-клиента. Без "на всякий случай".

### Фаза 1 (месяцы 1-2): Foundation, ~70 ч

| # | Задача | Часов |
|---|--------|-------|
| 1.1 | Customer-facing UI в `data/www/index.html`: убрать debug-инфо, RU-текст без жаргона, брендинг | 15 |
| 1.2 | Точность цитирования: `[документ.pdf, стр. 12, раздел 3.2]` + клик → pdf-вьюер с подсветкой | 20 |
| 1.3 | One-click installer: `curl ... \| sh` или `docker compose up` без doc-снежного-кома | 8 |
| 1.4 | Backup / restore CLI: `kb-cli backup` → tar.gz всех данных | 6 |
| 1.5 | Аудит-лог запросов: who, when, what, response. Доделать существующий `app/core/audit.py` | 8 |
| 1.6 | i18n-ready UI: gettext/i18next wrapper, не хардкодим RU-строки | 4 |
| 1.7 | Тех.долг: удалить `dev_kb_only.py`, синхронизировать MVP `/api/kb/*` и legacy `/api/v1/*` | 10 |

**Решение про two paths:** MVP `/api/kb/*` — основа для каждой инсталляции клиента. Mature `/api/v1/*` не трогаем. Пересматриваем после 5 клиентов.

### Фаза 2 (месяцы 3-4): Wedge productize, ~60 ч

| # | Задача | Часов |
|---|--------|-------|
| 2.1 | GigaChat + YandexGPT интеграция (по spec'у `2026-05-22-llm-providers-revision-design.md`, **сокращённый scope**) | 25 |
| 2.2 | Compliance Mode с per-country (ru_strict, kz_strict, by_strict, cis_universal) | 8 |
| 2.3 | LoRA Auto-Train UI: загрузка корпуса → выбор модели → "тренировать" → progress → активный адаптер | 20 |
| 2.4 | Docling parsing showcase: `POST /api/v1/admin/parse-preview` для side-by-side demo | 8 |

### Фаза 3 (месяцы 5-6): Pilot feedback driven, ~70 ч

**Не планируется в деталях.** Resource бюджет под фичи, валидированные через pilot-feedback. Принцип: ни одной фичи без явного pilot-запроса + проверки "если я это сделаю — подпишете контракт?".

### Cumulative budget

| Фаза | Часов кода | Часов sales | Календарно при 15-20ч/нед |
|------|------------|-------------|---------------------------|
| 1 | 70 | 20 (1-2 встречи/нед × 8 нед) | 4-6 нед |
| 2 | 60 | 25 | 4-5 нед |
| 3 | 70 | 30 (pilot management) | 4-6 нед |
| **Итого** | **~200** | **~75** | **~14 нед ≈ 3.5 мес** |

### Deferred (после 5 платящих клиентов)

- Multi-tenant SaaS vs. single-tenant per installation — решение по факту pull
- Slack/Teams/Telegram интеграция
- Mature `/api/v1/*` доработка
- Локальные LLM в Казахстане (ISSAI Kazakh-LLama, Sber Kazakhstan), Беларуси
- API для third-party интеграций
- Hybrid search (sparse + dense)
- Расширение LLM-провайдеров до 10 (Cerebras, Together, Mistral, etc.)

## 7. Success metrics and decision triggers

### 7.1 Weekly tracking (15 мин в воскресенье)

Простая таблица:
- Leading: discovery calls, avg pain score, referrals/call, pilot agreements, LoC/hour
- Lagging: paying customers, MRR, ARR run-rate, pilot→paid conversion, ACV

### 7.2 Checkpoint review (раз в 2 мес, 2 часа)

Формальный self-review записывается в `docs/superpowers/reviews/YYYY-MM-DD-checkpoint.md`. Шаблон:

```
Дата: ____
Часов вложено: ____
Discovery / Pilots / Paying / MRR: ____ / ____ / ____ / ₽____

Что валидировано / опровергнуто:
-
Что я отказался делать (anti-roadmap check):
-

Главное решение: [CONTINUE / PIVOT / ACCELERATE / STOP]
```

### 7.3 Checkpoint targets

| Месяц | Discovery | Pilot | Paying | MRR |
|-------|-----------|-------|--------|-----|
| 2 | 8+ | 1-2 (free) | 0 | ₽0 |
| 4 | 15+ | 3 (free) | 0 | ₽0 |
| 6 | 20+ | 3-5 | 1-2 | ₽30-80K |
| 9 | 30+ | 5+ | 3-4 | ₽100-200K |
| 12 | 50+ | 6-8 | 5-7 | ₽200-400K |

### 7.4 Kill criteria

**Месяц 4 KILL #1 (Discovery провалена):** <4 calls с pain≥7 ИЛИ <1 pilot готов → pivot на Direction A.

**Месяц 6 KILL #2 (Pilot провалена):** 3 pilots × 90 дней закончились, 0 готовы платить → pivot на Direction A или stop как pet project.

**Месяц 9 KILL #3 (Paid провалена):** 0 или 1 платящий, MRR<₽30K → варианты: (a) open-source и забыть про деньги, (b) перейти на консалтинг по AI/RAG с проектом как portfolio, (c) положить на полку.

**Месяц 12 KILL #4 (Не масштабируется):** 2-3 клиента, MRR₽60-100K, плато 3+ мес → lifestyle проект (₽1М/год side income) или закрыть.

### 7.5 Go full-time triggers

Прыжок full-time только при ВСЕХ трёх условиях:
1. MRR ≥ ₽300K/мес
2. 3+ платящих + 2-3 pilot в pipeline
3. Запас финансов на 9 месяцев

**Опасная зона "не прыгать вопреки желанию":** MRR ₽100-200K, 1-2 клиента, ощущение "вот-вот пойдёт". Это локальный максимум — частая ловушка. Дождитесь явного сигнала.

### 7.6 Что делать после kill criterion

Каждый kill — не провал, а заранее проговорённое решение:

- KILL #1 → Pivot Direction A: вся работа Фазы 1 (UI, инсталлятор, парсинг, цитаты) применима к legal-tech. Pivot = 2-4 нед переписать позиционирование и материалы.
- KILL #2 → Pivot Direction A или stop. Код становится open-source портфолио.
- KILL #3 → Консалтинг по RAG/корп-AI: ₽5-15K/час, продукт как demo. ARR ₽1-2М без сложного продукта.
- KILL #4 → Lifestyle / freeze: 2-3 клиента на support контрактах = ₽1-1.5М/год при 5-7ч/нед поддержки.

## 8. Status of LLM providers spec

Спецификация `2026-05-22-llm-providers-revision-design.md` написана **до** этого vision-документа. После применения vision'а scope сокращается:

| Phase из LLM-spec | Решение |
|--------------------|---------|
| Phase 0 (рефакторинг в пакет) | **Сохраняется**, низкий приоритет, делается в свободные слоты между discovery |
| Phase 1 (7 OpenAI-compat) | **Deferred to never** (YAGNI). Cerebras + Gemini опционально как easy wins, если останутся часы. |
| Phase 2 (AuthProvider + GigaChat) | **Сохраняется**, входит в Фазу 2 product roadmap'а |
| Phase 3 (YandexGPT native) | **Сохраняется**, входит в Фазу 2 product roadmap'а |
| Phase 4 (документация) | **Сокращается** до раздела "выбор провайдера для RU/CIS клиента" |
| Phase 5 (UI chip) | **Deferred to never** |

LLM-providers-spec остаётся валидным архитектурным документом, но как **subordinate** этого vision'а. Не блокирует product roadmap.

## 9. Open questions

Нет открытых вопросов на момент финального дизайна. Все стратегические выборы зафиксированы.

## 10. Document lifecycle

- Vision document **переоценивается каждые 8 недель** в checkpoint review
- Kill criteria применяются **строго** — без отмазок "ещё немного и пойдёт"
- Anti-roadmap пересматривается **только при явном pilot-запросе**, а не "потому что хочется"
- Когда product roadmap'а Фаза 3 заканчивается (месяц 6), пишется vision-v2 на основе данных пилотов
