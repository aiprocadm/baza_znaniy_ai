# Ревизия форков и план интеграций

Документ фиксирует решения по четырём форкам, изученным после Спринта 1:

| Форк | Upstream | Тип проекта | Вердикт |
|------|----------|-------------|---------|
| [`aiprocadm/LLMs-from-scratch`](https://github.com/aiprocadm/LLMs-from-scratch) | rasbt/LLMs-from-scratch | Образовательный (PyTorch GPT с нуля) | Reference, не зависимость |
| [`aiprocadm/strapi-cloud-template-blog-2b6cd9ca22`](https://github.com/aiprocadm/strapi-cloud-template-blog-2b6cd9ca22) | strapi/strapi-cloud-template-blog | Node.js CMS template | **Не интегрировать** |
| [`aiprocadm/langchain`](https://github.com/aiprocadm/langchain) | langchain-ai/langchain | Python framework для LLM | Выборочные идеи, не зависимость для `/api/kb/*` |
| [`aiprocadm/docling`](https://github.com/aiprocadm/docling) | DS4SD/docling | Парсер документов IBM | **Включить сейчас** |

Анализ выполнен через 4 параллельных Sonnet-агента. Сетевой доступ был ограничен — рекомендации опираются на известную верхнюю структуру upstream-репозиториев. Перед началом работы стоит сверить, не разошёлся ли форк с upstream.

---

## 1. LLMs-from-scratch — reference, не зависимость

### Что это
Учебник Sebastian Raschka «Build a Large Language Model (From Scratch)» — 7 глав на PyTorch (токенизатор → attention → transformer → pretrain → classification fine-tune → instruction fine-tune). Учебник, не библиотека.

### Что брать
| Идея | Источник | Куда |
|------|----------|------|
| Token-aware chunking (по токенам, а не символам) | ch02 sliding-window dataloader | `app/services/kb_store.py:split_text` через `tiktoken` |
| Sampling utilities для локального LLM (top_k, temperature) | ch05 `generate()` | `app/services/kb_llm.py` — fallback для Ollama-пути |
| Instruction dataset format (Alpaca-style) | ch07 `InstructionDataset`, `format_input` | Новый `app/training/instruction_dataset.py` — для сбора корпоративного датасета из истории Q&A |
| Classification head для роутера запросов | ch06 SpamDataset + classifier | Новый `app/services/kb_router.py` — «вопрос / FAQ / эскалация» |

### Что **не** брать
- ch03-ch04 (MultiHeadAttention, TransformerBlock, GPTModel) — у нас RAG поверх внешних LLM
- ch05 pretraining loop — обучать с нуля экономически бессмысленно
- Bonus chapters (KV-cache, FLOPS, LLaMA) — overkill
- Jupyter-ноутбуки целиком — формат не для production

### Вердикт
Взять **3 идеи** (token-chunking, instruction-dataset, classification router) и реализовать самостоятельно по референсу. Никаких `pip install` пакета и копирования `GPTModel`. К ch05 вернёмся, если будет своя локальная модель — за sampling/loading-утилитами.

---

## 2. Strapi blog template — НЕ интегрировать

### Что это
Форк официального Strapi v5 Cloud blog template — headless CMS на Node.js с админкой, RBAC, REST/GraphQL, Media Library. Content-types: Article, Author, Category, Tag, Global, About, Page.

### Аргументы ПРОТИВ
1. **Стек-разрыв**: +1 рантайм (Node 18+/Yarn), +1 БД (обычно Postgres), +1 контейнер, +1 источник логов
2. **Дублирование source of truth**: `KnowledgeBaseStore` уже хранит документы и чанки → Strapi станет вторым стором с двусторонней синхронизацией и conflict resolution
3. **Не решает целевую задачу**: chunking, embeddings, RAG — мимо. Strapi про CRUD + UI
4. **Vendor weight**: апгрейды v4→v5 болезненны, плагины отстают, админка тянет ~300 MB зависимостей
5. **Блоковый редактор**: rich-text JSON → нужен extractor → plaintext перед эмбеддингом

### Что плохого мы решаем без Strapi
| Зачем хотят CMS | Чем заменим |
|-----------------|-------------|
| Богатый редакторский UI | Directus поверх существующей `kb_mvp.sqlite` (zero-sync) или мини-админка FastAPI+HTMX |
| RBAC и роли | Существующий `app/core/auth.py` + `Settings.auth_disabled` flag |
| Media library | `/api/kb/documents/upload` + nginx статика |
| REST/GraphQL автоматом | OpenAPI генерируется FastAPI бесплатно |

### Если всё-таки понадобится (сценарий-проектория)
**Strapi как редакторский UI, FastAPI как brain**:
1. Content-type `KbDocument {title, body (blocks), tags, attachments}`
2. Lifecycle `afterCreate`/`afterUpdate` → webhook на `POST /api/kb/internal/sync` (HMAC-подпись)
3. FastAPI извлекает plaintext, чанкует, эмбедит, индексирует
4. Поиск/`/ask` остаются в FastAPI

Этот сценарий имеет смысл **только** если рядом с KB появится отдельный публичный сайт/блог.

### Вердикт
**Не интегрировать.** Стоимость > выгода. Если редактору нужен богатый UI прямо сейчас — **Directus** или мини-админка на HTMX, не Strapi.

---

## 3. LangChain — выборочно, не как зависимость для `/api/kb/*`

### Что уже используется
LangChain в `requirements*.txt` **не закреплён** — soft-зависимость, активируется через `LANGCHAIN_ENABLED=true`. Импорты под `try/except`.

| Файл | Что |
|------|-----|
| [`app/langchain/factory.py:41-46`](../app/langchain/factory.py) | `create_history_aware_retriever`, `create_retrieval_chain`, `create_stuff_documents_chain` |
| [`app/langchain/retrievers.py:11`](../app/langchain/retrievers.py) | `TenantFilteredQdrantRetriever` (dataclass) |
| [`app/retriever/qdrant.py:47-48`](../app/retriever/qdrant.py) | Опциональный импорт `Document` |
| [`app/core/config.py:836-848`](../app/core/config.py) | Флаги `langchain_enabled/mode/use_history_aware/return_source_docs/tracing/project` |
| [`app/services/chat_orchestrator.py:180-217`](../app/services/chat_orchestrator.py) | Ветка `ChatExecutionMode.LANGCHAIN` для `/api/v1/chat` (НЕ для `/api/kb/*`) |

### ⚠ Найден баг в существующей интеграции

В `app/langchain/factory.py` строки 59 и 63 — в `create_history_aware_retriever` и `create_retrieval_chain` передаются `str`-промпты, но API требует `ChatPromptTemplate.from_messages([...])`. При `LANGCHAIN_ENABLED=true` цепочка упадёт на первом вызове.

**Fix**: обернуть промпты в `ChatPromptTemplate`. Это надо сделать независимо от планов на интеграцию.

### Что брать **выборочно** (но без зависимости от langchain в /api/kb/*)

| Идея | Реализация без langchain |
|------|--------------------------|
| Cross-encoder reranker | `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")` напрямую |
| Streaming ответов SSE | `httpx.AsyncClient` с `stream=True` + `sse_starlette.EventSourceResponse` (~30 строк) |
| История диалогов | Нативно: таблицы `kb_conversations` + `kb_messages` в `kb_store.py` |
| Multi-query retrieval | LLM-вызов на 3-5 переформулировок → объединить hits |

### Что **не** брать
- `langchain-experimental` — нестабильный API
- Агенты/Tools/ReAct — overkill, добавляет латентность и непредсказуемость
- `langchain.vectorstores.Qdrant` (legacy) — есть свой клиент с tenant-фильтром
- `langchain-community` целиком — тянет 200+ зависимостей
- `LangSmith`-трейсинг в проде без оценки приватности промптов
- LCEL для замены `chat_orchestrator.handle_chat` — потеряем читаемость

### Вердикт
**MVP `/api/kb/*` не должен зависеть от langchain.** Берём идеи, пишем код сами. LangChain оставляем только в существующей ветке `/api/v1/chat` (`ChatExecutionMode.LANGCHAIN`). Перед использованием существующей ветки — починить баг в `factory.py:59,63`.

---

## 4. Docling — ВКЛЮЧИТЬ СЕЙЧАС

### Что это и как используется
[`app/ingest/docling_backend.py`](../app/ingest/docling_backend.py) обёртка над `DocumentConverter`. [`app/ingest/chunking.py:parse_document`](../app/ingest/chunking.py) умеет роутить на Docling при `DOCUMENT_PARSER_BACKEND=docling|auto` + `DOCLING_ENABLED=true` + `import docling` успешен.

**Сейчас Docling выключен** — `.env.example` ставит `DOCUMENT_PARSER_BACKEND=legacy` и `DOCLING_ENABLED=false`. MVP пользуется только legacy-парсерами (pdfminer / python-docx / pypdf).

### Какие фичи Docling не используются
- **Layout-aware parsing** (DocLayNet) — таблицы, фигуры, captions, reading order
- **Table extraction** (TableFormer) — структурированные ячейки + Markdown
- **Selectable OCR engine** — EasyOCR / RapidOCR / Tesseract вместо дефолта
- **HTML / AsciiDoc / CSV** ingestion
- **HybridChunker / HierarchicalChunker** — структурное чанкование
- **VLM picture description** — captions для изображений (отложить)
- **ASR audio transcription** (Whisper-Turbo) — отложить

### Что включить **прямо сейчас** (zero risk, благодаря fallback на legacy)

| Env | Сейчас | Сделать | Эффект |
|-----|--------|---------|--------|
| `DOCUMENT_PARSER_BACKEND` | `legacy` | `auto` | Docling для поддержанных MIME, fallback на legacy при ошибке |
| `DOCLING_ENABLED` | `false` | `true` | Открывает auto-ветку |
| `DOCLING_OCR_ENABLED` | `false` | `true` (с пониманием первого медленного запуска) | OCR без Tesseract |
| `DOCLING_TIMEOUT` | `60` | `180` | 60s слишком тесно для 50-страничного PDF на первом запуске |
| `UPLOAD_ALLOWED_EXTS` | `pdf,docx,pptx,xlsx,txt,md` | `+html` | Docling умеет HTML |

**Net gain без кода**: table extraction, layout-correct reading order, captions в Markdown, единый pipeline для PDF/DOCX/PPTX/XLSX.

### Что доработать в коде

| Задача | Сложность | ROI |
|--------|-----------|-----|
| Default `auto` вместо `legacy` в `.env.example` | low | high |
| `export_to_markdown()` first в `_extract_page_texts` (сейчас Markdown — fallback) | low | high |
| Передавать `PdfPipelineOptions` явно (`do_ocr`, `do_table_structure`, `ocr_options=RapidOcrOptions()`) | medium | medium |
| `HybridChunker` для Docling-парсенных вместо плоского re-chunking | medium | high — биггест retrieval-win |
| Фигуры/таблицы как типизированные чанки (`kind: "table"`, `kind: "figure"`) | medium | medium |
| VLM picture description | high | defer |
| Audio ASR | high | defer |

### Производительность (10 PDF × 50 страниц)
- Legacy: ~20-90s CPU
- Docling text-only: ~1-3s/page → 8-25 мин для 500 страниц
- Docling + EasyOCR: ~4-8s/page
- Docling + RapidOCR: ~2-4s/page; +3-5× ускорение на GPU
- Первый вызов: +30-90s одноразовая загрузка моделей (DocLayNet, TableFormer, OCR)

Рекомендация: держать ингест в фоновом воркере (`INGEST_USE_LOCAL_QUEUE=true` уже стоит), увеличить `DOCLING_TIMEOUT` до 180.

### Вердикт
**Включить прямо сейчас** (1-2 часа): флипнуть env, добавить markdown-first в `_extract_page_texts`, расширить `UPLOAD_ALLOWED_EXTS`. Это zero-risk благодаря legacy-fallback.

**Спринт после**: RapidOCR через explicit `PdfPipelineOptions`, `HybridChunker` для Docling-документов.

**Defer**: VLM и ASR — требуют дополнительных моделей/ключей.

---

## Сводный план интеграций

### Принципы
1. `/api/kb/*` остаётся **самодостаточным** — никаких новых тяжёлых зависимостей
2. Берём **идеи**, пишем **свой код** там, где это даёт ту же ценность за меньшую сложность
3. Каждый блок проходит ту же приёмку, что MVP: тесты + диагностика через `/api/kb/health`

### Приоритезированный план

#### Спринт 3.A — Docling включить (1-2 часа, **HIGH ROI**, **LOW RISK**)
- [ ] `.env.example`: `DOCUMENT_PARSER_BACKEND=auto`, `DOCLING_ENABLED=true`, `DOCLING_TIMEOUT=180`
- [ ] `docling_backend.py`: переставить `export_to_markdown()` в начало `_extract_page_texts`
- [ ] `kb_mvp.py`: добавить `"html"`, `"htm"` в `SUPPORTED_UPLOAD_EXT`
- [ ] Тест: upload PDF с таблицей → проверить, что таблица попадает в чанк как Markdown
- Acceptance: `/api/kb/health` показывает `docling: enabled`, upload PDF с таблицей возвращает 201

#### Спринт 3.B — Cross-encoder reranker (~3 часа, **HIGH ROI**)
Без langchain — через `sentence_transformers.CrossEncoder` напрямую.
- [ ] `app/services/kb_rerank.py` — обёртка `CrossEncoderReranker(model="BAAI/bge-reranker-v2-m3")` с lazy init
- [ ] `kb_store.search(top_k=20)` → `rerank → top_n=4`
- [ ] env: `KB_RERANK_ENABLED=true/false`, `KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3`, `KB_RERANK_TOPN=4`
- [ ] Тест: проверка, что reranker меняет порядок hits
- Acceptance: `/ask` показывает precision@1 выше при включённом reranker (subjective eval)

#### Спринт 3.C — История диалогов (1 день, **HIGH ROI**)
Нативно в `kb_store.py`, без langchain.
- [ ] Таблицы `kb_conversations(id, created_at, title)` + `kb_messages(id, conversation_id, role, content, citations_json, created_at)`
- [ ] `AskRequest.conversation_id: Optional[str]`
- [ ] Если `conversation_id` передан — добавить N последних сообщений в контекст RAG-промпта
- [ ] `GET /api/kb/conversations`, `GET /api/kb/conversations/{id}/messages`
- [ ] UI: tab «История диалогов»
- Acceptance: 36 тестов + 5 новых на conversation flow

#### Спринт 3.D — Streaming SSE (полдня, MEDIUM ROI)
Через `httpx.AsyncClient` + `sse-starlette`, не через langchain.
- [ ] `AsyncOpenAICompatibleProvider` рядом с sync-версией (или async-метод `generate_stream`)
- [ ] `POST /api/kb/ask/stream` — `EventSourceResponse`, события `meta` / `token` / `sources` / `done`
- [ ] UI: переключатель «потоковый ответ»
- Acceptance: visible token-by-token output для DeepSeek/Groq

#### Спринт 3.E — Token-aware chunking (полдня, **MEDIUM ROI**)
Идея из LLMs-from-scratch ch02.
- [ ] `app/services/kb_chunking.py` — token-aware sliding window через `tiktoken`
- [ ] Использовать только если `KB_CHUNK_TOKENIZER=tiktoken`; default — символьный (backward-compat)
- [ ] Тесты на корректность токенных границ
- Acceptance: чанки помещаются в контекст LLM без обрезки

#### Спринт 3.F — Fix существующего langchain бага (15 минут)
- [ ] `app/langchain/factory.py:59,63` — обернуть строки в `ChatPromptTemplate.from_messages([("system", ...), ...])`
- [ ] Прогнать `/api/v1/chat` ветку с `LANGCHAIN_ENABLED=true`

### Что **отложить** до явного запроса

| Идея | Источник | Почему отложить |
|------|----------|-----------------|
| Strapi/Directus CMS | Strapi | Нет потребности от редакторов; UI решается через HTMX |
| Multi-query retriever | LangChain | Сначала измерить, насколько reranker уже улучшил precision |
| Self-query retriever (NL→filters) | LangChain | Нужны структурированные metadata и LLM-парсер фильтров — high complexity |
| Instruction dataset для fine-tune | LLMs-from-scratch ch07 | Нужно сначала накопить данные через историю диалогов |
| Query classification router | LLMs-from-scratch ch06 | Сначала собрать 200-500 примеров |
| Своя локальная LLM | LLMs-from-scratch ch03-5 | Внешние API (DeepSeek/Groq/Ollama) покрывают потребности |
| VLM picture description | Docling | Не критично, +модели/GPU |
| ASR audio transcription | Docling | Не было запроса |

---

## Open questions

1. **Reranker модель**: `BAAI/bge-reranker-v2-m3` (~600 MB) vs более лёгкий `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB). Если корпоративный документооборот в основном русский — m3 предпочтительнее.
2. **Conversation persistence**: TTL сообщений? Сейчас в `.env.example` есть `CHAT_HISTORY_LIMIT=12` — переиспользовать или ввести `KB_HISTORY_LIMIT`?
3. **Async LLM transport**: переписать `OpenAICompatibleProvider` целиком на async, или держать sync + добавить параллельный async-класс? Второе — менее рискованно для совместимости.
4. **Token chunker default**: переводить ли весь чанкинг на токены или оставить символьный по умолчанию? Token-aware точнее, но добавляет `tiktoken` в hot path.

---

## История ревизии

- **2026-05-21** — первичный анализ 4 форков, документ создан. Источник: 4 параллельных Sonnet-агента.
