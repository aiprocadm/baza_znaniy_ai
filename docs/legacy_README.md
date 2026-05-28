# KB.AI

KB.AI — это многоконтейнерный сервис на FastAPI, который выполняет загрузку и
разбор документов, сохраняет чанки в Qdrant и отвечает на вопросы через локальную
модель `llama.cpp` с поддержкой LoRA-адаптеров. Репозиторий содержит всё
необходимое, чтобы развернуть API, отдельный воркер индексации, подготовить
GGUF-модель и получить осмысленный ответ с цитатами за 15–20 минут.

## Основные возможности

- современный веб-интерфейс «Operations Console» с живыми статусами, автообновлением и
  адаптивной вёрсткой для загрузки, индексации, поиска и чата;
- асинхронная индексация PDF/DOCX/TXT/XLSX/PPTX/MD с помощью `IngestService`;
- векторный поиск на Qdrant или локальном FAISS;
- генерация ответов через `llama.cpp` (по умолчанию) или детерминированную
  заглушку для офлайн-тестов;
- детальный эндпоинт `/warmup`, который прогревает провайдеры и сообщает
  длительность операций в миллисекундах;
- хранение истории диалогов и памяти чатов в каталоге `DATA_DIR`.

## MVP-режим `/api/kb/*` и `compose.yml`

Помимо зрелого многоконтейнерного стека репозиторий включает облегчённый
MVP-слой базы знаний с нейропоиском. Он рассчитан на быстрый ввод в
эксплуатацию и не требует ни мульти-tenant авторизации, ни Qdrant, ни
локальной LLM — всё, что нужно, работает «из коробки» поверх SQLite.

### Что внутри

- **Backend MVP** — модули `app/services/kb_store.py` (SQLite-стор
  документов и чанков с произвольным embedder'ом),
  `app/services/kb_embeddings.py` (pluggable эмбеддеры: Ollama,
  OpenAI-compat API, hashing fallback), `app/services/kb_llm.py`
  (DeepSeek / Groq / OpenRouter / OpenAI / Ollama / custom) и роутер
  `app/api/kb_mvp.py`, монтируемый под префиксом `/api/kb`.
- **Frontend** — статический `data/www/index.html` с разделами «Статус»,
  «Документы», «Поиск» и «Вопрос-ответ». Запросы идут на `/api/kb/*`.
- **nginx** — `data/nginx.conf` отдаёт статический фронтенд и проксирует
  `/api/` на контейнер `kb_api`.
- **Compose** — `compose.yml` поднимает три контейнера в сети `web`:
  `kb_qdrant`, `kb_api`, `kb_web` (nginx).

> **Это не обучение LLM с нуля.** MVP — собственная прикладная AI-система
> на базе RAG: чанкование → эмбеддинги → semantic search → LLM (или
> extractive fallback). Следующий этап — подключить локальную модель из
> `app/llm/llama_cpp_provider.py` или внешний API через `LLM_API_BASE_URL`,
> а в перспективе — fine-tuning при наличии данных и GPU.

### MVP endpoints

| Метод | Путь | Описание |
| ---- | ---- | -------- |
| GET | `/api/kb/health` | Health-check + статус LLM и embedder. |
| GET | `/api/kb/providers` | Снэпшот LLM-провайдеров (configured/selected/explicit). |
| POST | `/api/kb/documents` | Добавить текстовый документ `{title, text}`. |
| POST | `/api/kb/documents/upload` | Multipart-загрузка файла (PDF/DOCX/PPTX/XLSX/TXT/MD/HTML). |
| GET | `/api/kb/documents` | Список документов с признаком источника (text/file). |
| GET | `/api/kb/documents/{id}` | Получить документ с полным текстом. |
| DELETE | `/api/kb/documents/{id}` | Удалить документ и его чанки. |
| POST | `/api/kb/search` | Семантический поиск `{query, top_k}`. |
| POST | `/api/kb/ask` | RAG-ответ с цитатами `{question, top_k}`. |

### Подключение LLM-провайдеров

Все провайдеры используют один и тот же OpenAI-совместимый протокол
(`POST /v1/chat/completions`). Достаточно положить нужный ключ в `.env`
и перезапустить kb_api. Авто-приоритет: **DeepSeek → Groq → OpenRouter
→ OpenAI → custom**. Явный выбор — `KB_LLM_PROVIDER=<name>`.

| Провайдер | env (минимум) | Default model | Где получить ключ |
| --------- | ------------- | ------------- | ----------------- |
| **DeepSeek** | `DEEPSEEK_API_KEY=sk-...` | `deepseek-chat` | https://platform.deepseek.com |
| **Groq** | `GROQ_API_KEY=gsk_...` | `llama-3.3-70b-versatile` | https://console.groq.com |
| **OpenRouter** | `OPENROUTER_API_KEY=sk-or-...` | `deepseek/deepseek-chat` | https://openrouter.ai/keys |
| **OpenAI** | `OPENAI_API_KEY=sk-...` | `gpt-4o-mini` | https://platform.openai.com |
| **Ollama** | `KB_LLM_PROVIDER=ollama`, `OLLAMA_MODEL=llama3.2` | `llama3.2` | локально |
| **Custom** | `LLM_API_BASE_URL=https://...`, опц. `LLM_API_KEY` | `gpt-4o-mini` | любой OpenAI-compat |

Опциональные тюнинги для всех провайдеров: `KB_LLM_TIMEOUT` (сек),
`KB_LLM_MAX_TOKENS`, `KB_LLM_TEMPERATURE`. Для OpenRouter:
`OPENROUTER_REFERER`, `OPENROUTER_TITLE`. Для override-моделей:
`DEEPSEEK_MODEL`, `GROQ_MODEL`, `OPENROUTER_MODEL`, `OPENAI_MODEL`,
`OLLAMA_MODEL`.

Пример `.env` для DeepSeek:

```env
DEEPSEEK_API_KEY=sk-XXXXXXXXXXXXXXXX
DEEPSEEK_MODEL=deepseek-chat
KB_LLM_TIMEOUT=30
KB_LLM_TEMPERATURE=0.2
```

Пример `.env` для Groq (бесплатно, очень быстро):

```env
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXX
GROQ_MODEL=llama-3.3-70b-versatile
```

Пример `.env` для Ollama (полностью локально, без интернета):

```env
KB_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
```

Подходящий запуск Ollama добавляется в `compose.yml` дополнительным
сервисом (см. секцию «Локальная LLM через Ollama» ниже).

Активный провайдер виден в UI (chip в шапке) и через `GET /api/kb/providers`:

```bash
curl -sS http://localhost/api/kb/providers | python -m json.tool
```

### Embedding-модели

По умолчанию работает hashing-эмбеддер (быстрый, без зависимостей).
Для семантически более точного поиска переключитесь на реальный
embedder. Поддерживаются:

| Backend | env | Пример модели |
| ------- | --- | ------------- |
| Ollama (локально) | `KB_EMBEDDINGS_BACKEND=ollama`, `OLLAMA_EMBED_MODEL=nomic-embed-text` | nomic-embed-text (768-dim) |
| OpenAI-compat API | `KB_EMBEDDINGS_BACKEND=api`, `EMBEDDINGS_API_BASE_URL=...`, `EMBEDDINGS_API_KEY=...`, `EMBEDDINGS_API_MODEL=text-embedding-3-small` | text-embedding-3-small (1536-dim) |
| Hashing fallback | `KB_EMBEDDINGS_BACKEND=hash` (или ничего) | 256-dim |

**Важно про размерность**: после смены embedder'а старые чанки в БД
остаются с предыдущей размерностью и игнорируются при поиске. Чтобы
переиндексировать — удалите файл `var/data/kb_mvp.sqlite` и перезагрузите
документы. Endpoint полного reindex появится на следующем этапе.

### Prod-readiness: auth + DoS protection (B1+B2)

Спринт B1+B2 закрывает два критических риска перед деплоем на
публичный IP.

#### API key (opt-in)

По умолчанию `/api/kb/*` остаются open (как и были) — это удобно для
локальной разработки. Чтобы включить защиту, задайте в `.env`:

```env
KB_API_KEY=your-very-long-random-string-please
```

После рестарта `kb_api` все mutating endpoints требуют заголовок:

```http
POST /api/kb/documents
X-API-Key: your-very-long-random-string-please
Content-Type: application/json

{...}
```

Открытыми остаются **только** `/api/kb/health` и `/api/kb/providers`
— они нужны для healthcheck'ов (docker/nginx/k8s) и UI-инициализации.
Сравнение ключей — через `secrets.compare_digest` (constant-time,
защита от timing-атак).

Ответы при отсутствии/неправильном ключе:

```json
HTTP/1.1 401 Unauthorized
WWW-Authenticate: ApiKey realm="kb-mvp"

{"detail": "API_KEY_REQUIRED"}     // или INVALID_API_KEY
```

UI поддерживает auth «из коробки»:

* Кнопка **«Ключ»** в шапке открывает панель ввода
* Ключ сохраняется в `localStorage` только этого браузера
* Pill `auth: …` показывает состояние:
  - `auth: open` — KB_API_KEY не задан на сервере (warn)
  - `auth: key required` — сервер требует, ключа нет (error)
  - `auth: key saved` — ключ сохранён и отправляется во все запросы (ok)
* Все fetch автоматически добавляют `X-API-Key` если ключ есть

Для генерации ключа: `openssl rand -hex 32` или `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

#### DoS-защита поиска

`KnowledgeBaseStore.search` теперь применяет `LIMIT` к выборке
чанков:

```env
KB_SEARCH_HARD_LIMIT=10000   # default
```

Без этого корпус с 100K+ чанков аллоцировал бы сотни МБ Python
объектов на каждый search-запрос — тривиальный DoS-вектор. Лимит
clamp'ится в диапазон `[100, 1_000_000]`.

При попадании в лимит — `LOGGER.warning` сообщает оператору, что
пора мигрировать на Qdrant (он уже подключён в compose).

#### CORS lockdown

CORS уже конфигурируется через `Settings.cors_allow_origins`
(`app/core/app.py`). По умолчанию `["*"]` — открытый. Для prod,
если фронтенд деплоится отдельно от API:

```env
CORS_ALLOW_ORIGINS=https://kb.example.com,https://admin.kb.example.com
```

Если фронт раздаётся через тот же nginx (как в нашем `compose.yml`)
— CORS не нужен, браузер общается с тем же origin.

#### Минимальная prod-конфигурация

```env
# Безопасность
KB_API_KEY=$(openssl rand -hex 32)
CORS_ALLOW_ORIGINS=https://kb.example.com

# DoS guard (default OK для большинства случаев)
KB_SEARCH_HARD_LIMIT=10000

# LLM
DEEPSEEK_API_KEY=sk-...

# Embeddings (опционально — реальная модель)
KB_EMBEDDINGS_BACKEND=ollama
OLLAMA_EMBED_MODEL=nomic-embed-text

# Reranker (опционально — улучшение precision)
KB_RERANK_ENABLED=true
KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

### Streaming SSE (`/api/kb/ask/stream`)

Спринт 3.D добавил потоковый эндпоинт — токены приходят на клиент по
мере генерации, без ожидания полного ответа.

#### Контракт

```http
POST /api/kb/ask/stream
Content-Type: application/json

{
  "question": "Какой регламент отпусков?",
  "conversation_id": "ba02259ab52e...",
  "top_k": 4
}

Response: text/event-stream

event: meta
data: {"conversation_id":"ba02...","sources":[...],"rerank":{...}}

event: token
data: {"text":"Регламент"}

event: token
data: {"text":" отпусков"}

...

event: done
data: {"provider":"deepseek","model":"deepseek-chat","elapsed_ms":1842.3}
```

Если транспорт LLM упадёт — приходит `event: error` с
`{message: "..."}` и стрим закрывается. На пустой KB сразу
отдаётся один `event: token` с дружелюбным сообщением и
`event: done` с `provider=none`.

#### Поведение

| Сценарий | Поведение |
|----------|-----------|
| LLM настроен (DeepSeek/Groq/…) | Реальный SSE-стрим от провайдера |
| LLM не настроен | extractive-ответ как один token, `provider=extractive` |
| Legacy `state.llm_provider` | sync-вызов в threadpool, один token |
| `conversation_id` отсутствует | Создаётся новый, id в `meta` |
| `conversation_id` неизвестный | HTTP 404 |
| История подгружается | Если `history_limit > 0`, ставится в RAG-prompt |

#### Persist

Сохранение в БД происходит **после** завершения стрима — собранный
полный ответ (склейка всех `token`-чанков) идёт в `kb_messages` вместе
с user-вопросом. Источники сохраняются как JSON-snapshot.

#### Nginx буферизация

В response headers выставлен `X-Accel-Buffering: no` — без этого
nginx по умолчанию буферизует ответ полностью и streaming не работает
вживую. В `data/nginx.conf` ничего менять не нужно — header управляет
поведением сам.

#### UI

Tab «Вопрос-ответ» получил checkbox **«потоковый ответ»** (включён по
умолчанию). При отправке:

* fetch с `Accept: text/event-stream` к `/api/kb/ask/stream`
* `response.body.getReader()` + TextDecoder построчно
* Парсер `parseSseChunks` → `parseSseEvent` → диспетчеризация
  `meta` / `token` / `done` / `error`

Если потоковый режим снят — используется обычный `/ask` (как прежде).

### История диалогов (multi-turn RAG)

Спринт 3.C превратил `/ask` из stateless-эндпоинта в полноценный
chat-assistant с памятью.

#### Контракт

```http
POST /api/kb/ask
{
  "question": "Сколько дней отпуска?",
  "top_k": 4
}

Response:
{
  "answer": "...",
  "sources": [...],
  "provider": "deepseek",
  "conversation_id": "ba02259ab52e4a10b313695d92bf2c3f"
}
```

Возвращённый `conversation_id` используется в следующих запросах:

```http
POST /api/kb/ask
{
  "question": "А кому положен дополнительный?",
  "conversation_id": "ba02259ab52e4a10b313695d92bf2c3f"
}
```

При втором запросе backend подгружает последние `history_limit`
сообщений и встраивает их в RAG-промпт как «Контекст предыдущего
диалога». LLM получает не только релевантные чанки из KB, но и нить
разговора. Один и тот же диалог можно вести между сессиями.

`history_limit` (по умолчанию 10, max 50) контролирует глубину истории
в промпте. `history_limit=0` отключает контекст — каждый вопрос становится
независимым (но всё равно сохраняется в conversation).

#### Управление диалогами

| Метод | Путь | Действие |
|-------|------|----------|
| POST | `/api/kb/conversations` | Создать пустой диалог (опц. `{title}`) |
| GET | `/api/kb/conversations` | Список диалогов (порядок: most-recent-updated) |
| GET | `/api/kb/conversations/{id}` | Диалог + все сообщения |
| PATCH | `/api/kb/conversations/{id}` | Переименовать `{title}` |
| DELETE | `/api/kb/conversations/{id}` | Удалить (каскадом удаляются сообщения) |

Если `conversation_id` не передан в `/ask`, диалог создаётся
автоматически — title формируется из первой строки первого вопроса.

#### Хранение

Две новых таблицы в `kb_mvp.sqlite`:

```sql
kb_conversations(id TEXT PK, title, created_at, updated_at);
kb_messages(id, conversation_id FK CASCADE, role, content,
            sources_json, provider, model, created_at);
```

`role ∈ {"user", "assistant", "system"}`. Источники сохраняются как
JSON snapshot вместе с сообщением — даже если документ позже удалят,
исторический контекст сохранится.

#### UI

Tab «История» в `data/www/index.html`:

- Список всех диалогов с количеством сообщений и временем обновления
- Открыть → видны все сообщения с источниками
- «Продолжить →» переходит на tab «Вопрос-ответ» с активным
  `conversation_id` — следующие вопросы автоматически идут в этот
  диалог
- «Удалить» с подтверждением

### Cross-encoder reranker (опционально, выключен по умолчанию)

Спринт 3.B добавил двухэтапный retrieval поверх MVP-поиска:

1. **bi-encoder shortlist** — `KnowledgeBaseStore.search` отдаёт
   `KB_RERANK_CANDIDATES` (по умолчанию 20) кандидатов по косинусной
   близости к эмбеддингу запроса (быстро, неточно).
2. **cross-encoder rerank** — `sentence_transformers.CrossEncoder`
   попарно оценивает `(query, chunk.text)` и оставляет top-N (точно,
   медленно).

Это стандартный RAG-паттерн «возьми много, отфильтруй мало». Реально
поднимает precision@k для русско/английских корпусов.

#### Включение

```env
KB_RERANK_ENABLED=true
# Lightweight по умолчанию (~80 MB, English):
KB_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# Multilingual для русского (~600 MB, рекомендуется для корпоративной KB на русском):
# KB_RERANK_MODEL=BAAI/bge-reranker-v2-m3
KB_RERANK_CANDIDATES=20
KB_RERANK_TOPN=5
```

Модель грузится **лениво** при первом запросе с `KB_RERANK_ENABLED=true`
— на холодном старте `/search` и `/ask` ждут загрузку (10-60 сек), потом
кэш в памяти процесса.

#### Сравнение моделей

| Модель | Размер | Язык | Точность RU |
|--------|--------|------|-------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB | EN | низкая |
| `BAAI/bge-reranker-base` | ~280 MB | EN | низкая |
| `BAAI/bge-reranker-v2-m3` | ~600 MB | multi | **высокая** |
| `jinaai/jina-reranker-v2-base-multilingual` | ~280 MB | multi | средняя |

Для prod-deploy с русскими документами — `bge-reranker-v2-m3`.

#### Диагностика

`GET /api/kb/health` возвращает блок `reranker`:

```json
{
  "reranker": {
    "enabled": true,
    "model": "BAAI/bge-reranker-v2-m3",
    "candidates": 20,
    "top_n": 5,
    "loaded": true
  }
}
```

`POST /api/kb/search` и `POST /api/kb/ask` дополнительно отдают:

```json
{
  "rerank": {
    "enabled": true,
    "used": true,
    "model": "BAAI/bge-reranker-v2-m3",
    "candidates": 20,
    "elapsed_ms": 142.7
  }
}
```

В UI это отображается chip'ом `rerank: bge-reranker-v2-m3 (loaded)` в
шапке и припиской `rerank=20→5 (143ms)` в feedback после поиска.

### Парсинг файлов через Docling (включён по умолчанию)

Начиная со Спринта 3.A репозиторий **по умолчанию** включает Docling
для парсинга загружаемых файлов. `.env.example` ставит:

```env
DOCUMENT_PARSER_BACKEND=auto
DOCLING_ENABLED=true
DOCLING_TIMEOUT=180
```

Это даёт:

* **Layout-aware extraction**: таблицы как Markdown, заголовки, captions
  фигур, корректный reading order.
* **Единый pipeline** для PDF/DOCX/PPTX/XLSX/HTML/MD/TXT.
* **Безопасный fallback на legacy** при любой ошибке Docling — это
  основа `auto`-режима: zero regression risk.

Документы парсятся в Markdown через `export_to_markdown()` (приоритет
над raw text), так что таблицы и структура попадают в чанки. См.
`app/ingest/docling_backend.py:_extract_page_texts`.

Поддерживаемые форматы upload через `/api/kb/documents/upload`:
`pdf, docx, pptx, xlsx, txt, md, markdown, html, htm` (см.
`app/api/kb_mvp.py:SUPPORTED_UPLOAD_EXT`).

Если нужно отключить Docling (например, для быстрых тестов без модели
DocLayNet):

```env
DOCLING_ENABLED=false
DOCUMENT_PARSER_BACKEND=legacy
```

См. также [docs/integrations_review.md](docs/integrations_review.md) —
полный план интеграций (Docling, LangChain, reranker, history).

Все эндпоинты валидируют входные данные через Pydantic, ограничивают
размер текста (`MAX_TEXT_LEN=200_000`) и запроса (`MAX_QUERY_LEN=2_000`),
возвращают понятный JSON с детализированными сообщениями об ошибках.

### Переменные окружения

| Переменная | Значение по умолчанию | Назначение |
| ---------- | --------------------- | ---------- |
| `APP_HOST` | `kb.local` (на сервере) | Имя хоста, используемое nginx (server_name `_` ловит любой). |
| `KB_MVP_DB_PATH` | `./var/data/kb_mvp.sqlite` | Путь к локальной SQLite для MVP-стора. |
| `DATA_DIR` | `./var/data` | База путей, из неё вычисляется `KB_MVP_DB_PATH` если он пуст. |
| `QDRANT_URL` | `http://qdrant:6333` | Используется зрелым `/api/v1/*` пайплайном. MVP к Qdrant не обращается. |
| `LLM_API_BASE_URL` | — | Если задан, `/api/kb/ask` будет вызывать внешний LLM, иначе extractive fallback. |

`.env` на сервере не перезаписывается — `APP_HOST=kb.local` сохраняется.
Файл `.env.example` пополнен примером значения `KB_MVP_DB_PATH`.

### Запуск на сервере `/srv/projects/kb/`

```bash
# 0. Бэкап перед изменениями
sudo mkdir -p /srv/backups/$(date +%Y%m%d_%H%M%S)
sudo tar -czf /srv/backups/$(date +%Y%m%d_%H%M%S)/kb.tar.gz -C /srv/projects kb

# 1. Сеть web должна существовать (one-time)
docker network inspect web >/dev/null 2>&1 || docker network create web

# 2. Поднять MVP стек
cd /srv/projects/kb
docker compose -f compose.yml up -d --build

# 3. Проверить
docker ps --filter "name=kb_"
docker logs -n 200 kb_web
docker logs -n 200 kb_api
curl -sS http://localhost/api/kb/health
```

После старта браузерно открыть `http://kb.local/` (если DNS/etc-hosts
настроены) или `http://<server-ip>/`. В UI можно добавить текст, увидеть
список документов, выполнить поиск и задать вопрос.

### Быстрая проверка через curl

```bash
# Health
curl -sS http://localhost/api/kb/health

# Добавление документа
curl -sS -X POST http://localhost/api/kb/documents \
  -H "Content-Type: application/json" \
  -d '{"title":"Регламент отпусков","text":"Сотрудник имеет право на 28 календарных дней отпуска в год."}'

# Список
curl -sS http://localhost/api/kb/documents

# Поиск
curl -sS -X POST http://localhost/api/kb/search \
  -H "Content-Type: application/json" \
  -d '{"query":"сколько дней отпуска","top_k":3}'

# Ask (RAG)
curl -sS -X POST http://localhost/api/kb/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Сколько дней отпуска положено?","top_k":4}'
```

### Логи и диагностика

```bash
docker logs -n 200 kb_web      # nginx
docker logs -n 200 kb_api      # FastAPI
docker logs -n 200 kb_qdrant   # Qdrant
docker compose -f compose.yml ps
docker compose -f compose.yml config       # печать резолвнутого compose
```

### Бэкап и откат

```bash
# Бэкап перед изменениями
TS=$(date +%Y%m%d_%H%M%S); sudo mkdir -p /srv/backups/$TS
sudo tar -czf /srv/backups/$TS/kb.tar.gz -C /srv/projects kb

# Откат — выберите ваш TS из /srv/backups/
docker compose -f /srv/projects/kb/compose.yml down
sudo rm -rf /srv/projects/kb
sudo tar -xzf /srv/backups/<TS>/kb.tar.gz -C /srv/projects
docker compose -f /srv/projects/kb/compose.yml up -d --build
```

### Локальная LLM через Ollama

Чтобы запустить DeepSeek/Llama локально, добавьте в `compose.yml`
сервис рядом с `kb_api`:

```yaml
  ollama:
    image: ollama/ollama:latest
    container_name: kb_ollama
    restart: unless-stopped
    volumes:
      - ./var/data/ollama:/root/.ollama
    networks:
      - web
    # Раскомментировать, если есть GPU:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
```

После `docker compose up -d ollama` подтяните модель:

```bash
docker exec -it kb_ollama ollama pull llama3.2
# или
docker exec -it kb_ollama ollama pull deepseek-r1:7b
# для эмбеддингов:
docker exec -it kb_ollama ollama pull nomic-embed-text
```

В `.env`:

```env
KB_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2
KB_EMBEDDINGS_BACKEND=ollama
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Перезапустите `kb_api`: `docker compose restart kb_api`.

### Что дальше (этап 3)

1. **Теги/категории документов** + фильтрация в `/search` и `/ask`.
2. **История диалогов** в SQLite, endpoint `GET /api/kb/history`.
3. **JWT auth и tenants** для prod-режима (интеграция с
   `/api/v1/auth/login`).
4. **Streaming ответов** через SSE для UX.
5. **Reindex endpoint** для миграции на новую embedding-модель.
6. **Гибрид с Qdrant** для больших объёмов (>100K фрагментов).

## Operations Console UI

- Глобальный статус инфраструктуры: отображение SQLite, Vector Store, LLM и LoRA с автообновлением и
  подсказками по ошибкам.
- Метрики и активность: карточки с количеством документов, активных индексаций и ошибок, последние
  загрузки в виде ленты событий.
- Улучшенный UX: drag-and-drop загрузка, прогресс-бар, тёмная/светлая темы, тост-уведомления и
  современный чат с цитатами.
- Быстрые действия: кнопка моментального обновления, очистка чата, отправка результатов поиска в диалог
  одним кликом.


## Active runtime path (source of truth)

- **Source-of-truth backend entrypoint:** `app/api/main.py` (инициализирует `app/core/app.py:create_app`).
- **Container/CI pin:** `uvicorn app.api.main:app` используется в Docker, Compose и CI smoke job.

- **Source-of-truth runtime path:** `app/` (API, ingestion, worker, retrieval, LLM, встроенный UI).
- **`backend/` статус:** legacy/experimental контур, не является основным runtime-путём и поддерживается ограниченно.
- **UI active branch:** `frontend/` — primary web UI для продуктового сценария.
- **`app/ui` статус:** встроенная диагностическая Operations Console для runtime-проверок и отладки.

Подробнее об архитектурных решениях, в том числе про два параллельных
HTTP-пути и причины их разделения — см. [`docs/architecture.md`](docs/architecture.md).

## Архитектура

- **FastAPI + Uvicorn** — REST API (`app/api`) и статический фронтенд (`app/ui`).
- **IngestService** (`app/ingest`) — разбивает документы на чанки и отправляет их
  в очередь индексации.
- **Ingest worker** (`app/worker/main.py`) — отдельный процесс, который подбирает
  задания из базы и индексирует документы.
- **Vector store** (`app/retriever`) — Qdrant или FAISS, выбирается через
  `VECTOR_BACKEND`.
- **LLM-провайдер** (`app/llm`) — новый `LlamaCppProvider` на базе
  `llama_cpp.Llama`, умеет подключать/отключать LoRA и отслеживать активный
  адаптер.
- **Reranker** (`app/retriever.rerank`) — опциональный слой для повторного
  упорядочивания поисковой выдачи.
- **Docker Compose** (`docker-compose.yml`) — поднимает `kb_api`, `kb_worker` и
  `qdrant` с общими томами для моделей и данных.

## Кодстайл API

Рекомендации по обработчикам REST-слоя собраны в документе
[`docx/api_code_style.md`](docx/api_code_style.md). Ключевые идеи:

- входные Pydantic-модели и объекты файловых загрузок остаются неизменными;
- нормализация данных выполняется через копии (`model_copy`) или отдельные DTO;
- обработчики возвращают строгие ошибки с короткими машинными кодами.

## Требования

- Python 3.12+
- Git
- Docker (опционально для контейнерного запуска)

## Быстрый старт (10 минут)

```bash
git clone https://github.com/<org>/kb_ai.git
cd kb_ai
python3.12 -m venv .venv
source .venv/bin/activate
make install
make lint
make test
```

После этого можно запускать dev-сервер:

```bash
make run
```

Запустить воркер индексации отдельно можно командой:

```bash
make worker
```

### Lightweight dev server (MVP only)

Если не нужен полный multi-tenant стек, можно запустить только MVP-роутер
`/api/kb/*` с минимальными зависимостями:

```bash
python -m uvicorn scripts.dev_server_mvp:app --reload --port 8001
```

Это удобно для UI-разработки и smoke-тестов без Qdrant/llama-cpp/sentence-transformers.

## Контейнерный запуск

В репозитории подготовлен многоконтейнерный стек:

- `kb_api` — FastAPI + веб-интерфейс.
- `kb_worker` — ingest-воркер, который обрабатывает очередь задач.
- `qdrant` — внешнее хранилище векторных эмбеддингов.

Для запуска достаточно выполнить:

```bash
docker compose up --build
```

API будет доступно на `http://localhost:8000`, Qdrant — на `http://localhost:6333`.
Тома `./var/data` и `./models` автоматически монтируются в оба сервиса.

## Установка в Codespaces/на VPS без GPU

В средах без GPU рекомендуется сразу ставить CPU-сборку PyTorch, а уже затем остальные пакеты:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip cache purge
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
pip install -r requirements-runtime.txt -r requirements-llm.txt
pip install -r requirements-dev.txt
```

## Установка зависимостей

Зависимости разделены по сценариям использования:

- `requirements-runtime.txt` — API, воркер и пайплайны индексации (без Torch и PEFT).
- `requirements-llm.txt` — локальный инференс (`llama.cpp`, эмбеддинги и FAISS).
- `requirements-train.txt` — обучение LoRA/QLoRA (Accelerate, PEFT, Transformers).
- `requirements-dev.txt` — линтеры, тесты и вспомогательные тулзы.

Можно установить их по отдельности либо через extras из `pyproject.toml`:

```bash
pip install .[runtime,llm]
pip install .[train]  # при необходимости обучения
pip install .[dev]
```

Команды `make install` и `make dev` выполняют те же шаги: первая ставит CPU PyTorch вместе с runtime+llm зависимостями, вторая добавляет dev-набор.

## Makefile и автоматизация

Для повседневных задач предусмотрен `Makefile`:

| Команда | Действие |
| --- | --- |
| `make venv` | Создаёт виртуальное окружение (`.venv`). |
| `make install` | CPU-режим: PyTorch (CPU) + runtime + llm зависимости. |
| `make dev` | Добавляет dev-зависимости поверх `make install`. |
| `make lint` | Запускает `ruff` и `black --check`. |
| `make format` | Применяет автоформатирование `black` и `ruff --fix`. |
| `make test` | Выполняет `pytest -q`. |
| `make run` | Стартует FastAPI через `uvicorn --factory`. |
| `make worker` | Запускает ingest-воркер локально (без Docker). |
| `make build` | Собирает локальный Docker-образ `kb-ai:local`. |
| `make clean` | Удаляет артефакты тестов и кеши (`__pycache__`, coverage). |

Команды кроссплатформенные и работают как в Linux, так и в WSL/PowerShell через `make`.

Dev-зависимости включают парсеры `openpyxl` и `python-pptx`, которые нужны для
юнит-тестов обработки XLSX и PPTX-документов.

## Как обучить свой LoRA-адаптер

### Автогенерация датасета через teacher-LLM

Вместо ручного составления `data/dev.jsonl` можно сгенерировать обучающие
пары вопрос-ответ автоматически из уже загруженного KB-корпуса:

```bash
# DeepSeek (дешёвый teacher, ~$0.50 за 1000 Q&A)
export DEEPSEEK_API_KEY=sk-...
python -m scripts.generate_synthetic_qa \
    --corpus var/data/kb_mvp.sqlite \
    --provider deepseek \
    --mode single \
    --output data/lora/synthetic.jsonl \
    --max-budget-usd 2.0
```

Поддерживаемые режимы (`--mode`):

| Режим | Описание |
|------|----------|
| `single` | Один Q&A на чанк (быстро, минимально) |
| `paraphrase` | Три перефразирования одного вопроса (аугментация) |
| `multi-hop` | Вопрос, требующий объединения 2-3 чанков (сложнее) |

Полезные флаги:

- `--resume` — продолжить с того места, где остановилась прошлая
  запуск (читает уже записанный JSONL, пропускает обработанные чанки).
- `--document-id N` — генерировать только по одному документу.
- `--no-self-consistency` — отключить проверку повторной генерации
  (быстрее, но качество ниже).
- `--no-budget-guard` — снять ограничение по стоимости (use with care).

Сгенерированный JSONL совместим с `scripts/validate_dataset.py` и
`scripts/train_lora.py` без преобразований:

```bash
python scripts/validate_dataset.py \
    --path data/lora/synthetic.jsonl \
    --base-model meta-llama/Llama-3-8b-Instruct
```

### RAG-aware датасет (W3)

Сами по себе W1-сиды учат модель отвечать на вопросы по содержимому чанков,
но **не** учат пользоваться извлечённым контекстом и не учат отказу при
нерелевантном retrieval'е. Workstream 3 надстраивает над W1 четырёхвариантную
смесь (relevant / irrelevant / partial / empty в пропорции 70 / 15 / 10 / 5):

```bash
python -m scripts.generate_rag_dataset \
    --corpus var/data/kb_mvp.sqlite \
    --seeds data/lora/synthetic.jsonl \
    --output data/lora/train_rag.jsonl \
    --target-pairs 1000
```

При обучении укажите `--prompt-mode rag`, чтобы трейнер использовал
`PROMPT_TEMPLATE_RAG` (система + контекст + вопрос) и читал поле
`retrieved_context` из датасета:

```bash
python scripts/train_lora.py \
    --base-model TheBloke/Llama-3-8B-Instruct-AWQ \
    --train data/lora/train_rag.jsonl \
    --output adapters/my-rag-lora \
    --prompt-mode rag
```

Полезные флаги `generate_rag_dataset`:

- `--top-k N` — сколько чанков извлекать на вопрос (по умолчанию 3).
- `--negative-document-id N` — брать чанки для IRRELEVANT/PARTIAL пула
  из конкретного документа (иначе случайная выборка).
- `--resume` — пропустить сиды, чьи `source_chunk_id` уже в output JSONL.

Ниже — минимальный путь от данных до подключенного адаптера. Все команды
рассчитаны на Linux/macOS; в PowerShell используйте эквиваленты.

1. **Подготовьте датасет.** Каждый пример — строка JSON с полями
   `instruction`, `input` (можно оставить пустым) и `output`:

   ```json
   {"instruction": "Объясни правило сложения", "input": "2+2", "output": "2+2=4"}
   ```

   Пример см. в [`data/dev.jsonl`](data/dev.jsonl). Чем чище ответы, тем выше
   метрики EM/ROUGE.

2. **Проверьте качество данных.** Скрипт ищет дубликаты, пустые поля и статистику
   токенов:

   ```bash
   python scripts/validate_dataset.py \
     --path data/dev.jsonl \
     --base-model "${LORA_TRAIN_BASE_MODEL:-sshleifer/tiny-gpt2}" \
     --max-seq-len "${LORA_TRAIN_MAX_SEQ_LEN:-4096}"
   ```

   Отчёты сохраняются в JSON и Markdown; при критичных проблемах код возврата >0.

3. **Запустите обучение.** Минимальный пример c tiny-моделью (для быстрой
   проверки) и отключённым QLoRA:

   ```bash
   export LORA_USE_QLORA=0
   python scripts/train_lora.py \
     --base-model sshleifer/tiny-gpt2 \
     --train data/lora/smoke_train.jsonl \
     --eval data/lora/smoke_eval.jsonl \
     --output "$LORA_TRAIN_OUTPUT_DIR" \
     --max-seq-len 512 \
     --epochs 1 \
     --lr 2e-4 \
     --batch-size 1 \
     --gradient-accumulation 4 \
     --lora-r 16 \
     --lora-alpha 32 \
     --lora-dropout 0.05 \
     --target-modules c_attn,c_proj \
     --no-qlora
   ```

   Скрипт сохранит артефакты в каталоге `runs/<timestamp>_*`: `adapter/`,
   `metrics.json`, `trainer_state.json`, `tokenizer/` и журнал `logs/training.jsonl`.

4. **Оцените адаптер.** Получите EM/ROUGE-L на валидации и проверьте пороги:

   ```bash
  python scripts/eval_lora.py \
    --base-model sshleifer/tiny-gpt2 \
    --adapter runs/<...>/adapter/adapter.safetensors \
     --dataset data/lora/smoke_eval.jsonl \
     --max-new-tokens 128 \
     --min-em 0.2 \
     --min-rouge 0.2
   ```

5. **(Опционально) конвертируйте в GGUF для llama.cpp.**

   ```bash
  python scripts/convert_lora_to_gguf.py \
    --base-model ./models/model.gguf \
    --adapter runs/<...>/adapter/adapter.safetensors \
     --out ./data/lora/registry/my-adapter/adapter.gguf
   ```

6. **Зарегистрируйте адаптер.** В каталоге `LORA_REGISTRY_DIR` создайте папку с
   файлами и манифестом:

   ```bash
   mkdir -p data/lora/registry/demo
  cp runs/<...>/adapter/adapter.safetensors data/lora/registry/demo/
   cat > data/lora/registry/demo/manifest.json <<'JSON'
   {"name":"demo","base":"meta-llama/Llama-3-8b-Instruct","type":"peft","seq_len":4096,"created_at":"2024-01-01T00:00:00Z"}
   JSON
   ```

7. **Горячо подключите адаптер через API.**

   ```bash
   curl -X POST http://localhost:8000/admin/lora/load \
     -H "X-API-Key: <ключ>" \
     -H "Content-Type: application/json" \
     -d '{"name":"demo"}'

   curl http://localhost:8000/admin/lora/list -H "X-API-Key: <ключ>"
   curl -X POST http://localhost:8000/admin/lora/unload -H "X-API-Key: <ключ>" -d '{"name":"demo"}'
   ```

Типовые ошибки:

- «Adapter ... not found» — неправильное имя каталога или отсутствует `manifest.json`.
- «does not support GGUF adapters» — провайдер LLM не умеет подключать данный формат (нужно конвертировать).
- «token count exceeds max_seq_len» — пересмотрите `LORA_TRAIN_MAX_SEQ_LEN` или обрежьте источник.

### Минимальный набор для pytest

Если требуется лишь прогнать тесты без полного окружения, заранее установите
критичные бинарные колёса (объём скачивания может превышать 2 ГБ и занимать до
10 минут на средней сети):

- `numpy` — базовые численные операции для пайплайна эмбеддингов;
- `faiss-cpu` — локальный бэкенд векторного поиска;
- `sentence-transformers` — генерация эмбеддингов для тестов;
- `llama-cpp-python` — локальный LLM-провайдер.
- `openpyxl` — чтение XLSX-таблиц в пайплайне индексации;
- `python-pptx` — парсинг презентаций PowerPoint.

## Миграции базы данных

Сервис использует Alembic для управления схемой SQLite. После установки
зависимостей выполните миграции (каталог `var/data` будет создан автоматически):

```bash
mkdir -p var/data
alembic upgrade head
```

По умолчанию Alembic использует `DB_URL` из переменных окружения или значение из
`alembic.ini`. При необходимости можно переопределить путь к базе данных:

```bash
DB_URL="sqlite+aiosqlite:///./var/data/custom.sqlite" alembic upgrade head
```

## Конфигурация и ключевые переменные окружения

Сервис полностью настраивается через переменные окружения. Шаблон `.env`
лежит в корне репозитория — скопируйте его и заполните нужные поля перед
запуском:

```bash
cp .env.example .env
```

По умолчанию приложение читает переменные из `.env` и падает обратно на
встроенные значения из `app/core/config.py`. Таблицы ниже показывают эти
значения по умолчанию и помогают подобрать настройки под инфраструктуру.

### Основные параметры

| Переменная | Значение по умолчанию | Описание |
| --- | --- | --- |
| `APP_ENV` | `development` | Активный профиль приложения (например, `production`). |
| `APP_HOST` | `0.0.0.0` | Хост для HTTP-сервера FastAPI. |
| `APP_PORT` | `8000` | Порт HTTP-сервера. |
| `LOG_LEVEL` | `INFO` | Глобальный уровень логирования. |
| `RATE_LIMIT` | пусто | Лимит запросов в формате `<кол-во>/<интервал>`, например `100/1m`. |
| `RATE_BURST` | `0` | Допустимый всплеск запросов сверх основного лимита. |
| `DATA_DIR` | `./var/data` | Корневой каталог данных; внутри создаются `files/`, `db/`, `logs/`. |
| `FILES_SUBDIR` | `files` | Подкаталог в `DATA_DIR` для загруженных файлов. |
| `DB_URL` | `sqlite+aiosqlite:///./var/data/kb.sqlite` | Строка подключения для БД метаданных и прогресса индексации. |
| `MAX_UPLOAD_MB` | `40` | Максимальный размер одного загружаемого файла в мегабайтах. |
| `UPLOAD_ALLOWED_EXTS` | `pdf,docx,pptx,xlsx,txt,md` | Список разрешённых расширений через запятую. |
| `CORS_ALLOW_ORIGINS` | `*` | Допустимые origin-ы для API (поддерживает список через запятую). |
| `INGEST_MAX_RETRIES` | `3` | Количество повторных попыток при ошибке индексации. |
| `INGEST_BACKOFF_SECONDS` | `1.0` | Базовая задержка (в секундах) между повторами индексации. |

### Векторное хранилище

| Переменная | Значение по умолчанию | Описание |
| --- | --- | --- |
| `VECTOR_BACKEND` | `qdrant` | Тип стора: `qdrant` или `faiss`. |
| `VECTOR_EMBED_MODEL` | `intfloat/multilingual-e5-small` | Модель эмбеддингов для расчёта векторов. |
| `VECTOR_EMBED_DIMENSION` | `384` | Размерность эмбеддингов (нужно для внешних стораджей). |
| `EMBED_BATCH_SIZE` | `64` | Размер батча при генерации эмбеддингов. |
| `RETRIEVE_TOPK` | `10` | Количество документов, извлекаемых из стора. |

| `RERANK_ENABLED` | `true` | Включить повторное ранжирование кандидатов. |

| `RERANK_ENABLED` | `true` | Повторное ранжирование кандидатов (отключите, установив `false`). |

| `RERANK_TOPK` | `50` | Ограничение результатов после повторного ранжирования. |
| `RAG_TOKENIZER_NAME` | `cl100k_base` | Токенизатор для разбивки текста на чанки. |
| `RAG_CHUNK` | `900` | Размер чанка в токенах. |
| `RAG_OVERLAP` | `140` | Перекрытие соседних чанков. |
| `QDRANT_URL` | пусто | HTTP-эндпоинт внешнего Qdrant; оставьте пустым для встроенного режима. |
| `QDRANT_PATH` | `<DATA_DIR>/qdrant` | Каталог для встроенного Qdrant (указывается автоматически при пустом значении). |
| `QDRANT_COLLECTION` | `kb_chunks` | Название коллекции с документами. |
| `QDRANT_API_KEY` | пусто | Ключ доступа к управляемому Qdrant. |

### LLM

| Переменная | Значение по умолчанию | Описание |
| --- | --- | --- |
| `LLM_PROVIDER` | `llama-cpp` | Провайдер языковой модели. |
| `LLM_MODEL_NAME` | `kb-llama` | Отображаемое имя модели в логах и ответах. |
| `LLM_MODEL_VERSION` | пусто | Произвольный тег версии модели. |
| `LLM_MODEL_PATH` | `./models/model.gguf` | Путь к файлу GGUF. Файл должен существовать. |
| `LLM_CTX` | `4096` | Размер контекстного окна. |
| `LLM_THREADS` | `4` | Количество потоков CPU для инференса. |
| `LLM_GPU_LAYERS` | `0` | Число слоёв, переносимых в GPU (`0` — только CPU). |
| `LLM_TEMPERATURE` | `0.7` | Температура сэмплирования. |
| `LLM_TOP_P` | `0.95` | Порог nucleus sampling. |
| `LLM_TOP_K` | `40` | Ограничение словаря при генерации. |
| `LLM_MAX_TOKENS` | `1024` | Лимит выходных токенов. |
| `LLM_API_BASE_URL` | пусто | Базовый URL OpenAI-совместимого API (используется при `LLM_PROVIDER=api`). |
| `LLM_API_KEY` | пусто | Ключ доступа для внешнего API. |
| `LLM_API_MODEL` | `gpt-4o-mini` | Имя модели во внешнем API. |
| `LLM_API_TIMEOUT_SEC` | `60` | Таймаут запроса к внешнему API в секундах. |
| `LLM_LORA_ADAPTER` | пусто | Имя активного адаптера (используется вместе с Ollama). |
| `LORA_ADAPTER_PATH` | пусто | Файл адаптера для `llama.cpp`. |
| `LORA_SCALING` | `1.0` | Коэффициент смешивания LoRA. |
| `LORA_ADAPTER_VERSION` | пусто | Тег версии подключённого адаптера. |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Эндпоинт Ollama при использовании удалённого инференса. |

### Чат и память

| Переменная | Значение по умолчанию | Описание |
| --- | --- | --- |
| `CHAT_DB_BACKEND` | `sqlite` | Тип стора истории (`sqlite` или `postgres`). |
| `CHAT_DB_PATH` | `<DATA_DIR>/db/chat_history.sqlite` | Путь к SQLite-файлу истории (заполняется автоматически). |
| `CHAT_DB_DSN` | пусто | Строка подключения PostgreSQL при выборе `postgres`. |
| `CHAT_DB_SCHEMA` | пусто | Необязательная схема PostgreSQL. |
| `CHAT_HISTORY_LIMIT` | `12` | Количество сообщений, попадающих в краткосрочный контекст. |
| `CHAT_SUMMARY_TRIGGER` | `10` | Порог сообщений для автосаммаризации. |
| `CHAT_MIN_CITATIONS` | `3` | Минимум источников в ответе. |
| `CHAT_MAX_CITATIONS` | `5` | Максимум источников в ответе. |
| `CHAT_MEMORY_ENABLED` | `false` | Включить долговременную память. |
| `MEMORY_DB_PATH` | `<DATA_DIR>/db/memory.sqlite` | Путь к базе памяти (подставляется автоматически). |
| `CHAT_MEMORY_TTL_DAYS` | `90` | Время хранения памяти в днях. |
| `CHAT_MEMORY_MAXTOK` | `2000` | Верхняя граница «токенов» памяти. |

### Безопасность

| Переменная | Значение по умолчанию | Описание |
| --- | --- | --- |
| `SECRET_KEY` | `change-me` | Секрет для подписи JWT и cookies. |
| `JWT_ALGORITHM` | `HS256` | Алгоритм подписи токенов. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Срок жизни access-токена в минутах. |

> Совет: переменные с путями (`*_PATH`) могут задаваться относительными
> значениями — при пустом поле сервис использует пути внутри `DATA_DIR`.

Перед запуском убедитесь, что `LLM_MODEL_PATH` указывает на существующий GGUF.
Без модели чатовые эндпоинты возвращают `503 Service Unavailable`.

## Подготовка модели GGUF и LoRA

### Базовая модель TinyLlama 1.1B Chat

По умолчанию сервис использует **TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf** из
репозитория [bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF)
на Hugging Face. Модель распространяется по лицензии **Apache-2.0** и подходит
для коммерческого использования при сохранении уведомлений об авторстве.

1. Убедитесь, что активировано виртуальное окружение и установлены зависимости.
2. Выполните команду загрузки (скачивание ~625 МБ):

   ```bash
   python -m scripts.download_model --allow-missing-hash --max-retries 5
   ```

   Скрипт автоматически:

   - создаст каталог `models/`,
   - загрузит файл `model.gguf`,
   - проверит контрольную сумму SHA-256 (берётся из `model_manifest.json` или
     метаданных Hugging Face),
   - обновит `models/model_manifest.json`, записав фактический хэш для повторных запусков.

   При необходимости авторизуйтесь на Hugging Face, указав токен в переменной
   `HUGGINGFACE_HUB_TOKEN` (см. `.env.example`).

3. Проверьте файл вручную при желании:

   ```bash
   sha256sum models/model.gguf
   ```

4. После загрузки можно стартовать сервис или контейнеры — шаги `docker compose`
   и Dockerfile автоматически вызывают тот же скрипт и пропускают скачивание,
   если хэш уже совпадает.

Эндпоинт `/health` проверяет только то, что приложение запущено. Для полной
проверки зависимостей используйте `/ready` — он валидирует соединение с SQLite,
состояние векторного стора, доступность LLM и активный LoRA-адаптер. Метрики
Prometheus доступны по `/metrics` и включают показатели парсинга, OCR,
индексации, поиска и чата.

### Мониторинг `SQLModel.metadata`

- Гейдж `kb_sqlmodel_metadata_health{origin="..."}` устанавливается в `1`, когда
  `SQLModel.metadata` присутствует и содержит все таблицы моделей, и `0`, если
  объект потерян или повреждён.
- Счётчик `kb_sqlmodel_metadata_alerts_total{origin="...",reason="..."}`
  увеличивается при каждом срабатывании фонового стража
  `sqlmodel-metadata-guard` (интервал 15 секунд) или при неуспешной попытке
  инициализации схемы.

**Как реагировать на алерт:**

1. Проверить логи приложения (`tail -f var/logs/app.log` или `docker compose logs`)
   на запись `SQLModel metadata integrity check failed` и уточнить причину в поле
   `reason`.
2. Убедиться, что SQLite-файл доступен и не повреждён: `sqlite3 var/data/kb.sqlite ".tables"`.
3. Если таблицы отсутствуют, повторно применить миграции `alembic upgrade head`
   и перезапустить сервис (`docker compose restart kb_web` или `make run`).
4. После восстановления состояния выполнить `curl -s http://localhost:8000/metrics`
   и убедиться, что `kb_sqlmodel_metadata_health` снова равен `1`.

```bash
docker compose exec kb_web curl -s http://localhost:8000/ready
```

### Альтернативные модели

Если требуется другая модель:

1. Скачайте оригинальный чекпоинт с Hugging Face или другого источника.
2. Конвертируйте его в GGUF (пример — квантование в Q4_K_M):

   ```bash
   python -m llama_cpp.convert --quantize q4_k_m --outfile ./models/model.gguf \
       path/to/hf/model
   ```

3. Обновите `models/model_manifest.json`, указав ссылку на источник и хэш
   (можно получить командой `sha256sum`).
4. Перезапустите `python -m scripts.download_model` — он проверит контрольную
   сумму и перезапишет файл, если локальная копия отличается.

5. Откройте веб-интерфейс `http://<сервер>:8000` и авторизуйтесь с учётными данными из `.env`.

## Веб-интерфейс

По адресу `/` доступна статическая страница «KnowLab Operations Console». Она не
требует сборки и работает поверх REST API.

- Блок «Загрузка документа» отправляет файлы на `/api/v1/upload`, показывает
  прогресс загрузки и автоматически инициирует индексацию через `/api/v1/ingest`.
  Статусы обновляются по API `/api/v1/files` с периодическим опросом.
- Поисковая форма выполняет запросы к `/api/v1/search` и показывает
  предпросмотр чанков.
- Чат общается с `/api/v1/chat`, отображает ответы ассистента и ссылки на
  цитаты. При клике по цитате выполняется поиск по связанному документу.
- В выпадающем поле можно указать значение заголовка `X-Tenant`; если оно не
  задано, используется стандартный tenant.

Страница корректно обрабатывает ситуации, когда сервис временно недоступен или
возвращает неполные данные: сообщения об ошибках отображаются прямо в интерфейсе.

## API управления LoRA

LoRA-адаптеры можно подгружать и выгружать через REST без перезапуска сервиса.
При успешной загрузке состояние отображается в эндпоинте `/ready`. Примеры запросов:

```bash
curl -X POST http://localhost:8000/api/v1/lora/load \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/adapter.gguf","scaling":0.85}'
```

```bash
curl -X POST http://localhost:8000/api/v1/lora/unload \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/adapter.gguf","scaling":1.0}'
```

Текущее состояние адаптера можно проверить:

```bash
curl -s http://localhost:8000/ready
```

Ответ содержит раздел `details` со статусами всех ключевых подсистем:

```json
{
  "status": "ok",
  "details": {
    "sqlite": {"status": "ok"},
    "vector_store": {"status": "ok"},
    "llm": {"status": "ok", "model": "ok"},
    "lora": {"status": "ok", "detail": {"loaded": false}}
  }
}
```

### Пайплайн ассетов

Front-end состоит из одного HTML-файла с нативным JavaScript и встроенными
стилями. Дополнительные сборщики (Webpack, Vite, Parcel и т.д.) не используются
и не требуются.

### Переключение провайдера и моделей

По умолчанию сервис использует `llama-cpp` с локальной GGUF-моделью, путь к
которой задаётся переменной `LLM_MODEL_PATH`. Чтобы переключиться на другую
модель, обновите путь и при необходимости параметры инициализации (`LLM_CTX`,
`LLM_THREADS`, `LLM_GPU_LAYERS`). Если файл отсутствует или повреждён, чатовые
эндпоинты возвращают `503 Service Unavailable` с кодом `LLM_MODEL_MISSING`.

Дополнительно можно ограничить размер вводного контекста (`LLM_CTX`) и объём
генерации (`LLM_MAX_TOKENS`).

## Логирование и память

## Обучение и развёртывание LoRA адаптеров

### Экспорт датасета из индекса

1. Активируйте виртуальное окружение и загрузите переменные из `.env`, чтобы скрипты получили доступ к Qdrant и базе данных.
2. Выполните `python -m scripts.export_all ./var/backups/index.tar.gz`. Архив содержит `vector_payloads.json`, в котором для каждого чанка хранится исходный текст и дополнительные поля.
3. Сформируйте тренировочный набор с колонками `question`, `context`, `answer`. Для простого чата можно использовать заголовок чанка в качестве вопроса, сам текст в поле контекста и ожидаемый ответ (либо пустую строку) — сохраните файл в формате CSV или JSONL.

### Аппаратные требования

- **CUDA**. Для обучения на GPU требуется установленный драйвер NVIDIA и CUDA 12.x. После проверки `nvidia-smi` можно дополнительно поставить `bitsandbytes`, чтобы активировать 4-битный режим.
- **Минимум VRAM**. Для моделей до 8B параметров достаточно 12 ГБ видеопамяти (QLoRA с градиентным накоплением). Для 13B потребуется 24 ГБ. На CPU обучение тоже возможно, но займёт часы/сутки и потребует 32+ ГБ RAM.

### Обучение QLoRA на CPU или GPU

1. Установите зависимости для обучения:

   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
   pip install -r requirements-train.txt
   # или через extras: pip install .[train]
   ```

   При наличии CUDA дополнительно выполните `pip install bitsandbytes` и убедитесь, что `nvidia-smi` отображает устройство.
2. Запустите обучение:
   ```bash
   python -m scripts.train_lora \
       ./data/dataset.jsonl \
       meta-llama/Meta-Llama-3-8B-Instruct \
       ./var/adapters/my-llama \
       --lora-r 8 \
       --lora-alpha 16 \
       --lora-dropout 0.05 \
       --num-epochs 3 \
       --per-device-train-batch-size 2 \
       --gradient-accumulation-steps 4
   ```
   Скрипт подхватывает токены и прокси из `.env`, логирует процесс обучения и сохраняет LoRA-адаптер в подкаталоге `adapter/`.

### Конвертация в форматы llama.cpp

- После обучения в каталоге `ggml/` автоматически появится файл `adapter.ggml`, совместимый с `llama.cpp` (скрипт вызывает `llama_cpp.convert_lora`).
- Для генерации GGUF можно выполнить `python -m llama_cpp.convert_lora --to-gguf --base-model <модель> --adapter ./var/adapters/my-llama/adapter --output ./var/adapters/my-llama/gguf/my-llama.gguf`.

### Размещение и горячая загрузка

1. Скопируйте каталоги `adapter/` и `ggml/` в `./var/adapters/<имя_адаптера>/`.
2. Вызовите новый эндпоинт `POST /api/v1/llm/adapters/hot-load` с JSON `{ "name": "<имя_адаптера>" }`, чтобы перезагрузить адаптер без рестарта сервиса. Эндпоинт ищет файлы в `./var/adapters/` и подключает их для текущей LLM.

### Переключение провайдера и модели LLM

История диалогов по умолчанию сохраняется в SQLite-файле по пути `DATA_DIR/db/chat_history.sqlite`. При необходимости можно переключить приложение на PostgreSQL, задав `CHAT_DB_BACKEND=postgres` и передав строку подключения через `CHAT_DB_DSN`. Для каждого сообщения хранится пользователь, идентификатор диалога, роли (`user`/`assistant`) и содержание. Эти данные используются для восстановления контекста между запросами.

Логи приложения пишутся как в stdout, так и в файл `DATA_DIR/logs/app.log`. Каталоги `DATA_DIR` и вложенные директории создаются автоматически при старте сервиса.

Метаданные загрузок и чанков по умолчанию сохраняются в SQLite-базе `./var/data/kb.sqlite` (см. `DB_URL`). Путь можно поменять, указав собственный DSN в переменной окружения.

Приложение по умолчанию использует `llama-cpp` и локальный GGUF-файл. Настроить
поведение можно с помощью переменных окружения:

- `LLM_MODEL_PATH` — путь к файлу модели. Если файл отсутствует, чатовые
  эндпоинты возвращают `503` с ошибкой `LLM_MODEL_MISSING`.
- `LLM_CTX`, `LLM_THREADS`, `LLM_GPU_LAYERS` — параметры инициализации модели.
- `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K` — ограничения и
  параметры генерации ответов.

## Логирование и память

История диалогов по умолчанию сохраняется в SQLite-файле по пути `DATA_DIR/db/chat_history.sqlite`. При необходимости можно переключить приложение на PostgreSQL, задав `CHAT_DB_BACKEND=postgres` и передав строку подключения через `CHAT_DB_DSN`. Для каждого сообщения хранится пользователь, идентификатор диалога, роли (`user`/`assistant`) и содержание. Эти данные используются для восстановления контекста между запросами.

Логи приложения пишутся как в stdout, так и в файл `DATA_DIR/logs/app.log`. Каталоги `DATA_DIR` и подкаталоги `logs` создаются автоматически при старте сервиса.

3. Укажите полученный путь в переменной `LLM_MODEL_PATH`.
        main

4. Чтобы подключить LoRA, разместите адаптер в доступном каталоге и задайте
   переменные `LORA_ADAPTER_PATH=/abs/path/to/adapter.safetensors` и
   `LORA_SCALING=0.5` (значение по умолчанию — `1.0`). Адаптер можно получить из
   PEFT-проекта, совместимого с `llama.cpp`.

## Запуск сервиса

Перед первым запуском убедитесь, что модель скачана и проверена:

```bash
python -m scripts.download_model --allow-missing-hash --max-retries 5
```

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

При старте приложение:

- инициализирует хранилище чатов и, при необходимости, долговременную память;
- поднимает очередь ingestion и воркер для разбора файлов;
- подготавливает LLM-провайдера, в том числе прогружает LoRA-адаптер, если он
  указан в настройках.

Веб-интерфейс доступен по адресу `http://<host>:8000/`. Панель «Operations
Console» позволяет загружать документы, смотреть прогресс индексации, выполнять
поиск и вести диалог в рамках одного окна.

## Примеры запросов

Загрузка документа:

```bash
curl -F "file=@./docs/manual.pdf" http://localhost:8000/api/v1/upload
```

RAG-чат (с указанием пользователя и необязательного `conversation_id`):

```bash
curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": "alice",
           "message": "Как настроить репликацию?",
           "conversation_id": null,
           "top_k": 8
         }'
```

Ответ содержит финальный текст, источники (файл/страница) и метрики генерации.

## Тестирование

Перед запуском `pytest` рекомендуем заранее подтянуть бинарные сборки тяжёлых
зависимостей, чтобы исключить компиляцию на лету:

```bash
pip install --only-binary=:all: numpy faiss-cpu sentence-transformers llama-cpp-python
pytest
```

Тесты покрывают фабрику LLM-провайдера, загрузку моделей, поведение чата в
условиях отсутствующей модели и основные сценарии пайплайна.


## Legacy compatibility path (`backend/app/*`)

- Путь `backend/app/*` помечен как **legacy/compatibility only**.
- Точка входа legacy-контура: `backend/app/main.py`.
- Запуск legacy-контура в CI разрешён только в job `legacy-compatibility-tests` (контрактные/совместимые тесты).
- Smoke/основные тесты всегда проверяют только active path `app/api/main.py` + `app/core/app.py`.
