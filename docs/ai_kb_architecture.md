# KB.AI: целевая архитектура (backend + frontend)

## 1) Продуктовые контуры

- **Ingestion контур**: загрузка документов, OCR/парсинг, чанкинг, индексация в векторное хранилище.
- **Retrieval контур**: семантический поиск, rerank, фильтрация по tenant/tag/owner.
- **Generation контур**: RAG-ответы, стриминг, память диалога, цитаты и trace-id для аудита.
- **Ops контур**: наблюдаемость, health/readiness, управление пользователями, API ключами, настройками.

## 2) Backend слои

1. **API слой (FastAPI)**
   - Версионирование `/api/v1`.
   - Роуты задач генерации документов и batch-паков.
   - Роуты для Operations Console (`status`, `search`, `files`, `activities`, `admin/*`, `auth/*`).
2. **Application слой**
   - Оркестрация ingestion/search/chat операций.
   - Chat-flow вынесен в отдельный оркестратор `app/services/chat_orchestrator.py`, чтобы API-роут оставался тонким и переиспользуемым.
   - Инварианты: валидация входа, trace-id, единый формат ошибок.
3. **Domain слой**
   - Шаблоны документов, паки, версии документов.
   - Контекст генерации и политики рендера.
4. **Infrastructure слой**
   - SQLAlchemy/SQLite(Postgres-ready), S3-совместимое хранилище, Celery воркеры.
   - Qdrant/FAISS и LLM provider (`llama.cpp`/stub).

## 3) Frontend слои

1. **App Shell**: роутинг, layout, темы, i18n, auth guards.
2. **Feature pages**: Home, Dashboard, Search, Admin.
3. **Data access**: типизированный `apiClient` + слой `src/api/index.ts`.
4. **UI primitives**: таблицы, карточки метрик, таймлайн активностей, чат-панель.

## 4) Потоки данных

- **Upload → Index**: `POST /upload` → запись файла → событие в activity timeline.
- **Search**: `POST /search` → retrieval из индекса → ранжирование → выдача top-k.
- **Admin loop**: `GET/PUT /admin/settings`, `CRUD /admin/users`, ротация ключей.
- **Session loop**: `GET /auth/session`, `POST /auth/refresh`.

## 5) Нефункциональные требования

- **Надежность**: отдельные health/readiness, детерминированные fallback/stub провайдеры.
- **Масштабирование**: горизонтальные воркеры ingestion/generation, stateless API.
- **Наблюдаемость**: trace-id в middleware, единая модель ошибок.
- **Безопасность**: RBAC, API-ключи, изоляция по tenant на уровне retrieval фильтров.

## 6) Ближайший roadmap

1. Перевести runtime memory-store в Postgres + Redis для shared state между инстансами.
2. Добавить WebSocket/SSE для стриминга токенов ответа в `ChatPanel`.
3. Включить гибридный retriever (dense + keyword) и quality-metrics на offline eval.
4. Ввести feature flags для безопасного rollout экспериментальных LLM/LoRA.
