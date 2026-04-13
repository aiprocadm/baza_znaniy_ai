# Проектирование базы данных для KB.AI

## 1) Цели БД

База данных должна решать 4 задачи:

1. **Мультиарендность (multi-tenant)** — строгая изоляция данных по `tenant_id`.
2. **Надёжный ingest-конвейер** — файлы, страницы, чанки, джобы, ретраи, ошибки.
3. **Онлайн-чат и аудит** — диалоги, сообщения, источник цитат, трассировка ответов.
4. **Операционные настройки и LoRA** — конфиги, активные адаптеры, история переключений.

Рекомендованный движок для production: **PostgreSQL 16+** (JSONB, частичные индексы, CTE, `FOR UPDATE SKIP LOCKED`).

---

## 2) Предлагаемая логическая модель

Ниже — минимально-достаточная схема с разделением на домены.

### 2.1 Identity / tenancy

- `tenants`
- `users`
- `user_sessions`
- `api_keys` (опционально, для сервисных интеграций)

### 2.2 Документы и ingest

- `documents`
- `files`
- `pages`
- `chunks`
- `ingest_jobs`
- `ingest_job_events`

### 2.3 Поиск и RAG-трассировка

- `search_queries`
- `search_results`
- `answer_citations`

### 2.4 Chat / memory

- `conversations`
- `messages`
- `conversation_memory`

### 2.5 Конфигурация и ML-операции

- `settings`
- `lora_adapters`
- `lora_activation_history`

---

## 3) Ключевые таблицы (поля + ограничения)

## tenants
- `tenant_id` (PK, text/uuid)
- `slug` (unique, not null)
- `name` (not null)
- `status` (`active|inactive|suspended`)
- `contact_email`
- `storage_quota`, `storage_used`
- `document_quota`, `document_count`
- `created_at`, `updated_at`

Индексы:
- `uq_tenants_slug (slug)`
- `ix_tenants_status (status)`

## users
- `id` (PK)
- `tenant_id` (FK -> tenants.tenant_id, not null)
- `email`
- `external_id`
- `full_name`
- `role` (`owner|admin|member|viewer`)
- `status` (`active|invited|disabled`)
- `hashed_password`
- `last_login_at`, `created_at`, `updated_at`

Индексы/уникальность:
- `uq_users_tenant_email (tenant_id, email)`
- `uq_users_tenant_external (tenant_id, external_id)`

## documents
- `id` (PK)
- `tenant_id` (FK, not null)
- `slug` (nullable, tenant-scoped unique)
- `title`
- `sha256` (dedup)
- `mime_type`
- `status` (`queued|processing|indexed|failed|archived`)
- `chunks_count`
- `meta` (jsonb)
- `created_at`, `updated_at`

Индексы:
- `uq_documents_tenant_slug (tenant_id, slug)`
- `ix_documents_tenant_status (tenant_id, status)`
- `ix_documents_tenant_updated (tenant_id, updated_at desc)`

## files
- `id` (PK)
- `tenant_id` (FK)
- `document_id` (FK -> documents.id, nullable)
- `sha256`
- `filename`, `path`, `size`
- `status` (`uploaded|parsing|indexed|failed`)
- `retries`, `error`
- `created_at`, `updated_at`

Индексы:
- `uq_files_tenant_sha (tenant_id, sha256)`
- `ix_files_tenant_status (tenant_id, status)`

## pages
- `id` (PK)
- `tenant_id` (FK)
- `file_id` (FK)
- `number`
- `sha256`
- `text`
- `tokens`
- `meta` (jsonb)
- `created_at`

Индексы:
- `uq_pages_file_number (file_id, number)`
- `ix_pages_file (file_id)`

## chunks
- `id` (PK)
- `tenant_id` (FK)
- `page_id` (FK)
- `chunk_index`
- `sha256`
- `text`
- `tokens`
- `vector_backend` (`qdrant|faiss`)
- `vector_id` (id точки во внешнем vector store)
- `meta` (jsonb)
- `created_at`

Индексы:
- `uq_chunks_page_idx (page_id, chunk_index)`
- `ix_chunks_tenant_created (tenant_id, created_at desc)`
- `ix_chunks_vector (tenant_id, vector_backend, vector_id)`

## ingest_jobs
- `id` (PK)
- `tenant_id` (FK)
- `job_type` (`ingest|reindex|delete|sync`)
- `resource_type` (`document|file|tenant|chunk`)
- `resource_id`
- `status` (`queued|processing|completed|failed|cancelled`)
- `priority`
- `attempt`
- `payload` (jsonb)
- `error`
- `scheduled_at`, `started_at`, `finished_at`, `created_at`, `updated_at`

Индексы:
- `ix_jobs_queue_pick (status, priority desc, created_at asc)`
- `ix_jobs_tenant_status (tenant_id, status)`

## conversations
- `id` (PK, uuid)
- `tenant_id` (FK)
- `user_id` (FK -> users.id)
- `title`
- `status` (`active|archived`)
- `created_at`, `updated_at`, `last_message_at`

Индексы:
- `ix_conv_user_recent (tenant_id, user_id, last_message_at desc)`

## messages
- `id` (PK, bigint)
- `conversation_id` (FK)
- `tenant_id` (FK)
- `role` (`system|user|assistant|tool`)
- `content`
- `model_name`
- `prompt_tokens`, `completion_tokens`, `latency_ms`
- `meta` (jsonb)
- `created_at`

Индексы:
- `ix_messages_conv_created (conversation_id, created_at)`
- `ix_messages_tenant_created (tenant_id, created_at desc)`

## answer_citations
- `id` (PK)
- `tenant_id` (FK)
- `message_id` (FK -> messages.id)
- `document_id` (FK -> documents.id)
- `chunk_id` (FK -> chunks.id)
- `score`
- `quote_start`, `quote_end`
- `created_at`

Индексы:
- `ix_citations_message (message_id)`
- `ix_citations_doc (tenant_id, document_id)`

## settings
- `id` (PK)
- `tenant_id` (FK)
- `name` (tenant scoped unique)
- `value` (jsonb)
- `status`
- `updated_by`
- `created_at`, `updated_at`

Индексы:
- `uq_settings_tenant_name (tenant_id, name)`

## lora_adapters
- `id` (PK)
- `tenant_id` (FK)
- `name`
- `base_model`
- `adapter_type` (`peft|gguf-lora`)
- `path`
- `checksum`
- `is_active`
- `created_at`, `updated_at`

Индексы:
- `uq_lora_tenant_name (tenant_id, name)`
- `ix_lora_tenant_active (tenant_id, is_active)`

---

## 4) Важные технические решения

1. **`tenant_id` в каждой “горячей” таблице** (documents, chunks, jobs, messages) — это ускоряет фильтрацию и упрощает row-level security.
2. **JSONB только для слабо-структурированных полей** (`meta`, `payload`, `value`), а не вместо нормализации.
3. **Очередь воркера через БД**: выборка задач
   `SELECT ... FOR UPDATE SKIP LOCKED`.
4. **Мягкое удаление (soft delete)** для документов и диалогов через `deleted_at` (можно добавить вторым этапом).
5. **Аудит изменений** для настроек и LoRA-активаций (`*_history` таблицы).

---

## 5) Минимальные SQL-паттерны

### Забрать задачу воркером

```sql
WITH next_job AS (
  SELECT id
  FROM ingest_jobs
  WHERE status = 'queued'
    AND scheduled_at <= now()
  ORDER BY priority DESC, created_at ASC
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
UPDATE ingest_jobs j
SET status = 'processing', started_at = now(), updated_at = now()
FROM next_job
WHERE j.id = next_job.id
RETURNING j.*;
```

### Последние диалоги пользователя

```sql
SELECT id, title, last_message_at
FROM conversations
WHERE tenant_id = :tenant_id
  AND user_id = :user_id
  AND status = 'active'
ORDER BY last_message_at DESC
LIMIT 30;
```

---

## 6) План внедрения без большого риска

### Этап 1 (быстрый)
- Унифицировать ключ арендатора: везде `tenant_id` (с сохранением совместимости со `slug` на переходный период).
- Добавить `messages` + `conversations`.
- Добавить индексы на `jobs` и `documents` под реальные запросы.

### Этап 2
- Добавить `answer_citations` и `search_results` для наблюдаемости RAG.
- Добавить `lora_adapters` + `lora_activation_history`.

### Этап 3
- Перевести тяжёлые JSON поля на JSONB + GIN (только где есть реальные фильтры).
- Включить RLS-политики для multi-tenant production.

---

## 7) Что это даст проекту

- Стабильную работу ingest-воркеров при росте очереди.
- Нормальную диагностику “почему ответ такой” через citations и search trace.
- Удобный аудит пользовательской активности и операций с LoRA.
- Предсказуемую производительность за счёт tenant-scoped индексов.

