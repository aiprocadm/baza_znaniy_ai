# KnowLab MVP

KnowLab — база знаний с Retrieval-Augmented Generation. Репозиторий содержит минимально жизнеспособную реализацию: загрузка документов, индексирование, чат с цитатами и развёртывание в Docker. Авторизация реализована через basic auth на стороне nginx, а само приложение не требует JWT и ролей.

## Цели и функциональность

- Загрузка документов (DOCX, PDF с текстом, TXT) через веб-интерфейс.
- Извлечение текста, разбиение на чанки по 900 токенов с overlap 140 и сохранение в Qdrant с эмбеддингами `intfloat/multilingual-e5-small`.
- Чат с моделью `qwen2.5:3b-instruct` (Ollama) и возвратом цитат.
- Запоминание истории диалогов в локальной SQLite-базе (`DATA_DIR/db/kb.sqlite`).

## Архитектура

Сервисы запускаются через `docker compose` и взаимодействуют по внутренней сети.

| Сервис  | Назначение | Порты |
|---------|------------|-------|
| `api`   | FastAPI: загрузка документов и чат | 8000 (внутр.) |
| `qdrant` | Векторное хранилище чанков | 6333 (внутр.) |
| `ollama` | LLM `qwen2.5:3b-instruct` | 11434 (внутр.) |
| `nginx` | HTTPS reverse proxy с basic auth и статикой | 80/443 |

## Подготовка окружения

1. Установите Docker и Compose (Ubuntu 22.04):
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose-plugin
   sudo systemctl enable --now docker
   ```

2. Клонируйте репозиторий и перейдите в каталог:
   ```bash
   git clone https://example.com/knowlab.git
   cd knowlab
   ```

3. Создайте каталоги для данных:
   ```bash
   sudo mkdir -p /opt/knowlab/data/{files,qdrant,logs,logs/nginx,logs/api,ssl,www,ollama}
   sudo chown -R "$USER" /opt/knowlab
   ```

4. Сгенерируйте самоподписанный сертификат и файл basic-auth (файл `.htpasswd` должен оказаться доступен nginx по пути `/data/ssl/.htpasswd` внутри контейнера):
   ```bash
   sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout /opt/knowlab/data/ssl/kb.local.key \
     -out /opt/knowlab/data/ssl/kb.local.crt \
     -subj "/CN=kb.local"
   htpasswd -bc /opt/knowlab/data/ssl/.htpasswd admin admin
   # при локальном запуске через docker compose из репозитория:
   htpasswd -bc srv/projects/kb/data/ssl/.htpasswd admin admin
   ```

5. Создайте `.env` на основе примера из репозитория:
   ```bash
   cp .env.example .env
   # отредактируйте параметры моделей и Qdrant при необходимости
   ```

## Запуск

1. Создайте внешнюю сеть (повторный запуск безопасен):
   ```bash
   docker network create web || true
   ```

2. Соберите и поднимите контейнеры:
   ```bash
   docker compose -f /srv/projects/kb/compose.yml build
   docker compose -f /srv/projects/kb/compose.yml up -d
   ```

3. Загрузите модель в Ollama (выполнить один раз после запуска):
   ```bash
   docker compose -f /srv/projects/kb/compose.yml exec kb_web \
     curl -s http://localhost:11434/api/pull -d '{"name":"qwen2.5:3b-instruct"}'
   ```

4. Проверьте работоспособность API и health-check:
   ```bash
   docker compose -f /srv/projects/kb/compose.yml exec kb_web curl -k https://localhost/health
   ```

5. Зайдите в веб-интерфейс `https://<сервер>` с учётными данными basic-auth (по умолчанию `admin/admin`). Страница содержит форму для ввода `user_id`, `conversation_id`, отправки сообщений и загрузки документов.

## Переменные окружения

| Переменная | Значение по умолчанию | Назначение |
|------------|-----------------------|------------|
| `DATA_DIR` | `/opt/knowlab/data/files` | Каталог для загружаемых документов и базы чатов. |
| `VECTOR_BACKEND` | `qdrant` | Тип векторного движка (поддерживается `qdrant`). |
| `QDRANT_URL` | `http://qdrant:6333` | Эндпоинт Qdrant. |
| `QDRANT_COLLECTION` | `kb_chunks` | Коллекция для документов. |
| `QDRANT_API_KEY` | пусто | Ключ доступа к Qdrant при необходимости. |
| `VECTOR_EMBED_MODEL` | `intfloat/multilingual-e5-small` | Модель эмбеддингов. |
| `VECTOR_EMBED_DIMENSION` | `384` | Размерность эмбеддингов (для контроля совместимости). |
| `LLM_PROVIDER` | `ollama` | Провайдер LLM (`ollama` или `stub`). |
| `LLM_MODEL_NAME` / `OLLAMA_MODEL` | `qwen2.5:3b-instruct` | Модель генерации ответов. |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Базовый URL Ollama. |
| `MAX_CONTEXT_TOKENS` | `4096` | Максимальное количество токенов контекста для Ollama. |
| `RAG_TOKENIZER_NAME` | `cl100k_base` | Название токенизатора для разбиения. |
| `RAG_CHUNK` | `900` | Размер чанка в токенах. |
| `RAG_OVERLAP` | `140` | Перекрытие чанков. |
| `RETRIEVE_TOPK` | `10` | Количество кандидатов из векторного поиска. |
| `RERANK_TOPK` | `10` | Сколько документов оставлять после повторного ранжирования. |
| `CHAT_MEMORY_ENABLED` | `true` | Включить долговременную память чата. |
| `MEMORY_DB_PATH` | пусто | Необязательный путь к базе памяти (по умолчанию `DATA_DIR/db/memory.sqlite`). |
| `CHAT_MEMORY_TTL_DAYS` | `90` | Сколько дней хранить сообщения в памяти. |
| `CHAT_MEMORY_MAXTOK` | `2000` | Максимальный объём памяти в условных «токенах». |
| `CHAT_SUMMARY_TRIGGER` | `10` | Порог сообщений для запуска саммаризации. |
| `CHAT_HISTORY_LIMIT` | `12` | Количество последних сообщений в краткосрочном контексте. |
| `CHAT_DB_BACKEND` | `sqlite` | Тип хранилища истории чатов: `sqlite` или `postgres`. |
| `CHAT_DB_PATH` | пусто | Необязательный путь к SQLite (по умолчанию `DATA_DIR/db/chat_history.sqlite`). |
| `CHAT_DB_DSN` | пусто | Строка подключения к PostgreSQL при `CHAT_DB_BACKEND=postgres`. |
| `CHAT_DB_SCHEMA` | пусто | Необязательная схема PostgreSQL. |
| `CHAT_MIN_CITATIONS` | `3` | Минимальное число источников в ответе. |
| `CHAT_MAX_CITATIONS` | `5` | Максимальное число источников в ответе. |
| `SECRET_KEY` | `change-me` | Ключ подписи JWT. |
| `JWT_ALGORITHM` | `HS256` | Алгоритм подписи JWT. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Время жизни токена в минутах. |

> Примечание: если размер окна разбиения (`RAG_CHUNK`) нормализуется до 1, система
> автоматически переключается на символьный токенизатор и генерирует окна по
> символам с учётом заданного перекрытия.

## Логирование и память

История диалогов по умолчанию сохраняется в SQLite-файле по пути `DATA_DIR/db/kb.sqlite`. При необходимости можно переключить приложение на PostgreSQL, задав `CHAT_DB_BACKEND=postgres` и передав строку подключения через `CHAT_DB_DSN`. Для каждого сообщения хранится пользователь, идентификатор диалога, роли (`user`/`assistant`) и содержание. Эти данные используются для восстановления контекста между запросами.

## Тестирование

Локально можно запустить unit-тесты (без Docker):
```bash
pip install -r requirements.txt
pip install pytest
pytest
```

Тесты используют набор локальных заглушек для зависимостей (`fastapi`,
`pydantic`, `pypdf`, `python-dotenv`). Эти минимальные реализации располагаются
в каталоге `tests/stubs`, а `pytest` добавляет его в `sys.path` при запуске, что
изолирует тестовый контур и не мешает настоящему приложению импортировать
реальные пакеты, когда они установлены в окружении.

Тесты, требующие PostgreSQL (`pytest -m requires_postgres`), подразумевают наличие утилиты `pg_config` в `PATH` и установленного сервера. При их выполнении используется временная база данных, которая создаётся и удаляется автоматически.

## Утилиты командной строки

В каталоге `scripts/` находятся вспомогательные CLI:

- `python -m scripts.ingest_path <путь>` — проиндексировать документы из файла или каталога.
- `python -m scripts.rebuild_index [<путь>]` — очистить коллекцию и переиндексировать содержимое (`DATA_DIR`, если путь не указан).
- `python -m scripts.export_all [export.json]` — выгрузить все чанки из векторного хранилища в JSON.
- `python -m scripts.import_all [--reset] export.json` — импортировать чанки из JSON и при необходимости сбросить коллекцию перед загрузкой.

## Проверка сценария с загрузкой документов

Для имитации боевого сценария предусмотрен тест `tests/test_acceptance.py`. Перед
обработкой файлов он вызывает функцию `ensure_demo_assets` из модуля
`tests.demo_assets`, которая автоматически генерирует PDF/DOCX/TXT-заготовки в
каталоге `srv/projects/kb/data/files/`. Благодаря этому тест всегда работает с
актуальными файлами, не требуя хранения бинарников в репозитории.

## Критерии приёмки

- ≥50 документов загружены и проиндексированы.
- Чат отвечает ≤15 секунд и приводит ≥3 цитаты.
- История диалогов сохраняется в SQLite и используется при последующих запросах.

MVP готов к демонстрации пользователям и сбору обратной связи.

## План расширений

Этот раздел служит roadmap для следующих итераций развития продукта. Детальная проработка и оценка задач будут выполнены в будущих тикетах.

- Поддержка OCR для сканов и изображений в документах.
- Импорт и разбор вложенных архивов и файлов формата XLSX/ZIP.
- Добавление кросс-энкодера для улучшенного повторного ранжирования результатов.
- Реализация ролевой модели доступа (RBAC) на уровне API и интерфейса.
- Настройка мониторинга метрик и логов для ключевых сервисов.
- Организация DR-бэкапов для критически важных данных и конфигураций.
