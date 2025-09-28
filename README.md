# KnowLab MVP

KnowLab — база знаний с Retrieval-Augmented Generation. Репозиторий содержит минимально жизнеспособную реализацию: загрузка документов, индексирование, чат с цитатами и развёртывание в Docker. Авторизация реализована через basic auth на стороне nginx, а само приложение не требует JWT и ролей.

## Цели и функциональность

- Загрузка документов (DOCX, PDF с текстом, TXT) через веб-интерфейс.
- Извлечение текста, разбиение на чанки по 900 токенов с overlap 140 и сохранение в Qdrant с эмбеддингами `intfloat/multilingual-e5-small`.
- Чат с моделью `qwen2.5:3b-instruct` (Ollama) и возвратом цитат.
- Запоминание истории диалогов в локальной SQLite-базе (`FILES_ROOT/db/kb.sqlite`).

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

5. Создайте `.env` на основе примера:
   ```bash
   cp data/scripts/example.env .env
   # отредактируйте параметры моделей и Qdrant при необходимости
   ```

## Запуск

1. Стартуйте стек:
   ```bash
   docker compose up -d
   ```

2. Загрузите модель в Ollama (выполнить один раз после запуска):
   ```bash
   curl -s http://localhost:11434/api/pull -d '{"name":"qwen2.5:3b-instruct"}'
   ```

3. Проверьте работоспособность API:
   ```bash
   curl -k https://localhost/health
   ```

4. Зайдите в веб-интерфейс `https://<сервер>` с учётными данными basic-auth (по умолчанию `admin/admin`). Страница содержит форму для ввода `user_id`, `conversation_id`, отправки сообщений и загрузки документов.

## Переменные окружения

| Переменная | Значение по умолчанию | Назначение |
|------------|-----------------------|------------|
| `FILES_ROOT` | `/opt/knowlab/data/files` | Каталог для загружаемых документов и памяти чата. |
| `GEN_MODEL` | `qwen2.5:3b-instruct` | Модель для генерации ответов. |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Модель эмбеддингов. |
| `RAG_CHUNK` | `900` | Размер чанка в токенах. |
| `RAG_OVERLAP` | `140` | Перекрытие чанков. |
| `RETRIEVE_TOPK` | `10` | Количество кандидатов из Qdrant. |
| `RERANK_TOPK` | `10` | Сколько документов оставлять после повторного ранжирования. |
| `CHAT_MEMORY_ENABLED` | `true` | Сохранять ли историю диалогов. |
| `CHAT_MEMORY_TTL_DAYS` | `90` | Сколько дней хранить сообщения в памяти. |
| `CHAT_SUMMARY_TRIGGER` | `10` | Порог для свёртки памяти. |
| `CHAT_MEMORY_MAXTOK` | `2000` | Максимальный объём памяти (в «токенах» текста). |
| `CHAT_HISTORY_LIMIT` | `12` | Количество последних сообщений, включаемых в краткосрочный контекст. |
| `CHAT_MIN_CITATIONS` | `3` | Минимальное число источников в ответе. |
| `CHAT_MAX_CITATIONS` | `5` | Максимальное число источников в ответе. |
| `QDRANT_URL` | `http://qdrant:6333` | Эндпоинт Qdrant. |
| `QDRANT_COLLECTION` | `kb_chunks` | Коллекция для документов. |
| `QDRANT_API_KEY` | пусто | Ключ доступа к Qdrant при необходимости. |
| `OLLAMA_HOST` | `http://ollama:11434` | Эндпоинт Ollama. |
| `RATE_LIMIT` | `30r/m` | Глобальный лимит запросов, прокидываемый в Nginx. |
| `RATE_BURST` | `20` | Допустимый «всплеск» запросов поверх лимита. |
| `BASIC_USER` | `admin` | Имя файла с паролями для basic-auth (`/data/ssl/$BASIC_USER`). |

## Логирование и память

История диалогов сохраняется в SQLite-файле по пути `FILES_ROOT/db/kb.sqlite`. Для каждого сообщения хранится пользователь, идентификатор диалога, роли (`user`/`assistant`) и содержание. Эти данные используются для восстановления контекста между запросами.

## Тестирование

Локально можно запустить unit-тесты (без Docker):
```bash
pip install -r requirements.txt
pip install -r app/requirements.txt
pytest
```

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
