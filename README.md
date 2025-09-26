# KnowLab MVP

KnowLab — база знаний с поддержкой Retrieval-Augmented Generation. Этот репозиторий реализует MVP, описанный в техническом задании: загрузка документов, индексирование, чат с цитатами, учёт ролей и развёртывание в Docker.

## Цели и функциональность

- Загрузка документов (DOCX, PDF с текстом, TXT) через веб-админку.
- Автоматическое извлечение текста, разбиение на чанки по 900 токенов с overlap 140 и сохранение в Qdrant с эмбеддингами `intfloat/multilingual-e5-small`.
- Поиск и чат через веб-интерфейс с генерацией ответов моделью `qwen2.5:3b-instruct` (Ollama) и минимум тремя цитатами (файл + страница).
- Разделение ролей: `admin` (загрузка, просмотр логов) и `staff` (чат).
- Логирование в PostgreSQL: пользователь, вопрос, цитируемые файлы/страницы, задержка ответа.

## Архитектура

Сервисы запускаются через `docker compose` и взаимодействуют по внутренней сети.

| Сервис  | Назначение | Порты |
|---------|------------|-------|
| `api`   | FastAPI: загрузка документов, чат, админка | 8000 (внутр.) |
| `postgres` | PostgreSQL: пользователи и логи | 5432 (внутр.) |
| `qdrant` | Векторное хранилище чанков | 6333 (внутр.) |
| `ollama` | LLM `qwen2.5:3b-instruct` | 11434 (внутр.) |
| `nginx` | HTTPS reverse proxy с basic auth | 80/443 |

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
   sudo mkdir -p /opt/knowlab/data/{files,qdrant,pg,logs,logs/nginx,logs/api,ssl,www,ollama}
   sudo chown -R "$USER" /opt/knowlab
   ```

4. Сгенерируйте самоподписанный сертификат и файл basic-auth (минимальная защита админки):
   ```bash
   sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout /opt/knowlab/data/ssl/kb.local.key \
     -out /opt/knowlab/data/ssl/kb.local.crt \
     -subj "/CN=kb.local"
   htpasswd -bc /opt/knowlab/data/ssl/.htpasswd admin admin
   ```

5. Создайте `.env` на основе примера:
   ```bash
   cp data/scripts/example.env .env
   # отредактируйте APP_SECRET, пароли и при необходимости Qdrant/DB параметры
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

4. Зайдите в веб-интерфейс `https://<сервер>` с учётными данными `admin/admin`. При первом входе система потребует сменить пароль.

5. В админке загрузите ≥50 документов (DOCX, PDF, TXT) и задайте вопрос от учётной записи `staff`. Ответ должен прийти ≤15 секунд и содержать ≥3 цитаты.

## Переменные окружения

| Переменная | Значение по умолчанию | Назначение |
|------------|-----------------------|------------|
| `APP_SECRET` | `change-me` | Секрет для JWT. |
| `FILES_ROOT` | `/opt/knowlab/data/files` | Каталог для загружаемых документов. |
| `GEN_MODEL` | `qwen2.5:3b-instruct` | Модель для ответов. |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Модель эмбеддингов. |
| `RAG_CHUNK` | `900` | Размер чанка в токенах. |
| `RAG_OVERLAP` | `140` | Перекрытие чанков. |
| `RETRIEVE_TOPK` | `10` | Количество кандидатов из Qdrant. |
| `DATABASE_URL` | `postgresql+psycopg://knowlab:change-me@postgres:5432/knowlab` | Подключение к PostgreSQL. |

## Логирование

FastAPI записывает события чата в таблицу `chat_logs` (PostgreSQL): пользователь, вопрос, сокращённый ответ, задержка (мс) и список цитат (файл + страница + score). Просмотр доступен по `/admin/chat-logs` для роли `admin`.

## Тестирование

Локально можно запустить unit-тесты (без Docker):
```bash
pip install -r app/requirements.txt
pytest
```

## Критерии приёмки

- ≥50 документов загружены и проиндексированы.
- Чат отвечает ≤15 секунд и приводит ≥3 цитаты.
- Роли `admin`/`staff` работают согласно правам.
- Логи сохраняются в PostgreSQL и доступны в админке.

MVP готов к демонстрации пользователям и сбору обратной связи.
