# kb_ai

Сервис полнотекстового поиска и ответа по внутренней базе знаний с опорой на Retrieval-Augmented Generation (RAG).

## Возможности
- загрузка PDF, DOCX и TXT документов с автоматическим разбиением на фрагменты;
- хранение векторных представлений документов в Qdrant и поиск релевантных контекстов;
- генерация ответов с помощью локально развёрнутого Ollama;
- долговременная память диалогов в PostgreSQL с автоматическим свёртыванием длинных бесед;
- REST API для интеграции с внешними интерфейсами и административный веб-интерфейс за Nginx.

## Архитектура
- **FastAPI** — HTTP API (`/api/docs/upload`, `/api/chat`, `/health`).
- **Sentence Transformers + Qdrant** — извлечение и хранение векторных представлений фрагментов.
- **Ollama** — генерация ответов (модель определяется переменной `GEN_MODEL`).
- **PostgreSQL** — хранилище памяти диалогов.
- **Nginx** — TLS-терминация, обратный прокси и выдача статического административного интерфейса.
- **Docker Compose** — развёртывание компонентов и служебных зависимостей.

## Быстрый старт (Docker Compose)
1. Скопируйте пример окружения и обновите значения под свою среду (обязательно задайте уникальный `APP_SECRET`, модели `GEN_MODEL`/`EMBED_MODEL`, а также параметры доступа к PostgreSQL, если они отличаются):
   ```bash
   cp data/scripts/example.env .env
   ```
2. Подготовьте директории с данными и логами (сертификаты, статические файлы, модели Ollama и т.п.):
   ```bash
   sudo mkdir -p /opt/knowlab/data/{files,qdrant,pg,logs}
   sudo mkdir -p /opt/knowlab/data/files/{www,ssl,db,ollama}
   sudo chown -R "$USER" /opt/knowlab/data
   ```
   В каталоге `/opt/knowlab/data/files/ssl` разместите самоподписанный сертификат (`kb.local.crt`, `kb.local.key`) и `.htpasswd` для Basic Auth.
3. Запустите сервисы:
   ```bash
   docker compose up -d
   ```
   После запуска административный интерфейс и API будут доступны на `https://<ваш-домен>/` (эндпоинты API проксируются через `/api`).

## Локальный запуск без Docker
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
export FILES_ROOT=$(pwd)/data/local
mkdir -p "$FILES_ROOT/db"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Для полноценной работы локально также потребуется запущенный Qdrant (`qdrant`), PostgreSQL и Ollama, доступные по значениям переменных окружения `QDRANT_URL`, `PG*` и `OLLAMA_HOST`.

## Загрузка документов и чат
- Загрузка документа:
  ```bash
  curl -X POST \
    -F "file=@docs/manual.pdf" \
    https://kb.local/api/docs/upload
  ```
- Диалог с ассистентом:
  ```bash
  curl -X POST https://kb.local/api/chat \
    -H "Content-Type: application/json" \
    -d '{
          "user_id": "u123",
          "conversation_id": "conv-1",
          "message": "Какие требования описаны на странице 5?"
        }'
  ```

Ответ содержит текст модели и список цитат с файлами и страницами.

## Переменные окружения
| Переменная | По умолчанию | Описание |
| --- | --- | --- |
| `APP_SECRET` | `dev` | Секретный ключ приложения. |
| `FILES_ROOT` | `/opt/knowlab/data/files` | Каталог с загруженными файлами, статикой и БД памяти. |
| `GEN_MODEL` | `qwen2.5:3b-instruct` | Имя модели Ollama для генерации. |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Модель эмбеддингов для индекса. |
| `OLLAMA_HOST` | `http://ollama:11434` | Базовый URL сервиса Ollama. |
| `QDRANT_URL` | `http://qdrant:6333` | URL кластера Qdrant. |
| `QDRANT_COLLECTION` | `kb_chunks` | Коллекция для хранения фрагментов. |
| `PGHOST` | `postgres` | Хост PostgreSQL для памяти диалогов. |
| `PGPORT` | `5432` | Порт PostgreSQL. |
| `PGDATABASE` | `knowlab` | Имя БД PostgreSQL. |
| `PGUSER` | `knowlab` | Пользователь PostgreSQL. |
| `PGPASSWORD` | `change-me` | Пароль PostgreSQL. |
| `CHAT_MEMORY_ENABLED` | `true` | Включение памяти диалогов. |
| `CHAT_MEMORY_TTL_DAYS` | `90` | Срок хранения истории в днях. |
| `CHAT_SUMMARY_TRIGGER` | `10` | Количество сообщений до свёртки истории. |
| `CHAT_MEMORY_MAXTOK` | `2000` | Ограничение на размер свёрнутой памяти. |
| `RAG_CHUNK` | `900` | Размер фрагмента при разбиении текста. |
| `RAG_OVERLAP` | `140` | Перекрытие соседних фрагментов. |
| `RETRIEVE_TOPK` | `24` | Количество кандидатов для поиска. |
| `RATE_LIMIT` | `30r/m` | Базовый лимит запросов Nginx. |
| `RATE_BURST` | `20` | Допустимый burst для лимитера. |

## Структура репозитория
```
app/                исходный код приложения FastAPI
├─ main.py          HTTP-эндпоинты и диалоговая логика
├─ models/          клиенты Ollama и Qdrant
├─ rag/             парсинг и нарезка документов
├─ memory/          реализация памяти диалогов (PostgreSQL/SQLite)
compose.yml         docker-compose для продакшн-развёртывания
Dockerfile          образ API сервиса
```

## Лицензия
Проект распространяется под лицензией MIT (см. `LICENSE`, если файл присутствует в репозитории).
