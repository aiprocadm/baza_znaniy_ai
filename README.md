    codex/update-readme-with-detailed-setup-instructions
# kb_ai Deployment Guide

This guide walks you through preparing your environment, configuring secrets, building and running the Docker stack, preloading models, and validating that everything is working. It also covers rollback and recovery so you can undo changes safely. Follow each step carefully—most commands are copy-paste ready.

## 1. Prepare the Environment

1. **Install system dependencies** (Ubuntu/Debian example):
   ```bash
   sudo apt update
   sudo apt install -y git docker.io docker-compose nginx-full apache2-utils cron openssl
   ```
2. **Enable Docker at startup and add your user to the Docker group** (log out/in after running the second command):
   ```bash
   sudo systemctl enable --now docker
   sudo usermod -aG docker "$USER"
   ```
3. **Clone the repository**:
   ```bash
   git clone https://github.com/example/kb_ai.git
   cd kb_ai
   ```
4. **Keep the application host fixed**: the deployment expects `APP_HOST=kb.local`. Configure your local `/etc/hosts` or DNS to point `kb.local` to the server.

## 2. Create Configuration Files with Here-Docs

Use here-documents so the files are created with the exact content shown. Adjust values as needed, but keep `APP_HOST=kb.local` intact.

1. **`.env`** (environment variables):
   ```bash
   cat <<'EOF' > .env
   APP_ENV=production
   APP_HOST=kb.local
   APP_PORT=8443
   ADMIN_USER=admin
   ADMIN_PASSWORD=admin
   SECRET_KEY_BASE=$(openssl rand -hex 32)
   DATABASE_URL=postgres://kb_user:change-me@db:5432/kb_ai
   REDIS_URL=redis://cache:6379/0
   MODEL_CACHE_DIR=/models
   SSL_CERT_PATH=./nginx/certs/kb.local.crt
   SSL_KEY_PATH=./nginx/certs/kb.local.key
   AUTH_FILE=./nginx/.htpasswd
   ```
   EOF
   ```
   > **Note:** The default credentials are `admin` / `admin`. Change them before going to production.

2. **`docker-compose.yml`** (only if you need to create/replace it):
   ```bash
   cat <<'EOF' > docker-compose.yml
   version: "3.9"

   services:
     web:
       build: .
       env_file: .env
       ports:
         - "8443:8443"
       volumes:
         - ./data:/app/data
         - ./models:/models
       depends_on:
         - db
         - cache

     db:
       image: postgres:15
       environment:
         POSTGRES_DB: kb_ai
         POSTGRES_USER: kb_user
         POSTGRES_PASSWORD: change-me
       volumes:
         - db_data:/var/lib/postgresql/data

     cache:
       image: redis:7
       volumes:
         - cache_data:/data

   volumes:
     db_data:
     cache_data:
   EOF
   ```

3. **`nginx/default.conf`** (HTTPS reverse proxy):
   ```bash
   mkdir -p nginx
   cat <<'EOF' > nginx/default.conf
   server {
       listen 8443 ssl;
       server_name kb.local;

       ssl_certificate     /etc/nginx/certs/kb.local.crt;
       ssl_certificate_key /etc/nginx/certs/kb.local.key;

       auth_basic           "kb_ai Admin";
       auth_basic_user_file /etc/nginx/.htpasswd;

       location / {
           proxy_pass http://web:8443;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   EOF
   ```

## 3. Edit Files Manually with `nano`

If you need to tweak any generated file:

1. Open the file in nano, e.g. `.env`:
   ```bash
   nano .env
   ```
2. Use the arrow keys to navigate and make changes.
3. Press `Ctrl+O`, then `Enter` to save.
4. Press `Ctrl+X` to exit.

## 4. Generate Secrets and Credentials

1. **Regenerate secrets if needed**:
   ```bash
   openssl rand -hex 32
   ```
   Replace the value of `SECRET_KEY_BASE` in `.env` using nano if you regenerate.

2. **Create the `.htpasswd` file for basic auth**:
   ```bash
   mkdir -p nginx
   htpasswd -bc nginx/.htpasswd admin admin
   ```
   Replace the second `admin` with your secure password.

3. **Generate SSL certificates (self-signed example)**:
   ```bash
   mkdir -p nginx/certs
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/certs/kb.local.key \
     -out nginx/certs/kb.local.crt \
     -subj "/CN=kb.local"
   ```

## 5. Build and Run with Docker

1. **Build images**:
   ```bash
   docker compose build
   ```
2. **Start the stack in the background**:
   ```bash
   docker compose up -d
   ```
3. **View running containers**:
   ```bash
   docker compose ps
   ```

## 6. Preload Models

1. Ensure the `models` directory exists and is writable:
   ```bash
   mkdir -p models
   chmod 775 models
   ```
2. Download or copy your model files into `./models`. Example using `wget`:
   ```bash
   wget -O models/embeddings.bin https://example.com/models/embeddings.bin
   ```
3. If the application provides a preload command, run it inside the container (replace with your command if different):
   ```bash
   docker compose exec web python manage.py preload_models --path /models
   ```

## 7. Schedule Cron Jobs

If the app requires regular indexing or cleanup tasks, create a cron entry.

1. Open the crontab editor:
   ```bash
   crontab -e
   ```
2. When prompted, choose nano if asked.
3. Add a job, for example to refresh the knowledge base every hour:
   ```cron
   0 * * * * docker compose exec -T web python manage.py refresh_index >> /var/log/kb_ai_cron.log 2>&1
   ```
4. Save (`Ctrl+O`, `Enter`) and exit (`Ctrl+X`).

## 8. Verify the Deployment

Run these checks after the stack is running:

1. **Check logs**:
   ```bash
   docker compose logs -f web
   docker compose logs -f db
   ```
2. **Confirm the HTTPS endpoint responds** (ignore certificate warnings for self-signed certs):
   ```bash
   curl -k https://kb.local:8443/health
   ```
3. **Inspect container status**:
   ```bash
   docker ps --filter "name=kb_ai"
   ```
4. **Validate cron execution** (check the log file you configured):
   ```bash
   tail -f /var/log/kb_ai_cron.log
   ```

## 9. Rollback and Recovery

If something goes wrong, follow these steps to return to a known good state:

1. **Stop and remove the stack**:
   ```bash
   docker compose down
   ```
2. **Remove dangling images** (optional cleanup):
   ```bash
   docker image prune -a
   ```
3. **Restore configuration files from backup**:
   ```bash
   cp /backups/kb_ai/.env .env
   cp /backups/kb_ai/docker-compose.yml docker-compose.yml
   cp -r /backups/kb_ai/nginx ./nginx
   ```
4. **Restore database backups** (example using `pg_restore`):
   ```bash
   pg_restore -h localhost -U kb_user -d kb_ai /backups/db/kb_ai_latest.dump
   ```
5. **Rebuild and relaunch**:
   ```bash
   docker compose build --no-cache
   docker compose up -d
   ```
6. **Re-run verification commands** (Section 8) to confirm the system is healthy.

## 10. Default Credentials and Security Reminders

- The administrative interface defaults to `admin` / `admin`. Change both the username and password immediately.
- Keep `APP_HOST=kb.local` as provided, or update all references (environment variables, SSL certificates, DNS) consistently if you must change it.
- Rotate secrets regularly and store them in a secure vault.

## 11. Helpful Aliases (Optional)

Add these to your shell profile to streamline daily operations:

```bash
alias kbu='docker compose up -d'
alias kbd='docker compose down'
alias kbl='docker compose logs -f web'
```

Reload your shell (`source ~/.bashrc`) after adding the aliases.

---
Following this procedure will give you a repeatable deployment for `kb_ai`, covering setup, maintenance, and recovery. Share these instructions with teammates to keep everyone aligned.

# kb_ai

Сервис полнотекстового поиска и ответа по внутренней базе знаний с опорой на Retrieval-Augmented Generation (RAG).

## Возможности
- загрузка PDF, DOCX и TXT документов с автоматическим разбиением на фрагменты;
- хранение векторных представлений документов в Qdrant и поиск релевантных контекстов;
- генерация ответов с помощью локально развёрнутого Ollama;
- долговременная память диалогов в PostgreSQL с автоматическим свёртыванием длинных бесед;
- REST API для интеграции с внешними интерфейсами и административный веб-интерфейс за Nginx.

## Архитектура
        codex/split-existing-service-into-containers
- **FastAPI** — HTTP API (`/api/docs/upload`, `/api/chat`, `/health`) в отдельном контейнере `api`.
- **Nginx** — TLS-терминация, базовая авторизация и проксирование запросов к API (контейнер `nginx`).
- **Ollama** — генерация ответов (контейнер `ollama`, модель определяется переменной `GEN_MODEL`).
- **Qdrant** — хранение и поиск векторных представлений фрагментов (контейнер `qdrant`).
- **Sentence Transformers** — вычисление эмбеддингов для индекса.
- **PostgreSQL** — долговременное хранилище памяти диалогов (контейнер `db`, конфигурируется через `DATABASE_URL`).
- **Docker Compose** — оркестрация сервисов, общие сети и тома `data/{files,qdrant,pg,logs}`.

        codex/setup-postgresql-as-data-source
- **FastAPI** — HTTP API (`/api/auth/token`, `/api/docs/upload`, `/api/chat`, `/admin/logs`, `/health`).
- **Sentence Transformers + FAISS** — извлечение и хранение векторных представлений фрагментов.
- **Ollama** — генерация ответов (модель определяется переменной `GEN_MODEL`).
- **PostgreSQL** — пользователи, роли, журнал обращений.
- **SQLite** — хранение памяти диалогов.

- **FastAPI** — HTTP API (`/api/docs/upload`, `/api/chat`, `/health`).
- **Sentence Transformers + Qdrant** — извлечение и хранение векторных представлений фрагментов.
- **Ollama** — генерация ответов (модель определяется переменной `GEN_MODEL`).
      codex/split-monolithic-image-into-separate-services
- **PostgreSQL** — хранилище памяти диалогов.
- **Nginx** — TLS-терминация, обратный прокси и выдача статического административного интерфейса.
- **Docker Compose** — развёртывание компонентов и служебных зависимостей.

- **SQLite** — хранилище для памяти чата.
        main
- **Docker Compose** — развёртывание сервиса вместе с конфигурацией Nginx.
      main
        main

## Быстрый старт (Docker Compose)
1. Скопируйте пример окружения и обновите значения под свою среду (обязательно задайте уникальный `APP_SECRET`, модели `GEN_MODEL`/`EMBED_MODEL`, а также параметры доступа к PostgreSQL, если они отличаются):
   ```bash
   cp data/scripts/example.env .env
   ```
        codex/split-existing-service-into-containers
2. Подготовьте директории с данными и журналами (сертификаты, веб-ресурсы, документы, базы данных):
   ```bash
   sudo mkdir -p /srv/projects/kb/data/{www,ssl,files,qdrant,pg,logs,logs/nginx,logs/api,files/ollama}
   sudo chown -R "$USER" /srv/projects/kb/data

2. Подготовьте директории с данными и логами (сертификаты, статические файлы, модели Ollama и т.п.):
   ```bash
   sudo mkdir -p /opt/knowlab/data/{files,qdrant,pg,logs}
   sudo mkdir -p /opt/knowlab/data/files/{www,ssl,db,ollama}
   sudo chown -R "$USER" /opt/knowlab/data
        main
   ```
   В каталоге `/opt/knowlab/data/files/ssl` разместите самоподписанный сертификат (`kb.local.crt`, `kb.local.key`) и `.htpasswd` для Basic Auth.
3. Запустите сервисы:
   ```bash
   docker compose up -d
   ```
   После запуска административный интерфейс и API будут доступны на `https://<ваш-домен>/` (эндпоинты API проксируются через `/api`).

   > **Примечание:** контейнер `api` зависит от готовности сервисов `db`, `qdrant` и `ollama`. Для корректной работы памяти диалогов задайте валидный `DATABASE_URL` в `.env` (см. пример в `data/scripts/example.env`).

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
1. Получите токен доступа:
   ```bash
   curl -X POST http://localhost:8000/api/auth/token \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=admin"
   ```
   > При первом входе под учетной записью `admin/admin` необходимо сменить пароль через `/api/auth/change-password`.

2. Смените пароль администратора (обязательно при первом входе):
   ```bash
   curl -X POST http://localhost:8000/api/auth/change-password \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"current_password": "admin", "new_password": "S3cure!"}'
   ```

- Загрузка документа (роль `admin`):
  ```bash
  curl -X POST \
    -F "file=@docs/manual.pdf" \
        codex/setup-postgresql-as-data-source
    -H "Authorization: Bearer <token>" \
    http://localhost:8000/api/docs/upload
=======
    https://kb.local/api/docs/upload
        main
  ```
- Диалог с ассистентом (роль `staff` или `admin`):
  ```bash
  curl -X POST https://kb.local/api/chat \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <token>" \
    -d '{
          "conversation_id": "conv-1",
          "message": "Какие требования описаны на странице 5?"
        }'
  ```

Ответ содержит текст модели и список цитат с файлами и страницами.

## Переменные окружения
| Переменная | По умолчанию | Описание |
| --- | --- | --- |
| `APP_SECRET` | `dev` | Секретный ключ приложения. |
        codex/setup-postgresql-as-data-source
| `DATABASE_URL` | `postgresql+psycopg2://kb:kb@localhost:5432/kb` | Подключение к PostgreSQL. |
=======
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
        main
| `CHAT_MEMORY_ENABLED` | `true` | Включение памяти диалогов. |
| `CHAT_MEMORY_TTL_DAYS` | `90` | Срок хранения истории в днях. |
| `CHAT_SUMMARY_TRIGGER` | `10` | Количество сообщений до свёртки истории. |
| `CHAT_MEMORY_MAXTOK` | `2000` | Ограничение на размер свёрнутой памяти. |
| `RAG_CHUNK` | `900` | Размер фрагмента при разбиении текста. |
| `RAG_OVERLAP` | `140` | Перекрытие соседних фрагментов. |
       codex/split-monolithic-image-into-separate-services
| `RETRIEVE_TOPK` | `24` | Количество кандидатов для поиска. |
| `RATE_LIMIT` | `30r/m` | Базовый лимит запросов Nginx. |
| `RATE_BURST` | `20` | Допустимый burst для лимитера. |

| `QDRANT_URL` | — | Полный URL Qdrant (например, `http://qdrant:6333`). При указании `QDRANT_HOST`/`QDRANT_PORT` не требуется. |
| `QDRANT_HOST` | `qdrant` | Хост Qdrant при прямом подключении. |
| `QDRANT_PORT` | `6333` | Порт Qdrant при прямом подключении. |
| `QDRANT_API_KEY` | — | API ключ Qdrant при использовании облачной версии. |
| `QDRANT_COLLECTION` | `kb_chunks` | Название коллекции для хранения фрагментов. |
    main

## Структура репозитория
```
app/                исходный код приложения FastAPI
        codex/setup-postgresql-as-data-source
├─ main.py          HTTP-эндпоинты, аутентификация и журналирование
├─ database.py      инициализация подключения к PostgreSQL
├─ models/          клиенты Ollama/векторного индекса и ORM-модели
├─ templates/       HTML-шаблоны административных страниц

├─ main.py          HTTP-эндпоинты и диалоговая логика
├─ models/          клиенты Ollama и Qdrant
        main
├─ rag/             парсинг и нарезка документов
├─ memory/          реализация памяти диалогов (PostgreSQL/SQLite)
compose.yml         docker-compose для продакшн-развёртывания
Dockerfile          образ API сервиса
```

## Лицензия
Проект распространяется под лицензией MIT (см. `LICENSE`, если файл присутствует в репозитории).
      main
