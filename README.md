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

## Operations Console UI

- Глобальный статус инфраструктуры: отображение SQLite, Vector Store, LLM и LoRA с автообновлением и
  подсказками по ошибкам.
- Метрики и активность: карточки с количеством документов, активных индексаций и ошибок, последние
  загрузки в виде ленты событий.
- Улучшенный UX: drag-and-drop загрузка, прогресс-бар, тёмная/светлая темы, тост-уведомления и
  современный чат с цитатами.
- Быстрые действия: кнопка моментального обновления, очистка чата, отправка результатов поиска в диалог
  одним кликом.

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
uvicorn app.main:app --host 0.0.0.0 --port 8000
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
