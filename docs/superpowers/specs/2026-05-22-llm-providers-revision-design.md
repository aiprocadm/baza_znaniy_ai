# LLM Providers Revision — Design

**Date:** 2026-05-22
**Branch:** `feat/kb-mvp-corporate-rag`
**Scope:** Полная ревизия LLM-слоя `app/services/kb_llm.py`: рефакторинг в пакет с абстрактным транспортом, добавление 8 новых бесплатных провайдеров (7 OpenAI-совместимых + GigaChat + YandexGPT native), обновление документации и `.env.example`.

---

## 1. Background and motivation

В MVP-слое `app/services/kb_llm.py` сейчас 6 LLM-провайдеров: DeepSeek, Groq, OpenRouter, OpenAI, Ollama, custom. Все говорят OpenAI-совместимым протоколом `POST /v1/chat/completions`, поэтому обслуживаются одним классом `OpenAICompatibleProvider`.

Требуется добавить ещё несколько бесплатных провайдеров и оставить место для дальнейшего расширения, в том числе теми, кто не использует OpenAI-протокол. Среди новых:

- **Международные бесплатные OpenAI-совместимые:** Cerebras, Together AI, Mistral La Plateforme, GitHub Models, SambaNova, Hugging Face Inference, Google Gemini (OpenAI-compat beta).
- **Российские с особой авторизацией:** GigaChat (OAuth2 client_credentials с обновлением токена), YandexGPT (свой формат JSON + IAM/API-key auth).
- **Локальные:** обновление `OLLAMA_MODEL`-примеров для русскоязычных моделей (Qwen 2.5, Saiga, Vikhr, T-pro).

Простое расширение `KNOWN_PRESETS` для большинства из них работает, но YandexGPT и GigaChat нарушают унифицированный путь. Чтобы не плодить хаки, делаем чистую multi-layer архитектуру через `Protocol` + registry.

## 2. Goals / Non-goals

### Goals

- Добавить 8 новых LLM-провайдеров в кодовую базу.
- Сохранить публичную поверхность `kb_llm` (`select_provider`, `provider_status`, `build_provider`, типы) **полностью обратно совместимой** — без изменений в `kb_mvp.py` и существующих тестах.
- Обновить `.env.example`, README с таблицей сравнения и decision tree выбора провайдера.
- Покрыть новый код unit-тестами через моки HTTP, чтобы CI не зависел от внешних API.

### Non-goals

- **Не** перерабатываем зрелый `app/llm/` (llama.cpp + LoRA) — это отдельный путь.
- **Не** добавляем live-тесты против реальных провайдеров в CI (только опциональный `pytest -m live` запускаемый локально).
- **Не** изменяем UI кроме опциональной chip-метки в Phase 5 (не блокирующая).
- **Не** интегрируем LiteLLM или другую внешнюю абстракцию — сохраняем минималистичный подход.
- **Не** трогаем embeddings (`kb_embeddings.py`) — отдельная подсистема.

## 3. Architecture

### 3.1 Multi-layer overview

```
┌────────────────────────────────────────────────────────────────────┐
│ Public API (app/services/kb_llm/__init__.py)                       │
│   select_provider, build_provider, provider_status                 │
│   LLMResponse, LLMUnavailable, LLMTransportError                   │
└────────────────────────────────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ Protocol layer (base.py)                                           │
│   class LLMProvider(Protocol):                                     │
│       name, model, is_available()                                  │
│       generate() → LLMResponse                                     │
│       generate_stream() → AsyncIterator[str]                       │
│   class AuthProvider(Protocol):                                    │
│       get_bearer_token() → str                                     │
└────────────────────────────────────────────────────────────────────┘
                                ▼
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ OpenAICompatible-    │ YandexGPTProvider    │ (future: native      │
│ Provider             │ (IAM + custom JSON)  │  transports)         │
│ + StaticTokenAuth    │                      │                      │
│ + OAuth2Credentials- │                      │                      │
│   Auth (GigaChat)    │                      │                      │
└──────────────────────┴──────────────────────┴──────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ Registry (registry.py)                                             │
│   REGISTRY: dict[name → factory]                                   │
│   auto-detect order (free-first)                                   │
│ Presets (presets.py)                                               │
│   OPENAI_COMPAT_PRESETS dict                                       │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 File layout

```
app/services/kb_llm/
├── __init__.py            # Публичный фасад, re-exports всё.
│                          # Импорты в kb_mvp.py не меняются.
├── base.py                # LLMProvider, AuthProvider (Protocol-ы),
│                          # LLMResponse, LLMConfig,
│                          # LLMUnavailable, LLMTransportError
├── presets.py             # OPENAI_COMPAT_PRESETS dict
│                          # (deepseek, groq, openrouter, openai,
│                          #  ollama, custom, cerebras, together,
│                          #  mistral, github_models, sambanova,
│                          #  huggingface, gemini, gigachat)
│                          # + KNOWN_PRESETS — алиас для совместимости
├── openai_compat.py       # OpenAICompatibleProvider (sync + stream),
│                          # StaticTokenAuth, OAuth2ClientCredentialsAuth,
│                          # _extract_text, _extract_delta_text
├── yandex.py              # YandexGPTProvider, OpenAI ↔ Yandex
│                          # converters, NDJSON streaming
├── registry.py            # REGISTRY (name → factory),
│                          # build_provider, select_provider,
│                          # provider_status,
│                          # auto-detect order constant
└── _utils.py              # _safe_error_message, общие helpers
```

### 3.3 Backward compatibility contract

`__init__.py` re-exports всю старую публичную поверхность:

```python
from .base import (
    LLMProvider, LLMResponse, LLMConfig,
    LLMUnavailable, LLMTransportError,
)
from .openai_compat import (
    OpenAICompatibleProvider,
    StaticTokenAuth,
    OAuth2ClientCredentialsAuth,
    _extract_delta_text,
)
from .yandex import YandexGPTProvider
from .presets import OPENAI_COMPAT_PRESETS, KNOWN_PRESETS
from .registry import build_provider, select_provider, provider_status

__all__ = [
    "KNOWN_PRESETS",
    "LLMConfig",
    "LLMResponse",
    "LLMTransportError",
    "LLMUnavailable",
    "OpenAICompatibleProvider",
    "YandexGPTProvider",
    "build_provider",
    "provider_status",
    "select_provider",
]
```

После Phase 0:
- `from app.services import kb_llm` → работает.
- `kb_llm.select_provider()` → работает.
- `kb_llm.LLMTransportError` → работает.
- `kb_llm._extract_delta_text` → работает (re-exported).
- Никаких правок в `kb_mvp.py`.
- Никаких правок в `test_kb_mvp.py`.

## 4. New providers — Group breakdown

### 4.1 Group 1 — Pure OpenAI-compat (7 новых)

Добавляются как записи в `OPENAI_COMPAT_PRESETS`. Работают через существующий `OpenAICompatibleProvider`.

| Provider | `api_base` | Default model | `key_env` | Free tier |
|----------|-----------|---------------|-----------|-----------|
| **cerebras** | `https://api.cerebras.ai/v1` | `llama-3.3-70b` | `CEREBRAS_API_KEY` | ~30 req/min, 1M tok/день |
| **together** | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free` | `TOGETHER_API_KEY` | $1 кредит при регистрации |
| **mistral** | `https://api.mistral.ai/v1` | `mistral-small-latest` | `MISTRAL_API_KEY` | experimental free тариф |
| **github_models** | `https://models.inference.ai.azure.com` | `gpt-4o-mini` | `GITHUB_TOKEN` | rate-limited, любой gh token |
| **sambanova** | `https://api.sambanova.ai/v1` | `Meta-Llama-3.3-70B-Instruct` | `SAMBANOVA_API_KEY` | free tier |
| **huggingface** | `https://router.huggingface.co/v1` | `meta-llama/Llama-3.3-70B-Instruct` | `HF_TOKEN` | rate-limited, бесплатно (см. ниже) |
| **gemini** | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-1.5-flash` | `GEMINI_API_KEY` | щедрый free tier |

Каждая запись в `OPENAI_COMPAT_PRESETS` имеет поля:

```python
{
    "api_base": str,
    "default_model": str,
    "key_env": str | None,
    "model_env": str | None,
    "needs_key": bool,
    "extra_headers_env": dict[str, str] | None,
    "auth": "static" | "oauth2_client_credentials",  # NEW в Phase 2
    "auth_env": dict | None,                          # NEW в Phase 2 (только для oauth2)
}
```

**Замечание про HuggingFace env-vars:**

В существующем `.env.example` уже есть `HUGGINGFACE_HUB_TOKEN` — он используется для **скачивания GGUF-моделей через `huggingface_hub` SDK** в `scripts/download_model.py`. Это **другой кейс**, и мы его не трогаем.

Для нового inference-провайдера используется отдельная переменная `HF_TOKEN` (стандартное имя в экосистеме HF Inference Router). Эти две переменные могут содержать одно значение или разные — пользователь решает. В README будет явная пометка про различие.

### 4.2 Group 2 — OpenAI-compat + OAuth2 (GigaChat)

GigaChat REST принимает обычные OpenAI-запросы на `POST /chat/completions`, но access_token живёт 30 минут и обновляется через OAuth2 client_credentials.

**Auth flow:**
1. POST `https://ngw.devices.sberbank.ru:9443/api/v2/oauth`
   - Headers:
     - `Authorization: Basic <base64(client_id:client_secret)>`
     - `RqUID: <uuid4>` — обязательный уникальный идентификатор запроса (генерируется каждый раз, не из env)
     - `Content-Type: application/x-www-form-urlencoded`
     - `Accept: application/json`
   - Body: `scope=GIGACHAT_API_PERS` (или `GIGACHAT_API_CORP`)
2. Response: `{"access_token": "...", "expires_in": 1800000}` (миллисекунды Unix epoch до истечения, а не TTL!).
3. Затем обычный `POST /chat/completions` с `Authorization: Bearer <access_token>`.

> Внимание: API GigaChat возвращает абсолютный timestamp истечения в миллисекундах в поле `expires_at`, а не `expires_in` в секундах как в стандартном OAuth2. `OAuth2ClientCredentialsAuth` должен это учитывать (либо специальный adapter для GigaChat, либо параметризовать формат).

**Реализация:** новая абстракция `AuthProvider` + два класса:

```python
class AuthProvider(Protocol):
    def get_bearer_token(self) -> str: ...


class StaticTokenAuth:
    """Простой Bearer-токен из env. Используется всеми Group-1 провайдерами."""
    def __init__(self, token: str): self._token = token
    def get_bearer_token(self) -> str: return self._token


class OAuth2ClientCredentialsAuth:
    """OAuth2 client_credentials с автообновлением. Thread-safe кэш."""
    def __init__(self, client_id, client_secret, token_url, scope):
        ...
        self._lock = threading.Lock()
        self._cached_token = None
        self._expires_at = 0.0
    def get_bearer_token(self) -> str:
        # Fast path без lock
        if self._cached_token and time.time() < self._expires_at - 30:
            return self._cached_token
        # Slow path с double-check внутри lock
        with self._lock:
            if self._cached_token and time.time() < self._expires_at - 30:
                return self._cached_token
            self._fetch_new_token()
            return self._cached_token
```

`OpenAICompatibleProvider` получает поле `auth: AuthProvider` и использует `self.auth.get_bearer_token()` при формировании заголовка `Authorization`. Существующий путь сохраняется через `StaticTokenAuth(config.api_key)`.

GigaChat preset:
```python
"gigachat": {
    "api_base": "https://gigachat.devices.sberbank.ru/api/v1",
    "default_model": "GigaChat",
    "model_env": "GIGACHAT_MODEL",
    "needs_key": True,
    "auth": "oauth2_client_credentials",
    "auth_env": {
        "client_id": "GIGACHAT_CLIENT_ID",
        "client_secret": "GIGACHAT_CLIENT_SECRET",
        "token_url": "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        "scope_env": "GIGACHAT_SCOPE",   # default GIGACHAT_API_PERS
    },
}
```

### 4.3 Group 3 — Native transport (YandexGPT)

YandexGPT использует свой JSON-формат и Api-Key auth.

**Request:**
```http
POST https://llm.api.cloud.yandex.net/foundationModels/v1/completion
Authorization: Api-Key <YANDEX_API_KEY>
Content-Type: application/json

{
  "modelUri": "gpt://<folder_id>/yandexgpt-lite/latest",
  "completionOptions": {"stream": false, "temperature": 0.2, "maxTokens": "1024"},
  "messages": [
    {"role": "system", "text": "..."},
    {"role": "user",   "text": "..."}
  ]
}
```

**Response:**
```json
{
  "result": {
    "alternatives": [
      {"message": {"role": "assistant", "text": "..."}, "status": "ALTERNATIVE_STATUS_FINAL"}
    ],
    "usage": {"inputTextTokens": "5", "completionTokens": "1", "totalTokens": "6"},
    "modelVersion": "rc"
  }
}
```

**Streaming endpoint** возвращает NDJSON (одна строка = один JSON-объект с накапливающимся текстом).

**Реализация:** `YandexGPTProvider` в `yandex.py`. Тот же `LLMProvider` Protocol, но:
- Внутри — конвертеры `_openai_to_yandex_request()` и `_yandex_to_llm_response()`.
- HTTP: `Authorization: Api-Key ...` (не Bearer).
- Streaming: парсер NDJSON, каждый раз вычисляет delta от предыдущего фрагмента.
- Регистрируется в `REGISTRY` отдельно, не через `OPENAI_COMPAT_PRESETS`.

Env:
- `YANDEX_API_KEY` (required) — либо API-key.
- `YANDEX_FOLDER_ID` (required) — yandex cloud folder.
- `YANDEX_MODEL` (default `yandexgpt-lite/latest`).

### 4.4 Group 4 — Ollama defaults для русского

Не код, только примеры в `.env.example`:

```env
# Ollama: русскоязычные модели (выберите одну, потом ollama pull):
# OLLAMA_MODEL=qwen2.5:7b                  # Alibaba, отличный RU
# OLLAMA_MODEL=ilyagusev/saiga_llama3      # русско-tuned Llama 3
# OLLAMA_MODEL=vikhrmodels/it-5.4-fp16-1k  # Vikhr — RU instruction
# OLLAMA_MODEL=t-tech/T-pro-it-1.0          # T-Bank RU модель
#
# Embeddings (если включён ollama-эмбеддер):
# OLLAMA_EMBED_MODEL=bge-m3                # multilingual, рекомендуется
# OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b  # компактный, мультиязычный
```

## 5. Auto-detection priority

**Текущий порядок:** `deepseek → groq → openrouter → openai → custom`.

**Новый порядок (free-first):**
```
cerebras → groq → gemini → sambanova → together → huggingface →
deepseek → openrouter → mistral → github_models →
gigachat → yandex → openai → custom
```

**Логика:** если у пользователя есть ключ бесплатного провайдера, используем его. Платные (DeepSeek, OpenAI) уходят ниже.

**Ollama не участвует в auto-detection** — у него нет API-ключа, по которому можно детектить «настроенность». Используется только явно через `KB_LLM_PROVIDER=ollama`. Это сохраняется как есть (текущее поведение).

**UX-предупреждение в README:** для предсказуемого выбора используйте `KB_LLM_PROVIDER=<name>` явно. Иначе порядок может удивить.

## 6. Build phasing

### Phase 0 — Package refactor (zero behavior change)

**What:** перенос `app/services/kb_llm.py` → `app/services/kb_llm/` пакета.

**DoD:**
- Все существующие тесты в `test_kb_mvp.py` проходят без правок.
- `from app.services import kb_llm; kb_llm.select_provider()` работает.
- `make lint && make test` зелёные.

**Commit:** `refactor(kb-llm): extract provider transport layer into package`

### Phase 1 — Group 1 providers (7 OpenAI-compat)

**What:** добавить 7 записей в `OPENAI_COMPAT_PRESETS`, обновить auto-detection, расширить `.env.example`.

**DoD:**
- `kb_llm.build_provider("cerebras", env={"CEREBRAS_API_KEY": "k"})` возвращает рабочий объект.
- `provider_status()` показывает все 7 как `configured: false` без ключей.
- README получает таблицу сравнения провайдеров.
- Параметризованный тест проходит на все 7.

**Commit:** `feat(kb-llm): add 7 OpenAI-compatible free providers`

### Phase 2 — AuthProvider + GigaChat

**What:** ввести `AuthProvider` Protocol, реализации `StaticTokenAuth` и `OAuth2ClientCredentialsAuth`, добавить GigaChat preset.

**DoD:**
- Все Phase-1 провайдеры работают через `StaticTokenAuth` (zero regression).
- GigaChat: тест с mocked OAuth2 endpoint (token refresh, expiry, concurrent refresh).
- Thread-safe кэш токена.

**Commit:** `feat(kb-llm): add OAuth2 auth provider and GigaChat support`

### Phase 3 — YandexGPT native transport

**What:** новый класс `YandexGPTProvider` в `yandex.py`.

**DoD:**
- Mocked-httpx тесты для `generate()` (sync) и `generate_stream()` (async).
- Конвертер двусторонне симметричен.
- Обработка `status=ALTERNATIVE_STATUS_TRUNCATED` (warning + возврат текста).
- Отсутствие `YANDEX_FOLDER_ID` → `LLMUnavailable` с понятным сообщением.

**Commit:** `feat(kb-llm): add native YandexGPT transport`

### Phase 4 — Documentation

**What:** README таблица сравнения, decision tree, примеры русскоязычных Ollama-моделей.

**DoD:**
- Новый раздел «Как выбрать LLM-провайдера?»
- Все упомянутые модели существуют (ссылки на HF / Ollama Hub).
- `.env.example` имеет полные секции для всех новых провайдеров.

**Commit:** `docs(kb-llm): expand provider comparison and Ollama examples`

### Phase 5 (optional) — UI chip

**What:** `data/www/index.html` отражает активного провайдера; RU-метка для gigachat/yandex.

**DoD:**
- Chip в шапке UI корректно отражает `GET /api/kb/providers`.
- Для `gigachat` / `yandex` дополнительный label `RU`.

**Commit:** `feat(ui): show active LLM provider chip in header`

## 7. Testing strategy

### 7.1 Mock-based unit tests (CI-safe)

Все тесты используют `httpx_mock` или ручные моки. Никаких реальных API-вызовов в CI.

**Phase 0:** нет новых тестов, только подтверждение что старые проходят.

**Phase 1:** параметризованный тест на 7 OpenAI-compat провайдеров:

```python
@pytest.mark.parametrize("provider,key_env,model_default,api_base_substr", [
    ("cerebras",     "CEREBRAS_API_KEY",  "llama-3.3-70b",                "cerebras.ai"),
    ("together",     "TOGETHER_API_KEY",  "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "together.xyz"),
    ("mistral",      "MISTRAL_API_KEY",   "mistral-small-latest",         "mistral.ai"),
    ("github_models","GITHUB_TOKEN",      "gpt-4o-mini",                  "models.inference.ai.azure.com"),
    ("sambanova",    "SAMBANOVA_API_KEY", "Meta-Llama-3.3-70B-Instruct",  "sambanova.ai"),
    ("huggingface",  "HF_TOKEN",          "meta-llama/Llama-3.3-70B-Instruct", "huggingface.co"),
    ("gemini",       "GEMINI_API_KEY",    "gemini-1.5-flash",             "generativelanguage.googleapis.com"),
])
def test_build_provider_openai_compat(provider, key_env, model_default, api_base_substr):
    p = kb_llm.build_provider(provider, env={key_env: "k"})
    assert p.name == provider
    assert p.model == model_default
    assert api_base_substr in p.config.api_base
    assert p.is_available()
```

**Phase 2:** OAuth2 token caching, refresh, concurrent access:

- `test_oauth2_auth_caches_token`
- `test_oauth2_auth_refreshes_after_expiry`
- `test_oauth2_auth_concurrent_refresh_takes_one_lock`
- `test_gigachat_uses_oauth2_auth`

**Phase 3:** YandexGPT converters + transport:

- `test_yandex_request_conversion` (round-trip property: `from_yandex(to_yandex(x)) == x`)
- `test_yandex_response_conversion`
- `test_yandex_generate_calls_correct_endpoint`
- `test_yandex_generate_stream_parses_ndjson`
- `test_yandex_truncated_status_logs_warning`
- `test_yandex_missing_folder_id_raises_llm_unavailable`

### 7.2 Optional live smoke tests

`pytest -m live` — пропускается в CI, запускается локально с реальными ключами. Покрывает E2E с каждым провайдером отдельно. Не блокирует merge.

## 8. Migration

### Сценарий 1: уже запущенный prod с `DEEPSEEK_API_KEY`

После апгрейда `kb_llm` стал пакетом, но `import kb_llm` работает идентично. Auto-detection: `cerebras` пустой → `groq` пустой → ... → `deepseek` найден. **Zero migration cost.**

### Сценарий 2: добавить Cerebras

1. Получить ключ на cerebras.ai.
2. Добавить `CEREBRAS_API_KEY=...` в `.env`.
3. Перезапустить `kb_api`.
4. Auto-detection ставит cerebras первым.

### Сценарий 3: явный DeepSeek несмотря на Cerebras

`KB_LLM_PROVIDER=deepseek` в `.env`. Auto-detection пропускается.

### Сценарий 4: GigaChat

1. Регистрация в Сбере, `client_id` + `client_secret`.
2. `.env`: `GIGACHAT_CLIENT_ID=...`, `GIGACHAT_CLIENT_SECRET=...`.
3. Перезапуск.
4. На первом запросе → OAuth2 refresh, потом обычный chat completion.

### Сценарий 5: YandexGPT

1. Yandex Cloud: создать folder, выпустить API-key с правом `ai.languageModels.user`.
2. `.env`: `YANDEX_API_KEY=...`, `YANDEX_FOLDER_ID=...`.
3. `KB_LLM_PROVIDER=yandex`.
4. Запросы через native transport.

## 9. Risks

| Риск | Митигация |
|------|-----------|
| Рефакторинг ломает существующие импорты | Phase 0 отдельным коммитом, все тесты должны пройти **без правок** |
| Free tiers провайдеров меняются (API endpoints, лимиты) | Все детали в `presets.py` — обновление в одном файле; README имеет дату актуальности |
| OAuth2 refresh race condition | Thread-safe lock с double-check, явный тест на concurrent access |
| YandexGPT API изменится без предупреждения | Изолированный класс, легко отключить; интеграционный тест в `pytest -m live` |
| Auto-detection "free-first" удивит пользователей | Документация: явный `KB_LLM_PROVIDER=...` для предсказуемости |
| Слишком большой PR | Дробление на 5–6 коммитов, каждый ревёртабельный |

## 10. Open questions

Нет открытых вопросов на момент финального дизайна — все архитектурные выборы зафиксированы.

## 11. Documentation deliverables (Phase 4)

В README новый раздел «Как выбрать LLM-провайдера?»:

```
┌─ Нужна максимальная скорость + бесплатно?
│   → cerebras (если ключ есть) или groq
│
├─ Нужно высокое качество + бесплатно?
│   → gemini (Google AI Studio) или openrouter (с free моделями)
│
├─ Compliance / 152-ФЗ / данные не покидают РФ?
│   → gigachat или yandex
│
├─ Полная приватность / offline / без интернета?
│   → ollama + локальные модели (qwen2.5, saiga, vikhr, T-pro)
│
└─ Нужно gpt-4o / claude / специфическая модель?
    → openai (платно) или openrouter (один счёт на всё)
```

Таблица сравнения 14 провайдеров (10 OpenAI-compat + 4 особых) с колонками: имя, default model, free tier, language support (en/ru), notes.
