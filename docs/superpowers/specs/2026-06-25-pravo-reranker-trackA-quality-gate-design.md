# Track A — закрепить teacher-reranker + CI-гейт качества (право)

- **Дата:** 2026-06-25
- **Статус:** дизайн утверждён, ждёт implementation-плана
- **Скоуп:** без обучения, без GPU. Только закрепление измеримого выигрыша cross-encoder
  reranker'а на доменном golden-наборе и защита его CI-гейтом.
- **Связанное:** Phase 0 headroom probe (#597), Phase 1 student training (#607),
  offline frozen public-гейт (`tests/test_eval_frozen.py`, `data/eval/ci_thresholds.json`).

## 1. Контекст и мотивация

Стратегическая цель проекта — доказуемо точнее отвечать в узком домене (право РФ /
охрана труда), чем модель из коробки. Главный рычаг качества в юридическом RAG —
ранжирование (точное попадание в нужную статью важнее красноречия).

Замеры (Phase 0, на `golden_pravo_natural`, n=36, embedder `st`/384, корпус 6141 док)
показывают, что teacher cross-encoder `BAAI/bge-reranker-v2-m3` даёт реальный
доменный выигрыш над чистым bi-encoder-ретривалом:

| метрика | base (без rerank) | teacher (bge) | дельта |
|---|---|---|---|
| hit@1 | 0.778 | 0.889 | +0.111 |
| hit@3 | 0.861 | 0.944 | +0.083 |
| mrr@5 | 0.832 | 0.917 | +0.085 |
| recall@10 | 0.694 | 0.707 | +0.013 |

Teacher уже подключён дефолтом в рантайме через env-переключатель
(`KB_RERANK_MODEL`/`KB_RERANK_ENABLED` в `app/services/kb_rerank.py`,
`RERANK_*` для `/api/v1/*`). Чего НЕ хватает: этот выигрыш ничем не защищён —
регрессия ранжирования пройдёт CI незамеченной. Track A закрывает этот пробел.

**Важное уточнение по порогам.** Существующий `data/eval/ci_thresholds.json`
(hit@1≈0.70, recall@10≈0.92) скоупнут на `golden_public` (offline frozen-гейт,
n=33) — это НЕ пороги для pravo. Track A добавляет отдельные pravo-пороги, не
смешивая их с public.

## 2. Цель и критерии приёмки

Готово, когда:

1. Свежий замер base vs teacher на `golden_pravo_natural` закоммичен (таблица в
   фикстуре + замороженные ранжирования). Числа получены реальными моделями офлайн.
2. Frozen-гейт **зелёный в дефолтном `pytest`** (без модели в CI) и проверяет
   одновременно: (а) teacher ≥ абсолютных floors, (б) teacher − base ≥ мин-дельты.
3. Живой integration-тест существует (`@pytest.mark.integration`, вне дефолтного
   CI), зелёный при ручном прогоне на реальном teacher.
4. `pytest -q`, `ruff`, `black` зелёные. Существующие floors не понижены.

**Вне скоупа (anti-scope):**
- Латентность reranker'а (сейчас p95 ≈ 1.2 с на CPU при бюджете 200 мс) — вынесена
  в отдельный будущий шаг. Не решаем здесь.
- Никакого обучения (ни student-reranker, ни LoRA).
- Не меняем рантайм-поверхность (`KB_RERANK_*` уже работает).
- Не меняем публичные API (`/api/kb/*`, `/api/v1/*`) и формат golden.

## 3. Архитектура

Зеркалит установленную в репо философию «дешёвый детерминированный frozen-гейт на
push + тяжёлый живой прогон по требованию» (как `test_eval_frozen.py` +
`ci_thresholds.json` для public).

### 3.1 Поток данных

```
офлайн (раз, локально, реальные модели):
  pravo_public.sqlite + golden_pravo_natural.jsonl
        │
        ├── make_mvp_retriever ............ base ranked top-10  ┐
        └── make_mvp_reranking_retriever .. teacher ranked top-10 ┘
        │
        ▼
  data/eval/frozen_pravo_natural.json  (+ _sig, + _measured)
        │
  ┌─────┴───────────────────────────────────────────┐
  ▼                                                   ▼
CI (дефолтный pytest, БЕЗ модели):           ручной прогон:
test_eval_frozen_pravo.py                    test_pravo_rerank_integration.py
  пересчёт метрик из frozen ranked-списков     живой teacher по корпусу
  ассерты: floors + дельты                      те же ассерты против живых чисел
```

### 3.2 Компоненты

**1. `scripts/freeze_pravo_eval.py`** (новый)
Отдельный скрипт (НЕ `--freeze`-режим в `eval_rag.py`, чтобы не трогать его
публичную CLI-поверхность). Гоняет base и teacher пайплайны офлайн через
переиспользование `app/eval/adapter.py` (`make_mvp_retriever`,
`make_mvp_reranking_retriever`) и `app/eval/retrieval_eval`. Пишет фикстур и
печатает таблицу до/после.

Аргументы: `--store` (default `var/data/pravo_public.sqlite`),
`--golden` (default `data/eval/golden_pravo_natural.jsonl`),
`--out` (default `data/eval/frozen_pravo_natural.json`), `--top-k` (default 10).

Формат фикстуры:
```json
{
  "_sig": {"doc_count": 6141, "embedder_name": "st", "dim": 384},
  "_measured": {
    "date": "2026-06-25",
    "base":    {"hit@1": 0.778, "hit@3": 0.861, "mrr@5": 0.832, "recall@10": 0.694},
    "teacher": {"hit@1": 0.889, "hit@3": 0.944, "mrr@5": 0.917, "recall@10": 0.707}
  },
  "items": [
    {
      "relevant": ["ГК_РФ_ч.1__a00000:0", "ГК_РФ_ч.1__a00000:1"],
      "base_ranked":    ["<top-10 chunk keys>"],
      "teacher_ranked": ["<top-10 chunk keys>"]
    }
  ]
}
```
Замораживаем top-10 на запрос — достаточно для hit@{1,3,5}, mrr@5, recall@10.

**2. `data/eval/ci_thresholds_pravo.json`** (новый, отдельный от public)
```json
{
  "_comment": "Pravo retrieval floors + min teacher-over-base deltas for the frozen pravo gate (tests/test_eval_frozen_pravo.py). Measured on golden_pravo_natural by scripts/freeze_pravo_eval.py. Raise (never lower) after a real re-measure.",
  "_measured_2026_06_25": { "...": "копия _measured из фикстуры для аудита" },
  "teacher_floors": { "hit@1": 0.84, "mrr@5": 0.86, "recall@10": 0.65 },
  "min_delta_over_base": { "hit@1": 0.05, "mrr@5": 0.04 }
}
```
Floors сидят ~0.05 ниже замеренного teacher, дельты — с запасом ниже фактических
(+0.111 / +0.085). **Точные значения проставляются по СВЕЖЕМУ замеру** на шаге
реализации (могут чуть сдвинуться от артефактных).

**3. `tests/test_eval_frozen_pravo.py`** (дефолтный pytest, без модели)
- Грузит фикстуру и `ci_thresholds_pravo.json`.
- Сверяет `frozen._sig` с `data/eval/golden_pravo_natural.sig.json`; mismatch → fail
  с понятным сообщением (заставляет рефризнуть при смене корпуса/golden).
- Пересчитывает hit@1/hit@3/mrr@5/recall@10 для base и teacher из замороженных
  ranked-списков через `app/eval/metrics`.
- Ассертит: teacher ≥ teacher_floors И (teacher − base) ≥ min_delta_over_base.

**4. `tests/test_pravo_rerank_integration.py`** (`@pytest.mark.integration`)
- Гоняет настоящий teacher reranker по `pravo_public.sqlite` + golden через тот же
  харнес, что и freeze-скрипт.
- Те же ассерты (floors + дельты) против живых чисел.
- Помечен integration → исключён из дефолтного CI (репо предпочитает маркеры, а не
  `skip`). Это то, что запускают перед рефризом фикстуры.

## 4. Тестирование (TDD)

Тестируем **логику гейта**, детерминированно, без модели — через подсунутые
мини-фикстуры:

- **happy-path:** фикстура, где teacher бьёт base ≥ дельты и выше floors → гейт проходит.
- **edge-1:** teacher НЕ бьёт base (или ниже floor) → гейт падает (`pytest.raises`/assert).
- **edge-2:** `_sig` не совпадает с golden sig → падает с понятным сообщением.

Integration-тест верифицируется ручным прогоном `pytest -m integration -k pravo_rerank`.

## 5. План коммитов (маленькие, Conventional Commits)

1. `feat(eval): freeze script for pravo base/teacher rankings`
   — `scripts/freeze_pravo_eval.py` + сгенерированный `frozen_pravo_natural.json`.
2. `test(eval): frozen pravo gate (floors + delta) + ci_thresholds_pravo.json`
   — `tests/test_eval_frozen_pravo.py`, пороги; happy + 2 edge.
3. `test(eval): live integration rerank check (marked)`
   — `tests/test_pravo_rerank_integration.py`.
4. `docs(eval): Track A spec + runbook note; latency explicitly deferred`
   — этот спек + заметка в runbook про рефриз и про отложенную латентность.

## 6. Риски и митигации

- **Дрейф заморозки** (frozen ranked-списки расходятся с живым reranker'ом со
  временем) → митигируется `_sig`-пиннингом и маркированным integration-тестом,
  который ре-валидирует против живой модели перед каждым рефризом.
- **Замер на CPU может быть оборван** (на этой машине фоновые задачи умирают ~10
  мин) → teacher по 36 вопросам ≈ 1–2 мин + загрузка модели, влезает; при срыве
  freeze-скрипт перезапускается с нуля (идемпотентен, дешёвый).
- **Малый n=36** → дельта hit@1 +0.111 устойчива на этом наборе, но это не
  статзначимость на популяции; гейт защищает от регрессии на ЭТОМ зафиксированном
  наборе, что и является его задачей.
