# Track B — свип «число кандидатов ↔ латентность/качество» + ревизия p95-бюджета (право)

- **Дата:** 2026-07-02
- **Статус:** дизайн утверждён, ждёт implementation-плана
- **Скоуп:** без обучения, без GPU. Только измерение зависимости качества и CPU-латентности
  reranker'а от `KB_RERANK_CANDIDATES` и вывод честного, достижимого p95-бюджета.
- **Связанное:** Phase 0 headroom probe (#597), Track A frozen-гейт качества
  (`tests/test_eval_frozen_pravo.py`, `data/eval/ci_thresholds_pravo.json`, #620),
  латентностные скрипты `scripts/bench_reranker.py`, `scripts/quantize_reranker.py`.

## 1. Контекст и мотивация

Track A закрепил доменный выигрыш teacher-reranker'а `BAAI/bge-reranker-v2-m3` на
`golden_pravo_natural` (n=36): **+0.111 hit@1 / +0.085 mrr@5** над bi-encoder-базой
(Phase 0.5 clean re-measure).
Латентность там **осознанно вынесена за скоуп** (anti-scope §2 Track A): teacher на CPU
даёт p95 ≈ 1.2 с при 20 кандидатах, тогда как исходный бюджет — 200 мс.

Существующая латентностная линия целится в **отсутствующего** дистиллированного
студента (`var/models/kbai-reranker-ru`) и требует `var/data/rerank/pairs.jsonl`,
которого тоже нет; полноценный retrieval→rerank прогон упирался в отсутствующий стор.
Оба скрипта (`bench_reranker.py`, `quantize_reranker.py`) в текущем виде не
запускаются. При этом:

- `BAAI/bge-reranker-v2-m3` **есть в HF-кэше** → продакшн-reranker реально гоняется на
  этой CPU-машине.
- Стор `pravo_public.sqlite` (6141 док / 14231 chunk / embedder `st` / dim 384) **есть**
  в соседнем рабочем каталоге и **совпадает** с сигнатурой `golden_pravo_natural`.

Продакшн-рычаг латентности — `KB_RERANK_CANDIDATES` (дефолт 20, `app/services/kb_rerank.py:88`):
это размер шорт-листа, который би-энкодер достаёт и который cross-encoder скорит целиком.
Число форвардов bge = число кандидатов → **латентность ∝ кандидатам**. Меньше кандидатов =
быстрее, но золотой chunk за пределами шорт-листа вообще не достаётся (падает recall).

**Пробел, который закрывает Track B:** нет данных о том, как качество и латентность
меняются с числом кандидатов, поэтому «200 мс» — это бюджет, взятый без опоры на
доменные цифры. Приоритет (утверждён пользователем): **сохранить качество** — предложить
новый достижимый p95-бюджет на числе кандидатов, где качество выходит на плато, а не
резать качество ради недостижимого потолка.

## 2. Цель и критерии приёмки

Готово, когда:

1. Закоммичен воспроизводимый фикстур `data/eval/rerank_sweep_pravo.json` со «сырым
   материалом» свипа: на каждый вопрос — упорядоченный top-20 шорт-лист би-энкодера
   (chunk-ключи) + скор teacher'а (bge) по каждому кандидату + relevant. Числа получены
   реальными моделями офлайн, `_sig` совпадает с golden.
2. Чистый офлайн-модуль `app/eval/candidate_sweep.py` детерминированно (без модели)
   реконструирует ранжирование «реранк top-k» для любого k≤20 и считает метрики
   base/teacher. Покрыт юнит-тестами (happy + края).
3. Записана в runbook таблица Pareto: для k ∈ {1,2,3,5,8,10,12,16,20} — hit@1, hit@3,
   hit@5, mrr@5, recall@10 (качество) и p50/p95 CPU-латентность (реальный bge,
   single-process). Отмечена точка перегиба (knee).
4. Записан **ревизованный p95-бюджет** с обоснованием (какое k держит ≈весь выигрыш
   Track A и какую латентность это стоит). Дефолт `KB_RERANK_CANDIDATES` меняется
   **только если** данные покажут плато качества строго ниже 20; иначе дефолт остаётся,
   правится лишь бюджет `bench_reranker.py`.
5. `pytest -q` (новые тесты), `ruff`, `black`, `mypy` на новом модуле — зелёные.

**Вне скоупа (anti-scope):**
- Обучение/дистилляция студента, ONNX-экспорт, квантизация (это Track B'/Phase 1, отдельно).
- Изменение Track A frozen-гейта или его порогов — свип **аддитивен**.
- Изменение публичных API (`/api/kb/*`, `/api/v1/*`), формата golden, рантайм-поведения
  reranker'а (кроме, возможно, значения дефолта `KB_RERANK_CANDIDATES`).
- Изменение `bench_reranker.py`/`quantize_reranker.py` по существу (только, возможно,
  константа бюджета в `bench_reranker.py`, если п.4 её пересматривает).

## 3. Архитектура

Один инструментированный проход модели захватывает «сырьё», из которого обе оси
(качество и латентность) выводятся отдельно; качество — чисто офлайн и в CI.

### 3.1 Поток данных

```
офлайн (раз, локально, реальные модели, single-process):
  pravo_public.sqlite + golden_pravo_natural.jsonl
        │  scripts/sweep_rerank_candidates.py
        │  на каждый вопрос: store.search(top_k=20) -> шорт-лист (ключи+тексты),
        │                    bge.predict((q, text)) -> teacher-скор на кандидата
        ▼
  data/eval/rerank_sweep_pravo.json   (_sig + items[{relevant, shortlist_keys, teacher_scores}])
        │
   ┌────┴─────────────────────────────────────────────┐
   ▼                                                    ▼
качество (чисто офлайн, БЕЗ модели):          латентность (реальный bge, CPU):
app/eval/candidate_sweep.py                   scripts/sweep_rerank_candidates.py --latency
  rerank_topk(keys, scores, k) = sort            для каждого k: p50/p95 над
    shortlist[:k] по teacher desc                 захваченными (q, text) top-k
  sweep_quality(items, ks) -> {k: base/teacher}   single-process, с прогревом
        │                                           │
        └─────────────────┬─────────────────────────┘
                          ▼
        runbook: таблица Pareto + knee + ревизованный p95-бюджет
```

### 3.2 Ключевая семантика (почему офлайн-реконструкция корректна)

В проде (`app/services/kb_rerank.py` + `app/eval/adapter.py::_reranking_search`):
`candidates=k` → `store.search(top_k=k)` достаёт **ровно top-k** би-энкодера →
`rerank_hits` скорит все k → сортирует по `(score, -index)` desc (`rerank.py:141`) →
возвращает top_n. Значит при `candidates=k` возвращаемое ранжирование = `shortlist[:k]`,
переотсортированный по teacher-скору. Кандидаты за позицией k **не достаются вообще**
(не в шорт-листе), поэтому при малом k золото за его пределами — законный промах
(так же ведёт себя прод). Захватив top-20 шорт-лист + teacher-скор на кандидата, мы
воспроизводим ЛЮБОЙ k≤20 точным зеркалом прод-тай-брейка — без повторной загрузки модели.

> **Почему нельзя переиспользовать `frozen_pravo_natural.json` (Track A).** Он хранит
> ОДНУ рабочую точку: `base_ranked` = top-10 би-энкодера, `teacher_ranked` = реранк
> top-20, усечённый до top-10. Множества ключей в них расходятся во всех 36 items
> (teacher видел более широкий шорт-лист), скоры кандидатов не сохранены — из одной
> точки Pareto не построить. Track B нужен более богатый захват (top-20 ключи + скоры).

### 3.3 Компоненты

**1. `app/eval/candidate_sweep.py`** (новый, чистая логика — без I/O и без модели)
- `rerank_topk(shortlist_keys: Sequence[str], teacher_scores: Sequence[float], k: int) -> list[str]`
  — вернуть `shortlist_keys[:k]`, отсортированный по соответствующему `teacher_scores`
  убыв. с тай-брейком по исходной позиции (стабильно, зеркалит `rerank.py:141`).
- `base_topk(shortlist_keys, k) -> list[str]` — просто `shortlist_keys[:k]` (без реранка).
- `sweep_quality(items, ks) -> dict[int, dict[str, dict[str, float]]]`
  — на каждый k агрегировать метрики `base` и `teacher` через `app/eval/metrics`
  (`score_item`/`aggregate`, как `pravo_gate.aggregate_side`). Ключи item:
  `relevant`, `shortlist_keys`, `teacher_scores`.
- Без побочных эффектов → CI-able и юнит-тестируемо.

**2. `scripts/sweep_rerank_candidates.py`** (новый; стор + модель; single-process)
- Захват (дефолт-режим): env-пиннинг (см. §6), `store.search(q, top_k=N)` (N=20),
  `bge.predict([(q, text)…])` → на вопрос `{relevant, shortlist_keys[:N], teacher_scores[:N]}`.
  Пишет `data/eval/rerank_sweep_pravo.json` (+ `_sig` через `adapter.compute_signature`).
  Печатает таблицу качества по k (переиспользуя `candidate_sweep.sweep_quality`).
- Режим `--latency`: для каждого k из свипа мерит p50/p95 реального `bge.predict` над
  захваченными top-k парами `(q, text)`, single-process, с прогревом (переиспользуя
  `bench_reranker.measure`/`percentile`). Печатает таблицу латентности + PASS/FAIL к
  указанному `--budget-ms`.
- Аргументы: `--store` (дефолт `var/data/pravo_public.sqlite`), `--golden`
  (дефолт `data/eval/golden_pravo_natural.jsonl`), `--out`
  (дефолт `data/eval/rerank_sweep_pravo.json`), `--shortlist` (N, дефолт 20),
  `--ks` (дефолт `1,2,3,5,8,10,12,16,20`), `--latency`, `--budget-ms`, `--queries`.

**3. `data/eval/rerank_sweep_pravo.json`** (новый, коммитимый фикстур)
```json
{
  "_sig": {"doc_count": 6141, "max_chunk_id": 14231, "embedder_name": "st", "dim": 384},
  "_measured": {"date": "2026-07-02", "shortlist": 20, "ks": [1,2,3,5,8,10,12,16,20]},
  "items": [
    {
      "relevant": ["ГК_РФ_ч.1__a00000:0", "ГК_РФ_ч.1__a00000:1"],
      "shortlist_keys": ["<top-20 chunk keys, bi-encoder order>"],
      "teacher_scores": [0.98, 0.11, 0.42, "..."]
    }
  ]
}
```
Тексты кандидатов в фикстур **не** коммитим (объёмно и не нужно для качества); латентность
мерит по свежему `store.search` в момент прогона.

**4. `tests/test_candidate_sweep.py`** (дефолтный pytest, без модели)
Юнит-тесты чистой логики через мини-фикстуры:
- **happy:** 2 вопроса, teacher поднимает золото с позиции 2→1 → `sweep_quality`
  показывает teacher hit@1 > base hit@1 на достаточном k.
- **edge-1 (k>len):** k больше длины шорт-листа → берётся весь шорт-лист, без ошибки.
- **edge-2 (k=1):** возвращается ровно 1 ключ; метрики считаются.
- **edge-3 (ties):** равные teacher-скоры → тай-брейк по исходной позиции детерминирован.

**5. Апдейт `docs/superpowers/runbooks/2026-06-15-pravo-reranker-headroom.md`**
Секция «Candidate/latency sweep (Track B, 2026-07-02)»: таблица Pareto (k ↔ качество ↔
p50/p95), точка перегиба, ревизованный p95-бюджет с обоснованием, команда воспроизведения.

## 4. Тестирование (TDD)

- Чистая логика `candidate_sweep` — детерминированные юнит-тесты (§3.3 п.4), пишутся
  ДО модуля, сначала падают (`ModuleNotFoundError`), потом зеленеют.
- Захват/латентность (`scripts/sweep_rerank_candidates.py`) верифицируются ручным
  прогоном на реальном сторе+модели (это и есть источник чисел для runbook); чистые
  хелперы выбора k/тайминга переиспользуются из `bench_reranker` (импорт, не правка).

## 5. План коммитов (маленькие, Conventional Commits)

1. `feat(eval): pure candidate-sweep logic (rerank top-k reconstruction)`
   — `app/eval/candidate_sweep.py` + `tests/test_candidate_sweep.py`.
2. `feat(eval): instrumented candidate/latency sweep script + fixture`
   — `scripts/sweep_rerank_candidates.py` + сгенерированный `data/eval/rerank_sweep_pravo.json`.
3. `docs(eval): Track B candidate/latency Pareto + revised p95 budget`
   — этот спек + секция в runbook; при необходимости правка константы бюджета в
   `bench_reranker.py` и/или дефолта `KB_RERANK_CANDIDATES` (с обоснованием из данных).

## 6. Воспроизведение (Windows, env)

Стор в соседнем каталоге; на шаге реализации он копируется в
`var/data/pravo_public.sqlite` этого worktree (var/ не трекается), чтобы дефолт `--store`
работал без абсолютного пути.

```powershell
$env:KB_MVP_DB_PATH       = "var/data/pravo_public.sqlite"
$env:KB_EMBEDDINGS_BACKEND = "st"
$env:ST_EMBED_MODEL        = "intfloat/multilingual-e5-small"
$env:VECTOR_E5_PREFIX      = "1"
$env:KB_RERANK_ENABLED     = "1"
$env:KB_RERANK_MODEL       = "BAAI/bge-reranker-v2-m3"
py -3.13 -m scripts.sweep_rerank_candidates            # захват -> фикстур + таблица качества
py -3.13 -m scripts.sweep_rerank_candidates --latency  # + таблица p50/p95 по k
```

## 7. Риски и митигации

- **Латентность single-process — иначе p95 некорректен.** Параллельные torch-процессы
  на одном CPU искажают тайминг. Мера: замеры строго последовательны (один процесс),
  никакого workflow-фан-аута на измерениях.
- **Замер на CPU может быть оборван** (фоновые задачи на этой машине умирают ~10 мин):
  захват top-20 по 36 вопросам = 36×20 форвардов bge ≈ пара минут + загрузка модели —
  влезает; при срыве скрипт идемпотентен (перезаписывает фикстур). Латентностный режим
  разбит по k и печатает по мере готовности (частичный результат уцелеет).
- **Малый n=36.** Кривая качества показывает тренд по k на ЭТОМ наборе, не статзначимость
  на популяции. Достаточно для выбора рабочей точки и бюджета; так же ограничен Track A.
- **k>20 не покрыт** (фикстур — top-20). Это ок: продакшн-дефолт кандидатов = 20, а при
  near-ceiling recall (Track A: base hit@10 0.944) расширение шорт-листа даёт ~0 качества
  при линейном росте латентности — интересная зона свипа это k≤20, особенно k≤10.
- **Тай-брейк должен зеркалить прод.** `rerank_topk` обязан повторить `(score, -index)`
  desc из `rerank.py:141`, иначе офлайн-числа разойдутся с живым reranker'ом. Проверяется
  edge-3 (ties) и (опционально) точечной сверкой одного вопроса против живого прогона.
