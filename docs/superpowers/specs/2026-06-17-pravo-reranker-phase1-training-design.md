# Реранкер на pravo: Фаза 1 — обучение ученика (mr-TyDi → pravo) — Design

**Date:** 2026-06-17
**Status:** Proposed (брейнсторм 2026-06-17).
**Scope owner:** solo side-project (10–20 ч/нед).
**Parent strategy:** [own-reranker distillation](2026-06-10-own-reranker-distillation-design.md).
**Предшественник:** [Фаза 0 headroom-проба](2026-06-15-pravo-reranker-headroom-design.md) —
вердикт **условный GO** (Фаза 0.5: teacher bge-reranker-v2-m3 даёт **+11 пп hit@1 /
+8.6 пп mrr@5** над base e5-small на естественных юр-вопросах; hit@5 — у потолка,
поэтому снят как гейт).

> Этот спек охватывает **Фазу 1** — обучение ученика `kbai-reranker-ru`
> (rubert-tiny2) в две стадии: общий пред-трейн на русском mr-TyDi → доменное
> дообучение на структурно намайненных парах pravo. Латентностные гейты, int8 и
> продакшн-выкатка (Фаза 2) проектируются отдельным спеком после результата Фазы 1.

---

## 1. Контекст и мотивация

Фаза 0 ответила на главный вопрос: **headroom реранкинга на pravo реален** —
teacher обыгрывает base би-энкодер на верху ранжирования (+11 пп hit@1). Раз
потолок есть, обучение ученика осмысленно. Прошлый блокер v1/v2 — **не качество
модели, а нехватка обучающих данных**: 127–249 пар при цели 50–100k, потому что
генерация запросов гоняла LLM на CPU (~1.5 мин/запрос).

Фаза 1 снимает блокер двумя независимыми источниками данных, **оба без LLM**:

1. **Русский mr-TyDi** (`castorini/mr-tydi`, config `russian`) — натуральные русские
   вопросы, на каждый запрос 1 positive + ~30 уже намайненных hard-negatives.
   Даёт общий навык ранжирования в языке ученика. Решение брейнсторма: внешний
   датасет должен быть в языке ученика (rubert-tiny2 — русский монолингв; перенос
   с английского плох). **Ревизия при исполнении 2026-06-17:** исходно
   планировался русский mMARCO, но он оказался не стримящимися текстовыми
   тройками, а ID-джойном через корпус пассажей **5.77 ГБ** (+905 МБ троек-ID) —
   несовместимо с лёгким side-project. mr-TyDi — натуральный русский (не машинный
   перевод MS MARCO), сотни МБ, готовые hard-negatives. См. §6.
2. **Структурный майнинг pravo** из `corpus.jsonl` (6141 статья) — заголовок темы
   → запрос, статья → positive, top-k путающихся соседей из стора → hard-negatives.
   Без LLM, тысячи доменных пар.

Логика двух стадий: mr-TyDi ставит общее понятие «релевантности», pravo
специализирует под право (где Фаза 0 и нашла headroom).

## 2. Цель / не-цели

**Goal:** обучить `kbai-reranker-ru`, который на естественном pravo-golden
обыгрывает base би-энкодер на верху ранжирования (mrr@5 / hit@1), приближаясь к
потолку-teacher. Decision-gate для Фазы 2.

**Non-goals (НЕ в этом спеке):**

- Латентностные гейты, int8-квантизация, top-N рецепт (Фаза 2) — переиспользуют
  готовые `quantize_reranker.py` / `bench_reranker.py`.
- Продакшн-выкатка в `KB_RERANK_MODEL`.
- Перекрёстные ссылки как источник запросов — **отложено** (§6): Фаза 1 берёт
  только заголовки-темы (YAGNI для проверки гипотезы).
- Teacher-скоры на внешнем датасете (mr-TyDi) — **отклонено** (§6): для стадии 1 синтетических
  бинарных меток достаточно (pairwise-лоссу нужен только порядок пар).
- Слияние с mini-GPT — отклонено ещё в Фазе 0.

## 3. Архитектура решения

Две стадии обучения одного `rubert-tiny2` cross-encoder, обе через существующий
`scripts/train_reranker.py`. Новый код: два сборщика датасета + один флаг трейнера.
`quantize_reranker.py` / `bench_reranker.py` / `check_rerank_leak.py` —
переиспользуются как есть.

```
СТАДИЯ 1 (общий навык ранжирования, RU)
  castorini/mr-tydi (russian, streaming, datasets==3.6.0 + trust_remote_code)
    └─ scripts/build_mrtydi_pairs.py  [НОВЫЙ]
         → запись: {query, positive_passages[1], negative_passages[~30]} (натуральный RU)
         → подвыборка ~N запросов по seed; на запрос 1 pos + m hard-neg
         → {query, text, teacher_score} (синтетика: pos=1.0, neg=0.0)
    └─ train_reranker.py --loss pairwise --init-from cointegrated/rubert-tiny2
         → var/models/kbai-reranker-ru-stage1/

СТАДИЯ 2 (доменная адаптация, право)
  experiments/pravo_nn/data/corpus/corpus.jsonl (6141 статья)
    └─ scripts/build_pravo_pairs.py  [НОВЫЙ — структурный майнер, без LLM]
         → query=тема статьи; pos=статья; hard-neg=top-k bi-encoder из pravo-стора
         → teacher_score = bge-reranker-v2-m3 (пар мало → teacher оправдан)
         → анти-утечка против golden_pravo / golden_pravo_natural held-out
    └─ train_reranker.py --loss pairwise --init-from <stage1> --lr 1e-5
         → var/models/kbai-reranker-ru/

ОЦЕНКА (gate)
  eval_rag.py на golden_pravo_natural.jsonl:
    base (e5-small) vs student vs teacher → mrr@5 / hit@1 / recall@5
```

### 3.1 `scripts/build_mrtydi_pairs.py` (новый)

- Источник: `load_dataset("castorini/mr-tydi", "russian", split="train",
  streaming=True, trust_remote_code=True)`. Требует `datasets==3.6.0` (в 4.0+
  поддержку script-датасетов убрали; mr-TyDi грузится скриптом-обёрткой над
  parquet). Стримом — на диск целиком не материализуем.
- Схема записи: `{query, positive_passages: [{docid,text,title}],
  negative_passages: [~30×{docid,text,title}]}`. Негативы уже намайнены — своего
  майнинга не нужно.
- Подвыборка детерминированная по seed: `--limit N` запросов; на запрос берём
  `positive_passages[0]` и до `--negs m` негативов.
- Выход — формат трейнера: на запрос строки `{query, text, teacher_score}` с
  `teacher_score` = 1.0 (pos) / 0.0 (neg). `pairwise`-лоссу нужен только порядок.
- Без ML внутри (только чтение датасета) → пьюр-хелперы юнит-тестируемы без `datasets`.

### 3.2 `scripts/build_pravo_pairs.py` (новый, структурный майнер)

- Переиспользует `build_pairs` + анти-утечку из `build_rerank_dataset.py`, но
  источник запросов **структурный, без LLM**:
  - `query` = тема статьи (часть `article` после «Статья N.»);
  - `positive` = сама статья (документ pravo-стора);
  - `hard-neg` = top-k кандидатов через bi-encoder `store.search` (путающиеся
    соседние статьи — то, чего не было на 9-док корпусе).
- `teacher_score` = bge-reranker-v2-m3 по каждой паре (пар мало → прогон дёшев,
  это потолок из Фазы 0).
- Анти-утечка: held-out статьи `golden_pravo` / `golden_pravo_natural` исключаются
  из майнинга (assert-backstop, как в существующем коде).

### 3.3 `scripts/train_reranker.py` (правка: флаг `--init-from`)

Единственная правка существующего трейнера — стартовый чекпойнт параметризуется:

```python
parser.add_argument("--init-from", default=BASE_MODEL,
    help="стартовый чекпойнт: rubert-tiny2 (стадия 1) или путь к stage1 (стадия 2)")
# AutoModelForSequenceClassification.from_pretrained(args.init_from, num_labels=1)
```

Трейнер уже device-agnostic (`select_device`) → **тот же вызов** едет на CPU
(проба) и GPU (полный объём) без правок.

| | Стадия 1 (mr-TyDi) | Стадия 2 (pravo) |
|---|---|---|
| init-from | `cointegrated/rubert-tiny2` | `…/kbai-reranker-ru-stage1` |
| loss | `pairwise` | `pairwise` |
| lr | `5e-5` (дефолт) | **`1e-5`** (низкий — против забывания) |
| epochs | 1 (проба) → 2–3 (GPU) | 1–2 |
| выход | `…-ru-stage1/` | `…/kbai-reranker-ru/` |

### 3.4 Поэтапность железа (CPU-проба → GPU)

- **Шаг A (CPU, локально):** `--limit 10000` mr-TyDi + pravo-пары, 1 эпоха каждая,
  detached (правило detached-long-runs). Цель — НЕ качество, а валидация: (1)
  пайплайн зелёный end-to-end; (2) `val_pearson_vs_teacher` стадии 2 не NaN и
  растёт; (3) ученик на golden не ниже base. Это go/no-go на GPU-аренду.
- **Шаг B (GPU, аренда):** полная подвыборка (50–100k), больше эпох, та же
  команда с `--device cuda`.

## 4. Критерий успеха (decision-gate Фазы 1)

На `golden_pravo_natural.jsonl` (естественные вопросы Фазы 0.5), три модели через
`eval_rag.py`:

- **GO →** student бьёт base на **mrr@5 ≥ +0.05** ИЛИ **hit@1 ≥ +0.05**
  (метрики верха ранжирования из Фазы 0.5; hit@5 — у потолка, не гейт), И
  student ≥ base на recall@5 (не сломал отзыв). Идеал — student догоняет
  teacher (потолок +11 пп hit@1). При GO → Фаза 2 (квантизация, латентность,
  выкатка).
- **NO-GO →** student не превосходит base. Диагностика: сравнить со stage1-only
  (не съела ли стадия 2 общий навык), проверить достаточность пар.

## 5. Риски и митигации

| Риск | Митигация |
|---|---|
| Общий домен mr-TyDi (вики-вопросы) ≠ право | Стадия 2 (домен) идёт *после* и доминирует над верхом ранжирования; гейт меряет именно pravo |
| Катастрофическое забывание на стадии 2 | Низкий LR (1e-5), 1–2 эпохи; диагностика через stage1-only при NO-GO |
| CPU-проба медленная / умирает фон | `--limit 10000`, detached + Monitor (detached-long-runs); проба валидирует только пайплайн, не качество |
| Скачивание датасета | `streaming=True` + детерминированная подвыборка по seed; mr-TyDi ru — сотни МБ, на диск целиком не материализуем |
| Структурный майнинг завышает лёгкость | Гейт на естественном `golden_pravo_natural`, не на структурном golden |
| Дрейф pravo-стора vs golden sig | Sig-проверка стора перед майнингом и eval (как в Фазе 0) |

## 6. Decisions log (брейнсторм 2026-06-17)

- **Роль внешнего датасета — русский mr-TyDi**, не сырой английский MS MARCO
  (rubert-tiny2 монолингв-русский; английский переносится плохо). **Ревизия при
  исполнении 2026-06-17:** исходно — русский mMARCO; на деле он = ID-джойн через
  корпус пассажей 5.77 ГБ, не стримящиеся тройки. Пользователь выбрал лёгкий
  parquet RU IR (брейнсторм-развилка); mr-TyDi russian — натуральные русские
  вопросы, 1 pos + ~30 hard-neg на запрос, сотни МБ. Цена: `datasets==3.6.0` +
  `trust_remote_code=True` (script-датасет; в datasets 4.0+ не грузится).
- **Схема — пред-трейн mr-TyDi → дообучение pravo** (две стадии), не смешивание и
  не только-внешний: Фаза 0 показала ценность именно доменного верха ранжирования.
- **Метки стадии 1 — синтетические бинарные** (pos=1.0/neg=0.0); teacher по
  mr-TyDi отклонён — pairwise-лоссу нужен только порядок, прогон 568M teacher по
  десяткам тысяч пар не оправдан (негативы уже намайнены в датасете).
- **Метки стадии 2 — teacher bge-reranker-v2-m3** на структурно намайненных парах
  (пар мало → дёшево; это потолок из Фазы 0).
- **Железо — CPU-проба (10k) → GPU-аренда (полный объём)**: не жечь GPU-часы до
  валидации пайплайна.
- **Источник запросов pravo — только заголовки-темы**; перекрёстные ссылки
  отложены (YAGNI).
- **База — rubert-tiny2**, переиспользуем `train_reranker.py` (правка: `--init-from`).
- **Скоуп — только Фаза 1** (обучение + gate). Фаза 2 (квантизация/латентность/
  выкатка) — отдельным спеком после результата.
