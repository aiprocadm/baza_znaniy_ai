"""Build the committed curated golden set for the RAG eval harness.

Re-runnable: regenerates data/eval/golden_curated.jsonl + its corpus-signature
sidecar from the GoldenItem list below. Labels are global kb_chunks.id values
(reindex-stable). Reference answers are derived from the corpus and reviewed by
a domain owner before commit. After a real reindex, re-run to refresh the
sidecar (chunk-ids unchanged; embedder/dim flip).
"""

from __future__ import annotations

from pathlib import Path

from app.eval.adapter import compute_signature
from app.eval.dataset import GoldenItem, save_golden, write_signature
from app.services.kb_store import get_store

GOLDEN = Path("data/eval/golden_curated.jsonl")

# --- Answerable (single-chunk, multi-hop, paraphrase) ---
ANSWERABLE: list[GoldenItem] = [
    GoldenItem(
        "Какова ежемесячная стоимость услуг по договору?",
        (3,),
        "45 000 рублей в месяц; НДС не облагается (упрощённая система налогообложения у Исполнителя).",
        source="curated",
    ),
    GoldenItem(
        "В какие сроки и на каких условиях Заказчик оплачивает услуги?",
        (3, 4),
        "Ежемесячно, 100% предоплатой, в течение 5 рабочих дней с момента выставления счёта, но не ранее 5-го числа оплачиваемого месяца.",
        source="curated",
    ),
    GoldenItem(
        "За какой срок Исполнитель разрабатывает первичный пакет документов?",
        (6, 7),
        "В течение 20 рабочих дней со дня получения полной информации по Брифу.",
        source="curated",
    ),
    GoldenItem(
        "Какая неустойка предусмотрена за просрочку оплаты Заказчиком?",
        (22,),
        "Пени в размере 0,1% от суммы просроченного платежа за каждый день просрочки.",
        source="curated",
    ),
    GoldenItem(
        "До какой даты действует договор и как он продлевается?",
        (29,),
        "До 31 декабря 2025 года; продлевается на каждый следующий календарный год, если ни одна сторона не заявит о расторжении за 30 дней до окончания.",
        source="curated",
    ),
    GoldenItem(
        "Каков минимальный период оказания услуг по договору?",
        (30,),
        "3 месяца с даты подписания; при досрочном расторжении по инициативе Заказчика он оплачивает услуги за полный минимальный период (3 ежемесячных платежа).",
        source="curated",
    ),
    GoldenItem(
        "За сколько дней нужно уведомить о расторжении договора в одностороннем порядке?",
        (30,),
        "Не менее чем за 30 календарных дней до даты расторжения.",
        source="curated",
    ),
    GoldenItem(
        "Кто выступает Исполнителем и Заказчиком по договору?",
        (1,),
        "Исполнитель — ООО «ПРОМТЕХНОСФЕРА»; Заказчик — ООО «РУСКОНСТРУКТ Северо-Запад».",
        source="curated",
    ),
    GoldenItem(
        "В каком суде рассматриваются споры по договору?",
        (28, 29),
        "В Арбитражном суде города Санкт-Петербурга и Ленинградской области, если спор не урегулирован переговорами.",
        source="curated",
    ),
    GoldenItem(
        "Что относится к обстоятельствам непреодолимой силы по договору?",
        (24,),
        "Пожар, наводнение, землетрясение, забастовки, война и военные действия и иные обстоятельства вне контроля сторон.",
        source="curated",
    ),
    GoldenItem(
        "Включены ли в стоимость услуги по расследованию тяжёлых несчастных случаев?",
        (38,),
        "Нет. Расследование групповых, тяжёлых или смертельных несчастных случаев не входит в стоимость и оплачивается отдельно.",
        source="curated",
    ),
    GoldenItem(
        "Какие персональные данные сотрудников Заказчик поручает обрабатывать Исполнителю?",
        (35,),
        "ФИО, дату рождения, должность, наименование структурного подразделения и дату приёма на работу.",
        source="curated",
    ),
    GoldenItem(
        "Что произойдёт, если Заказчик в течение 10 рабочих дней не подпишет Акт и не направит отказ?",
        (21,),
        "Услуги считаются оказанными надлежащим образом и принятыми в полном объёме.",
        source="curated",
    ),
    GoldenItem(
        "Какие адреса электронной почты признаются для юридически значимой переписки?",
        (40,),
        "Со стороны Заказчика — snab@rusconstruct.com, со стороны Исполнителя — ot@otsfera.ru.",
        source="curated",
    ),
    # paraphrases (same facts, different wording → retrieval robustness)
    GoldenItem(
        "Сколько в месяц платит Заказчик по этому договору?",
        (3,),
        "45 000 рублей ежемесячно.",
        source="curated",
    ),
    GoldenItem(
        "Какие пени начисляются при несвоевременной оплате?",
        (22,),
        "0,1% от просроченной суммы за каждый день просрочки.",
        source="curated",
    ),
]

# --- Refusal probes (no relevant chunk; the system must decline) ---
REFUSALS: list[GoldenItem] = [
    # generic out-of-corpus
    GoldenItem(
        "Какая температура на поверхности Венеры?", (), "", expect_refusal=True, source="curated"
    ),
    GoldenItem("Кто победил в матче вчера вечером?", (), "", expect_refusal=True, source="curated"),
    GoldenItem(
        "Назови рецепт борща из нашей базы знаний.", (), "", expect_refusal=True, source="curated"
    ),
    # plausible-but-out-of-corpus (sound like the contract, but unanswerable from it)
    GoldenItem(
        "Какой размер банковской гарантии предусмотрен договором?",
        (),
        "",
        expect_refusal=True,
        source="curated",
    ),
    GoldenItem(
        "Какая неустойка предусмотрена для Исполнителя за нарушение сроков разработки документации?",
        (),
        "",
        expect_refusal=True,
        source="curated",
    ),
]

ITEMS = ANSWERABLE + REFUSALS


def main() -> None:
    save_golden(GOLDEN, ITEMS)
    write_signature(GOLDEN, compute_signature(get_store()))
    print(
        f"Wrote {len(ITEMS)} curated items ({len(ANSWERABLE)} answerable, "
        f"{len(REFUSALS)} refusal) + signature to {GOLDEN}"
    )


if __name__ == "__main__":
    main()
