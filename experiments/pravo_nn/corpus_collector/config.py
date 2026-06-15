"""The fixed list of Russian Federation codes to collect.

`name` is the canonical short legal name (lands in the corpus). `slug` is a
filename-safe id used for the on-disk raw cache. The *source URL* is NOT here —
it depends on the source chosen by the Task 3 spike and is built in fetch.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeSpec:
    name: str  # e.g. "ГК РФ"
    slug: str  # e.g. "gk-rf"
    nd: str = ""  # ИПС registry id; "" means not yet harvested -> skipped by collect


CODES: tuple[CodeSpec, ...] = (
    CodeSpec("ГК РФ ч.1", "gk-rf-1", "102033239"),  # Гражданский, часть 1
    CodeSpec("ГК РФ ч.2", "gk-rf-2", "102039276"),  # Гражданский, часть 2
    CodeSpec("ГК РФ ч.3", "gk-rf-3", "102073578"),  # Гражданский, часть 3
    CodeSpec("ГК РФ ч.4", "gk-rf-4", "102110716"),  # Гражданский, часть 4
    CodeSpec("УК РФ", "uk-rf", "102041891"),  # Уголовный
    CodeSpec("НК РФ ч.1", "nk-rf-1", "102054722"),  # Налоговый, часть 1
    CodeSpec("НК РФ ч.2", "nk-rf-2", "102067058"),  # Налоговый, часть 2
    CodeSpec("ТК РФ", "tk-rf", "102074279"),  # Трудовой
    CodeSpec("КоАП РФ", "koap-rf", "102074277"),  # Об административных правонарушениях
    CodeSpec("ЖК РФ", "zhk-rf", "102090645"),  # Жилищный
    CodeSpec("СК РФ", "sk-rf", "102038925"),  # Семейный
    CodeSpec("ГПК РФ", "gpk-rf", "102078828"),  # Гражданский процессуальный
    CodeSpec("УПК РФ", "upk-rf", "102073942"),  # Уголовно-процессуальный
    CodeSpec("АПК РФ", "apk-rf", "102035455"),  # Арбитражный процессуальный
    CodeSpec("БК РФ", "bk-rf", "102054721"),  # Бюджетный
    CodeSpec("ЗК РФ", "zk-rf", "102073184"),  # Земельный
    CodeSpec("УИК РФ", "uik-rf", "102045146"),  # Уголовно-исполнительный
    CodeSpec("КАС РФ", "kas-rf", "102380990"),  # Административного судопроизводства
    CodeSpec("ГрК РФ", "grk-rf", "102090643"),  # Градостроительный
    CodeSpec("ВК РФ", "vk-rf", "102107048"),  # Водный (74-ФЗ 2006, действующий)
    CodeSpec("ЛК РФ", "lk-rf", "102110364"),  # Лесной
    CodeSpec("ВзК РФ", "vzk-rf", "102046246"),  # Воздушный
    CodeSpec("КТМ РФ", "ktm-rf", "102059464"),  # Торгового мореплавания
)

SOURCE_BASE: str = "http://pravo.gov.ru/proxy/ips/"
