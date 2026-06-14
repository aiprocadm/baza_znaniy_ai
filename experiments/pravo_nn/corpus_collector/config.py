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


CODES: tuple[CodeSpec, ...] = (
    CodeSpec("ГК РФ", "gk-rf"),  # Гражданский
    CodeSpec("УК РФ", "uk-rf"),  # Уголовный
    CodeSpec("НК РФ", "nk-rf"),  # Налоговый
    CodeSpec("ТК РФ", "tk-rf"),  # Трудовой
    CodeSpec("КоАП РФ", "koap-rf"),  # Об административных правонарушениях
    CodeSpec("ЖК РФ", "zhk-rf"),  # Жилищный
    CodeSpec("СК РФ", "sk-rf"),  # Семейный
    CodeSpec("ГПК РФ", "gpk-rf"),  # Гражданский процессуальный
    CodeSpec("УПК РФ", "upk-rf"),  # Уголовно-процессуальный
    CodeSpec("АПК РФ", "apk-rf"),  # Арбитражный процессуальный
    CodeSpec("БК РФ", "bk-rf"),  # Бюджетный
    CodeSpec("ЗК РФ", "zk-rf"),  # Земельный
    CodeSpec("УИК РФ", "uik-rf"),  # Уголовно-исполнительный
    CodeSpec("КАС РФ", "kas-rf"),  # Административного судопроизводства
    CodeSpec("ГрК РФ", "grk-rf"),  # Градостроительный
    CodeSpec("ВК РФ", "vk-rf"),  # Водный
    CodeSpec("ЛК РФ", "lk-rf"),  # Лесной
    CodeSpec("ВзК РФ", "vzk-rf"),  # Воздушный
    CodeSpec("КТМ РФ", "ktm-rf"),  # Торгового мореплавания
)

# The base of the chosen text source. Empty until the Task 3 spike decides it;
# Task 8 (fetch) sets it to the spike's committed value.
SOURCE_BASE: str = ""
