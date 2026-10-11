"""
core/records.py — правила слияния ЗАПИСЕЙ об одной сущности.

Чистый модуль: только stdlib. Настоящее обоснование (ЗАМЕРЕНО, а не «архитектурно
правильно»): правило слияния нужно channels знания, которые обязаны работать в
ЛЮБОМ python3. На системном python3 этой машины нет numpy/scipy/torch/
sentence_transformers/pymorphy3 — и import torch не происходит вообще, потому
что encoder.py обёрнут в try/except, а SentenceTransformer импортируется лениво
внутри метода. То есть тяжёлый стек включается в момент КОДИРОВАНИЯ, а не
импорта — и потому лёгкий канал обязан оставаться лёгким по факту, а не по
надежде: он вызывает ровно эти 30 строк и ничего больше.

Формат данных живёт отдельно от движка хранения ещё и потому, что merge_entity
проверяется тестом за миллисекунды, без модели.См. merge_entity: это ЕДИНСТВЕННОЕ место, где решается, что имеет право
перезаписать известное.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

def merge_entity(prev: Optional[dict], new: dict) -> dict:
    """
    СЛИЯНИЕ ДВУХ ЗАПИСЕЙ ОДНОЙ СУЩНОСТИ. ОДНА РЕАЛИЗАЦИЯ ДЛЯ ВСЕХ.

    Правило: новое имеет право ДОБАВИТЬ, но не имеет права СТЕРЕТЬ.
    Списки (hierarchy/properties/actions/delta_properties) объединяются без
    повторов, скалярное поле заменяется только если было пустым.

    Почему это вынесено сюда: запись об одной сущности приходит из четырёх мест
    (base/derived от локалки, Википедия, ответ пользователя, извлечение из
    текста), и в трёх из них раньше был СВОЙ вариант слияния. Каждый вариант
    терял данные по-своему: `world[concept] = base` перезаписывал richer-запись
    вырожденной, а producer-derived порядок стирал иерархию. Правило должно быть
    одно, иначе его невозможно ни прочитать, ни проверить.
    """
    if not isinstance(new, dict):
        return prev if isinstance(prev, dict) else {}
    merged = dict(prev) if isinstance(prev, dict) else {}
    LISTS = ("hierarchy", "properties", "actions", "delta_properties")
    for k in LISTS:
        have = list(merged.get(k) or [])
        add = new.get(k) or []
        if not isinstance(add, list):
            add = [add]
        for x in add:
            if x and x not in have:
                have.append(x)
        if have:
            merged[k] = have
    for k, v in new.items():
        if k in LISTS or k == "provenance":
            continue
        if v and not merged.get(k):
            merged[k] = v
    # ПРОИСХОЖДЕНИЕ: доказательства накапливаются, но не бесконечно — файл
    # читается человеком, и запись на 500 строк нечитабельна.
    prov = list(merged.get("provenance") or [])
    new_prov = new.get("provenance") or []
    for p in (new_prov if isinstance(new_prov, list) else [new_prov]):
        if p and p not in prov:
            prov.append(p)
    if prov:
        merged["provenance"] = prov[-12:]
    return merged
