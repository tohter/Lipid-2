"""
core/records.py — правила ЗАПИСИ и СЛИЯНИЯ записей об одной сущности.

Здесь живёт ОДНО место, где решается:
  * что вообще имеет право называться знанием (is_empty_entity);
  * какая категория является категорией, а какая — формой самого слова или
    служебным мусором Википедии (meaningful_categories);
  * что имеет право перезаписать известное (merge_entity);
  * как отсекает само обозначаемое слово из пояснения (strip_self_name).

Модуль чистый: stdlib + ленивый pymorphy3. Настоящее обоснование (ЗАМЕРЕНО, а
не «архитектурно правильно»): правило слияния нужно каналам знания, которые
обязаны работать на ЛЮБОМ python3. На системном python3 этой машины нет
numpy/scipy/torch/sentence_transformers/pymorphy3 — и import torch не происходит
вообще, потому что encoder.py обёрнут в try/except, а SentenceTransformer
импортируется лениво внутри метода. Тяжёлый стек включается в момент
КОДИРОВАНИЯ, а не импорта — и потому лёгкий канал обязан оставаться лёгким по
факту, а не по надежде.

pymorphy3 подключается лениво и по той же причине: без него сравнение форм
слов выродилось бы в отсечение окончаний, а оно (ЗАМЕРЕНО) путает
«кошка»/«кошачьи» — настоящую категорию — и не сводит «сеть»/«сети».
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Служебные категории Википедии. Это НЕ знание о предмете, а разметка статьи:
# «страницы значений», «требует викификации», «заготовка». ЗАМЕРЕНО в живом
# lipid_world.json: у сущности «ввод» иерархия состояла из
# «Страницы значений по алфавиту» — то есть ответ на «что такое ввод» был
# описанием того, как оформлена вики-страница.
_SERVICE_CATEGORY_RE = re.compile(
    r"(страниц[ыы] значени|значени[йи] по алфавиту|перенаправлени"
    r"|требующ[иие]е?\s+(викификаци|переписывания|уточнения|источников)"
    r"|стать[ыии].*(викификаци|переписывания|уточнения|без источников|к улучшению)"
    r"|заготовк[ии] (статей|статьи)|некатегоризированн|многозначн"
    r"|дисамбиг|ссылка[ии] на|неавторитетн|необходимо? проверит)",
    re.I,
)

# Слова, которые поясняют связь, но сами знания не содержат.
_LINK_WORDS = {"это", "есть", "является", "означает", "так", "то", "же", "или"}

_MORPH: Any = None
_MORPH_TRIED = False
_LEMMA_CACHE: Dict[str, str] = {}


def _morph():
    """
    pymorphy3 — один раз, лениво, с ГРОМКИМ отказом.

    Молчание здесь стоило бы тихой деградации качества: без морфологии
    same_word переходит на эвристику (см. _lemma_simple), и часть полезных
    категорий начинает считаться «формой самого слова». Пользователь обязан
    видеть, что сравнение форм стало грубым, а не получать менее правильный
    ответ с тем же лицом.
    """
    global _MORPH, _MORPH_TRIED
    if _MORPH_TRIED:
        return _MORPH
    _MORPH_TRIED = True
    try:
        import pymorphy3                            # type: ignore
        _MORPH = pymorphy3.MorphAnalyzer()
    except Exception as exc:                        # pragma: no cover
        import sys
        print(f"[records] ⚠ pymorphy3 недоступен ({exc}) — сравнение форм слов "
              f"ДЕГРАДИРОВАЛО до эвристики: часть полезных категорий будет "
              f"принята за форму самого слова", file=sys.stderr, flush=True)
        _MORPH = None
    return _MORPH


def _lemma_simple(word: str) -> str:
    """
    Лемма одного слова. Пустая строка — если определить нечего.

    Без морфологии — наивное отсечение: ЗАМЕРЕНО, что оно путает
    «кошка»/«кошачьи» (оба -> «кошк») и НЕ сводит «сеть»/«сети». Поэтому
    эвристика — аварийный режим, а не второй равный способ, и она обязана быть
    объявлена в stderr (см. _morph).
    """
    w = (word or "").strip().lower()
    if not w or not re.fullmatch(r"[а-яёa-z-]+", w):
        return ""
    cached = _LEMMA_CACHE.get(w)
    if cached is not None:
        return cached
    m = _morph()
    if m is not None:
        try:
            # normal_form, а НЕ normalform: AttributeError ловился общим except и
            # МОЛЧА превращал лемму в пустую строку, а пустая лемма значит
            # «это не то же слово» — тавтология переставала отфильтровываться.
            # Проверено на pymorphy3 4.0.
            out = (m.parse(w)[0].normal_form or "").lower()
        except Exception as exc:                    # pragma: no cover
            import sys
            print(f"[records] ⚠ pymorphy3 не разбирал «{w}»: {exc} — форма слова "
                  f"останется unrecognized", file=sys.stderr, flush=True)
            out = ""
    else:
        out = w
        for suf in ("иями", "ями", "ами", "ов", "ев", "ам", "ям", "ов", "ей",
                    "ий", "ый", "ая", "яя", "ое", "ее", "ые", "ие", "ах", "ях",
                    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими",
                    "ы", "и", "а", "я", "о", "е", "у", "ю", "ь"):
            if len(w) - len(suf) >= 3 and w.endswith(suf):
                out = w[:len(w) - len(suf)]
                break
    _LEMMA_CACHE[w] = out
    return out


def _norm_phrase(text: str) -> str:
    """Сравнительная форма фразы: без регистра и лишних пробелов."""
    return " ".join((text or "").lower().split())


def same_word(a: str, b: str) -> bool:
    """
    Одно ли это слово в разных формах. ТОЛЬКО один токен с каждой стороны.

    Фразы здесь сравнивать нельзя: pymorphy разбирает первый токен фразы, и
    «Луга Ленинградской области» дало бы лемму «луга», то есть настоящую
    категорию мы выбросили бы как тавтологию.
    """
    ta = (a or "").strip().split()
    tb = (b or "").strip().split()
    if len(ta) != 1 or len(tb) != 1:
        return False
    la, lb = _lemma_simple(a), _lemma_simple(b)
    if not la or not lb:
        return False
    if la == lb:
        return True
    # Эвристика работает ТОЛЬКО без морфологии: общий префикс не короче 4 букв
    # и не короче «короткое слово минус 2». С pymorphy3 она наоборот склеивала
    # бы разные слова («кошка»/«кошачьи» -> общий префикс «кошк»).
    if _morph() is not None:
        return False
    n = min(len(la), len(lb))
    return n >= 4 and la[:min(n, 4)] == lb[:min(n, 4)]


def strip_self_name(concept: str, text: str, min_rest: int = 3) -> str:
    """
    Отрезать от пояснения САМО ОБОЗНАЧАЕМОЕ СЛОВО, если оно стоит в начале.

    «Пчела — летаемое насекомое» — полный ответ на вопрос «что такое пчела», но
    первая его половина только повторяет заголовок. Перенесённое в иерархию как
    есть, оно склеивает ответ из двух имён: ЗАМЕРЕНО в GUI-прогоне —
    «Пчела — это Пчела — летаемое насекомое».

    min_rest: сколько обязано остаться, чтобы срез был осмысленным. Для
    определения Вики просят 15 (там остаток обязан быть фразой), для ответа
    пользователя достаточно слова «прибор» — он короток по сути.
    Если после среза остаётся снова название (или только связки) — срез
    бессмысленен и возвращаем исходник: тавтологию лучше видеть, чем прятать
    за пустотой, иначе «знания нет» маскируется обрезком.
    """
    out = " ".join((text or "").split())
    concept = (concept or "").strip()
    if not out or not concept:
        return out
    for _ in range(3):            # «X X — ...» редкость, но три повтора — потолок
        first = out.split(" ", 1)[0].strip(",.:;»\"")
        if not (same_word(first, concept) or first.lower() == concept.lower()):
            break
        cut = -1
        for sep in (" — ", " - ", " -- ", ": ", ", "):
            pos = out.find(sep)
            if pos > 0:
                cut = pos + len(sep)
                break
        rest = out[cut:].strip(" ,.:;—-") if cut > 0 else ""
        core_words = [w for w in rest.replace("—", " ").split()
                      if w.lower() not in _LINK_WORDS]
        if len(rest) < min_rest or not core_words \
                or all(same_word(w, concept) for w in core_words):
            break
        out = rest
    return out


def meaningful_categories(concept: str, hierarchy) -> list:
    """
    Оставить только настоящие категории. Убираются три вида мусора.

    1. Форма самого концепта («экосистема» -> «экосистемы»). Источник —
       Википедия: категория множественного числа есть у статьи почти всегда.
       ЗАМЕРЕНО: 4 из 6 реальных сущностей имели такую иерархию, и ответ
       звучал как «Экосистема — это экосистемы»: правдоподобно, пусто.
    2. Служебные категории Вики (_SERVICE_CATEGORY_RE) — разметка статьи.
    3. Дубли, совпавшие после нормализации регистра.

    Пустой список ЛУЧШЕ тавтологии: пустая иерархия уводит обработчик в ветку
    «данных нет», а тавтология притворяется ответом.
    """
    out: List[str] = []
    if not hierarchy:
        return out
    for raw in hierarchy:
        if not isinstance(raw, str):
            continue
        cat = raw.strip()
        if not cat:
            continue
        if _SERVICE_CATEGORY_RE.search(cat):
            continue
        if same_word(cat, concept):
            continue
        if cat.lower() == concept.lower():
            continue
        norm = _norm_phrase(cat)
        if any(norm == _norm_phrase(o) for o in out):
            continue
        out.append(cat)
    return out


def is_empty_entity(rec) -> bool:
    """
    «Сущность есть в файле» и «о ней что-то известно» — РАЗНЫЕ утверждения.

    Пустая запись (только название) обязана отвечаться «знания нет». Пока она
    неотличима от знания, система отчитывается об успехе, записав одно слово, а
    пользователь получает строку из одного слова вместо ответа.
    """
    if not isinstance(rec, dict):
        return True
    for key in ("definition", "summary"):
        if str(rec.get(key) or "").strip():
            return False
    if rec.get("properties") or rec.get("actions") or rec.get("delta_properties"):
        return False
    if meaningful_categories(rec.get("concept") or "", rec.get("hierarchy")):
        return False
    return True


def action_name(action) -> str:
    """
    Имя действия из ЛЮБОЙ из двух форм, которые лежат в lipid_world.json:
    строка "бегать" и объект {"name": "бегать", "action_profile": [K, M, S]}.

    Профиль [K, M, S] — единственный носитель физического смысла действия для
    SDR-ядра, поэтому форму с профилем терять нельзя; но сравнивать действия
    надо по ИМЕНИ. Замер: у «гасить» в actions оказались {"name": "затухание"}
    и "затухание" рядом — слияние сверяло объект со строкой и видело два
    разных действия.
    """
    if isinstance(action, dict):
        return str(action.get("name") or "").strip()
    return str(action or "").strip()


def normalize_actions(actions) -> list:
    """Список действий -> без пустых и без повторов по имени (богатая форма
    сохраняется: словарь с action_profile не заменяется голой строкой)."""
    out: list = []
    index: dict = {}
    for a in (actions or []):
        name = action_name(a)
        if not name:
            continue
        low = name.lower()
        if low in index:
            # словари с профилем ценнее голой строки
            if isinstance(a, dict) and not isinstance(out[index[low]], dict):
                out[index[low]] = a
            continue
        index[low] = len(out)
        out.append(a)
    return out


def merge_entity(prev: Optional[dict], new: dict) -> dict:
    """
    СЛИЯНИЕ ДВУХ ЗАПИСЕЙ ОДНОЙ СУЩНОСТИ. ОДНА РЕАЛИЗАЦИЯ ДЛЯ ВСЕХ.

    Правило: новое имеет право ДОБАВИТЬ, но не имеет права СТЕРЕТЬ.
    Списки (hierarchy/properties/actions/delta_properties) объединяются без
    повторов, скалярное поле заменяется только если было пустым.

    Почему это вынесено сюда: запись об одной сущности приходит из четырёх мест
    (base/derived от локалки, Википедия, ответ пользователя, извлечение из
    текста), и в трёх из них раньше был СВОЙ вариант слияния. Каждый терял
    данные по-своему: `world[concept] = base` перезаписывал richer-запись
    вырожденной, а producer-derived порядок стирал иерархию.

    Здесь же чистится иерархия — с ОБЕИХ сторон. Чистить только «новое»
    бессмысленно: тавтология, уже лежащая в файле, пережила бы сколько угодно
    правок (замерено на реальном lipid_world.json).
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
            if not x:
                continue
            if x in have:
                continue
            if k == "actions" and any(action_name(x).lower() == action_name(h).lower()
                                      for h in have):
                continue      # то же действие во второй форме записи
            have.append(x)
        if have:
            merged[k] = normalize_actions(have) if k == "actions" else have
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
    # ЧИСТКА ИЕРАРХИИ (см. meaningful_categories). Concept берётся из обеих
    # записей: «новое» может прийти без него.
    concept = merged.get("concept") or new.get("concept") or ""
    if merged.get("hierarchy"):
        clean = meaningful_categories(concept, merged["hierarchy"])
        if clean:
            merged["hierarchy"] = clean
        else:
            merged.pop("hierarchy", None)
    return merged
