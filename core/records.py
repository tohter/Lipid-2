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


def _bracket_spans(text):
    """Интервалы скобочных вставок, вложенность учитывается.

    Нужен потому, что тире живёт не только в определениях, но и внутри
    этимологических скобок Википедии («(нид. Vleugel, нем. Flügel — крыло)»).
    Незакрытую скобку считаем открытой до конца строки: обрывок лида тоже
    обязан резаться по настоящей связке, а не по случайному тире.
    """
    spans = []
    stack = []
    for i, ch in enumerate(text):
        if ch in "([{«":
            stack.append(i)
        elif ch in ")])}»":
            if stack:
                spans.append((stack.pop(), i + 1))
    for i in stack:
        spans.append((i, len(text)))
    return spans


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
        # Разделитель ищется ВНЕ скобок. ЗАМЕРЕНО на живом канале: лида
        # «Флюгер (нид. Vleugel, нем. Flügel — крыло) — метеорологический
        # прибор…» давал ответ «Флюгер — крыло) — метеорологический прибор…",
        # потому что первое « — » стоит внутри этимологической скобки, а срез
        # по нему оставляет висячий «крыло)». Скобка — не часть определения,
        # резать по ней нельзя. Языковой список («лат.», «нид.», «нем.» …)
        # ДОКУПАТЬ НЕ СТАЛИ: он отсёк бы 0 новых случаев (замер по снимку
        # лидов), а маскирование скобок чинит класс целиком — тот же дефект
        # пришёл бы с «(англ. …)» или «(пол. …)».
        hide = _bracket_spans(out)
        cut = -1
        for sep in (" — ", " - ", " -- ", ": ", ", "):
            start = 0
            while True:
                pos = out.find(sep, start)
                if pos <= 0:
                    break
                if not any(a <= pos < b for a, b in hide):
                    cut = pos + len(sep)
                    break
                start = pos + 1
            if cut > 0:
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


def normalize_provenance(prov: Any, *, source: str = None) -> List[dict]:
    """
    ПРОИСХОЖДЕНИЕ К ЕДИНОЙ СХЕМЕ: список объектов, а не смесь строк и объектов.

    ПОЧЕМУ (очередь, п.14). В одном списке лежали И строки «wiki:Флюгер»
    (teachers/teacher_wiki.py), И объекты {source, evidence, at}
    (brain/learner.py, brain/lipid2_brain.py). Любой будущий код вида
    [(p["source"], p["at"]) for p in provenance] падает на строке — и падает
    НЕ на записи из Вики, а на первой попавшейся. Единая схема — объект:
        {"source": str, "evidence": str, "at": int|None}

    Строка разбирается по разделителю «:»: «wiki:Флюгер» ->
        {"source": "wiki", "evidence": "Флюгер"}.
    Строка без «:» — весь текст становится evidence, source берётся из
    аргумента source или пустой.
    """
    if prov is None:
        return []
    items = prov if isinstance(prov, list) else [prov]
    out: List[dict] = []
    for item in items:
        entry = _prov_entry(item, source)
        if entry is not None and entry not in out:
            out.append(entry)
    return out


def _prov_entry(item: Any, default_source: str = None) -> Optional[dict]:
    """Одна запись происхождения -> объект единой схемы (или None)."""
    if not item:
        return None
    if isinstance(item, dict):
        src = item.get("source")
        ev = item.get("evidence")
        at = item.get("at")
        # Незнакомые ключи не выбрасываем: расширение схемы не должно терять
        # данные, но базовые три поля приводим к единому виду.
        entry = dict(item)
        entry["source"] = str(src) if src else (default_source or "")
        entry["evidence"] = str(ev)[:200] if ev else ""
        if at is not None:
            try:
                entry["at"] = int(at)
            except (TypeError, ValueError):
                entry["at"] = None
        return entry
    # Строка вида "source:evidence" или просто "evidence"
    text = str(item).strip()
    if not text:
        return None
    if ":" in text:
        src, _, ev = text.partition(":")
        return {"source": src.strip() or (default_source or ""),
                "evidence": ev.strip()[:200], "at": None}
    return {"source": default_source or "", "evidence": text[:200], "at": None}


# СТАРШИНСТВО ДИАГНОЗОВ «не знаю» (очередь, п.15). Чем больше число, тем
# сильнее знание О НЕЗНАНИИ и тем меньше право его снять.
#   asked_empty — спросили честным вопросом и не нашли (сильнейшее: снимать
#                 его повторным спрашиванием = сжигать минуты на запрос);
#   action_no_profile — действие ждёт K/M/S (знание есть, профиля нет);
#   not_asked — не спрашивали (слабое: слово ОБЯЗАНО вернуться в очередь);
#   (нет статуса) — самое слабое (записи старше самого поля status).
UNKNOWN_STATUS_RANK = {"asked_empty": 3, "action_no_profile": 2,
                       "not_asked": 1}


def unknown_status_rank(status) -> int:
    """Сила диагноза «не знаю»; неизвестный/пустой статус — 0."""
    return UNKNOWN_STATUS_RANK.get(str(status), 0)


# ── ВЕС ФАКТА (УВЕРЕННОСТЬ) ═ ФОРМУЛА ОДНА, ЖИВЁТ ЗДЕСЬ ──────────────────
# Отношения, у которых верхняя половина шкалы (90..220), а не нижняя
# (60..160). Смысл: «что может / каким бывает» наблюдает текст напрямую,
# иерархию же приносят каналы знания, и шала на ней стоит дороже.
CONFIDENT_RELATIONS = ("CAN_DO", "HAS_PROPERTY")
# Прирост за повторное СВИДЕТЕЛЬСТВО (observations.support). Потолок 40:
# десятикратное повторение одного и того же рассказа не должно превращать
# догадку в неоспоримый факт — иначе один заученный текст перекрыл бы
# онтологию, проверенную двумя источниками.
SUPPORT_STEP = 10
SUPPORT_CAP = 40
# Делитель для отрицания. Отрицание — НЕ «знания нет», но и не уверенность:
# «кошка не боится воды» свидетельствует о связи слабее, чем утверждение.
NEGATION_DIVISOR = 3.0


def fact_weight(confidence: float, negated: Optional[bool] = None,
                relation: str = "", support: int = 1) -> int:
    """
    ЧИСЛО УВЕРЕННОСТИ в факте (0..255) — ЕДИНАЯ формула проекта.

    ЗАЧЕМ ОНА ПЕРЕЕХАЛА СЮДА (замер этой сессии). Формула жила приватным
    методом brain/learner.py::_weight. Из-за этого число «weight» не умел
    посчитать НИКТО, кроме обучающего, и оно НИКОГДА не попадало в
    observations — ЗАМЕРЕНО на живом мире: ключи записи head/relation/tail/
    confidence/negated/evidence/source/support/first_seen, ключа `weight` НЕТ.
    Следствие: brain/lipid2_brain.py::experience_conflicts() читает
    `v.get("weight", 0)` из этих записей, всегда получает 0 и никогда не
    выдаёт диагноз ontology_without_experience — канал, который выглядит
    работающим (тесты на него есть), был МЁРТВ. Формула в одном месте и
    доступная лёгким модулем — то, чего не хватало, чтобы её записали.

    Формула ПЕРЕНЕСЕНА БЕЗ ИЗМЕНЕНИЯ ЧИСЕЛ (brain/learner.py::_weight):
        CAN_DO / HAS_PROPERTY : 90 + 130*conf
        прочие                : 60 + 100*conf
        + min(40, 10*(support-1))
        при negated           : /3
       /clamp 8..255, int/
    Любая правка числа обязана идти от ЗАМЕРА и меняться ТОЛЬКО здесь.

    relation принимает ИМЯ отношения ("CAN_DO"), а не RelationType: лёгкий
    канал (системный python3, без numpy/torch) обязан считать тот же вес, что
    и ядро, иначе два писателя начнут расходятся числами.
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    rel = str(relation or "").upper()
    base = (90.0 + 130.0 * conf) if rel in CONFIDENT_RELATIONS \
        else (60.0 + 100.0 * conf)
    try:
        sup = int(support)
    except (TypeError, ValueError):
        sup = 1
    base += min(SUPPORT_CAP, SUPPORT_STEP * max(0, sup - 1))
    if negated:
        base /= NEGATION_DIVISOR
    return int(max(8, min(255, base)))


def observation_weight(obs: Optional[dict]) -> int:
    """
    УВЕРЕННОСТЬ в наблюдении (0..255) — В-store или ВЫВЕДЕННАЯ.

    ЗАЧЕМ ОТДЕЛЬНАЯ ФУНКЦИЯ, а не `obs.get("weight", 0)`. Ключ `weight` есть
    только у записей, написанных после этой правки; живой lipid_world.json
    старше него. Прямое .get(..., 0) на старых записях отдаёт НОЛЬ — и читатель
    делает вывод «факт совсем неуверенный», хотя правдивый ответ другой:
    «число не записано, но его надо вычислить». Это ровно тот класс лжи, что
    убил канал experience_conflicts (он читал weight из observations,
    вечно получал 0 и никогда не находил расхождение — ЗАМЕРЕНО).

    Поэтому: есть записанное число — берём его; нет — выводим из
    (confidence, negated, relation, support) ЕДИНОЙ формулой. Ноль возможен
    только как настоящее 0, а не как «не спросили».
    """
    if not isinstance(obs, dict):
        return 0
    raw = obs.get("weight")
    if isinstance(raw, bool):
        raw = None                      # True/False — не вес, а поломка данных
    if raw is not None:
        try:
            return max(0, min(255, int(raw)))
        except (TypeError, ValueError):
            pass                          # кривое число — выводим, а не нулим
    return fact_weight(obs.get("confidence", 0.5), obs.get("negated"),
                       str(obs.get("relation", "")),
                       int(obs.get("support", 1) or 1))


def merge_unknown_entry(prev: Optional[dict], new: Optional[dict]) -> dict:
    """
    СЛИЯНИЕ двух записей unknown_words ОДНОГО слова (очередь, п.15).

    ПОЧЕМУ. При слиянии двух снимков мира статус выбирался ФАКТИЧЕСКИ по
    порядку чтения (кто последний — тот и прав), а count брали max. Из-за
    этого правильный диагноз asked_empty мог быть завален старым not_asked,
    и слово возвращалось в очередь урока — или наоборот, снимался неверный
    asked_empty с глагола. Оба направления ЗАМЕРЕНЫ в проекте как ошибки.

    Правило: остаётся САМЫЙ СИЛЬНЫЙ диагноз (см. UNKNOWN_STATUS_RANK), а
    count — максимум. При равной силе побеждает новый (у него свежее
    status_note/reason).
    """
    prev = prev if isinstance(prev, dict) else {}
    new = new if isinstance(new, dict) else {}
    if not prev:
        return dict(new)
    if not new:
        return dict(prev)
    merged = dict(prev)
    # count — максимум: счётчик обращений не зависит от порядка снимков.
    merged["count"] = max(int(prev.get("count", 1)), int(new.get("count", 1)))
    rp, rn = unknown_status_rank(prev.get("status")), unknown_status_rank(new.get("status"))
    if rn > rp:
        winner, loser = new, prev
    elif rn < rp:
        winner, loser = prev, new
    else:
        winner, loser = new, prev          # равная сила — свежее важнее
    for k, v in winner.items():
        if k == "count":
            continue
        if v not in (None, "", [], {}):
            merged[k] = v
    # Поля проигравшего, которых нет у победителя, НЕ теряем: reason/seen_as
    # могут нести факты, которых у победителя нет вообще.
    for k, v in loser.items():
        if k == "count" or k in merged:
            continue
        if v not in (None, "", [], {}):
            merged[k] = v
    return merged


def merge_entity(prev: Optional[dict], new: dict) -> dict:
    """
    СЛИЯНИЕ ДВУХ ЗАПИСЕЙ ОДНОЙ СУЩНОСТИ. ОДНА РЕАЛИЗАЦИЯ ДЛЯ ВСЕХ.

    Правило: новое имеет право ДОБАВИТЬ, но не имеет права СТЕРЕТЬ.
    Списки (hierarchy/properties/actions/delta_properties) объединяются без
    повторов, скалярное поле заменяется только если было пустым.

    Почему это вынесено сюда: запись об одной сущности приходит из четырёх мест
    (base/derived от локалки, Википедия, ответ пользователя, извлечение из
    текста), и в трёх из них раньше был СВОЙ вариант слияния. Каждый терял
    данные по-своему: `world[concept] = base` перезаписывал насыщенную запись
    вырожденной, а порядок вызовов продюсера стирал иерархию.

    Здесь же чистится иерархия — с ОБЕИХ сторон. Чистить только «новое»
    бессмысленно: тавтология, уже лежащая в файле, пережила бы сколько угодно
    правок (замерено на реальном lipid_world.json).
    """
    if not isinstance(new, dict):
        return prev if isinstance(prev, dict) else {}
    merged = dict(prev) if isinstance(prev, dict) else {}
    # "aliases" обязан быть здесь: без этого второе имя сущности терялось при
    # первом же слиянии (правило «не перезаписывать непустое» оставило бы в
    # файле то написание, которое попало раньше, а вновь выученное исчезло бы
    # молча).
    LISTS = ("hierarchy", "properties", "actions", "delta_properties",
             "aliases")
    # ПОРЯДОК КАНАЛОВ ПРИ СЛИЯНИИ ИЕРАРХИИ (п.A-остаток3).
    # Иерархия — НЕ список свойств: её первый элемент становится единственным
    # родителем узла (IS_A в ядре, «X — это Y» в ответе). Прежнее правило
    # «новое дописывается в хвост» значило: родом станет категория того
    # канала, который УСПЕЛ ответить первым, а того, кто ответил раньше,
    # сдвинет на вторую позицию. ЗАМЕР (реальные записи двух каналов):
    #     Вики->локалка : [0] = 'ветроуказатели'
    #     локалка->Вики : [0] = 'метеорологические приборы'
    # один и тот же флюгер, РАЗНЫЙ РОДИТЕЛЬ В ЯДРЕ.
    # Ранги взяты не из воздуха и не из CHANNEL_CONFIDENCE (он про вес синапса,
    # здесь нужен вопрос «чей ПОРЯДОК ранжирован»):
    #   user — человек называет род словами («пчела — летаемое насекомое»);
    #   llm  — иерархия ЛОКАЛКИ ранжирована: 15 из 18 цепочек заканчиваются
    #          верхним корнем «*объект*», то есть направление «от ближнего к
    #          дальнему» внутри неё есть;
    #   text — единичная категория извлечения, ранжировать нечего;
    #   wiki — категории алфавитны (14 из 22 списков строго алфавитны после
    #          отсева служебных) и в редакционный род не попали НИ РАЗУ (0/7).
    # Ранжирование по СОДЕРЖАНИЮ (граф родителей категорий) этой замены НЕ
    # заменена: оно проверено и дало 1 совпадение из 5 (pA_rank.py), то есть
    # не лучше алфавита — см. test_experience_choice_contract.
    HIERARCHY_RANK = {"user": 0, "llm": 1, "text": 2, "wiki": 3}

    def _rank(prov):
        """
        Лучший (наименьший) ранг канала в происхождении записи, None если
        метки канала нет. Записей без метки в мире большинство (текстовые,
        старые файлы) — для них прежний порядок «ранее записанное главнее»,
        иначе правка молча переставила бы половину Вселенной.
        """
        ranks = [HIERARCHY_RANK[str(p.get("source") or "").lower()]
                 for p in normalize_provenance(prov)
                 if str(p.get("source") or "").lower() in HIERARCHY_RANK]
        return min(ranks) if ranks else None

    rank_prev = _rank(merged.get("provenance"))
    rank_new = _rank(new.get("provenance"))
    # Новое становится в начало ТОЛЬКО когда его канал ранжирован лучше
    # записанного. При равных рангах (ответ модели поверх ответа модели)
    # и при неизвестных — прежнее правило: главное то, что уже лежит.
    hier_first = (rank_new is not None
                  and (rank_prev is None or rank_new < rank_prev))
    for k in LISTS:
        have = list(merged.get(k) or [])
        add = new.get(k) or []
        if not isinstance(add, list):
            add = [add]
        if k == "hierarchy" and hier_first:
            have, add = add, have
        for x in add:
            if not x:
                continue
            if x in have:
                # Позиция уже записанного НЕ меняется. Соблазн «при дубле
                # поднять категорию в начало» ЗАМЕРЕНО и убит: дублем чаще
                # всего бывает как раз верхний корень («объекты»), и он
                # переехал бы в hierarchy[0] — грубое множество стало бы родом.
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
    # Обе стороны приводятся к ЕДИНОЙ схеме (п.14): иначе сравнение
    # «p not in prov» не ловило бы дубль строки и объекта об одном и том же.
    prov = normalize_provenance(merged.get("provenance"))
    for p in normalize_provenance(new.get("provenance")):
        if p not in prov:
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
