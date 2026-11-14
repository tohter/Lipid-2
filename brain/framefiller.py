"""ДУМАЛКА — заполнение ролевого каркаса из набора понятий.

Кирпич 5 (по консультации босса и Жозефины, 11.03): думалка — СЕЛЕКТОР-
ОБХОДЧИК ПО БАЗЕ, а не генератор. У неё нет источника, из которого можно
«придумать» пропущенное понятие: единственный источник заполнителя — уже
существующее ребро базы. Заполнение пропуска — это способность БАЗЫ, которую
думалка только обнаруживает.

Законы этого модуля:
  1. БАЗА — ТОЛЬКО ЧТЕНИЕ. Думалка ничего в неё не пишет (запись — дело
     подсистемы обучения). Иначе из-за петли «думалка выводит из собственных
     же догадок» накапливаются ложные рёбра.
  2. КАЖДЫЙ ЭЛЕМЕНТ КАРКАСА НЕСЁТ ОСНОВАНИЕ: EXPLICIT (дано в наборе) или
     DERIVED (выведено по ребру базы). Элемента без основания не существует.
  3. ТРИ ИСХОДА, взаимоисключающих:
       SUCCESS — путь в базе есть, каркас наполнен, всё заземлено;
       NO_PATH — пути нет: пропуск НЕ выдумывается, нехватка называется;
       ASK     — кандидаты есть, но равноправны/слабо заземлены: спросить,
                 а не выбирать произвольно.
  4. ГИПОТЕЗА НЕ БЫВАЕТ УВЕРЕННЫМ ИСХОДОМ. Элемент, заземлённый только как
     догадка, не может попасть в SUCCESS.

Модуль не знает русского: работает с леммами из базы и набора. Роль каждой
леммы в предложении и её склонение — дело lang/ru, а не думалки.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class FrameFill:
    """Результат заполнения каркаса.

    frame      — каркас {predicate, subject, object, place, time, ...};
    basis      — роль -> основание (EXPLICIT / DERIVED / HYPOTHETICAL);
    outcome    — SUCCESS | NO_PATH | ASK;
    gaps       — описание недостающих рёбер (для NO_PATH, питает покрытие);
    candidates — равноправные кандидаты (для ASK);
    unknown    — понятия набора, которых нет в базе.
    """
    frame: Dict[str, Any] = field(default_factory=dict)
    basis: Dict[str, str] = field(default_factory=dict)
    outcome: str = "NO_PATH"
    gaps: List[Tuple[str, str]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)


_EXPLICIT = "EXPLICIT"
_DERIVED = "DERIVED"
_HYPOTHETICAL = "HYPOTHETICAL"


def _verbs_over(k, noun: str) -> List[str]:
    """Глаголы, происходящие НАД существительным: heads наблюдений
    «шатать OBJECT_OF труба». Обратный ход: мир пишет наблюдение одной
    строкой, у существительного CAN_DO нет вовсе, поэтому «что делают с
    трубой» находится только движением от хвоста к голове.
    """
    if not k.knows(noun):
        return []
    out: List[str] = []
    for head, rel in k.referenced_by(noun):
        if rel == "OBJECT_OF" and head and head not in out:
            out.append(head)
    return out


def _is_verb_in_base(k, word: str) -> bool:
    """Слово — глагол, если у него ЕСТЬ объект, на который оно действует.

    Глагол-сказуемое — голова наблюдения «шатать OBJECT_OF труба»: у него
    есть объекты. Существительные, даже деятели («фонарь CAN_DO освещать»),
    таких исходящих рёбер НЕ имеют: фонарь — агент, а не действие. Поэтому
    признак глагола решает ТОЛЬКО прямое ребро «word OBJECT_OF ...».
    """
    if not k.knows(word):
        return False
    return bool(k.related(word, "OBJECT_OF"))


def _possessor_of(k, noun: str, remaining) -> Dict[str, Any]:
    """Принадлежность существительного: «труба PART_OF дом» -> possessor дом.

    Возвращает словарь {lemma: possessor} или None, если PART_OF не даёт
    понятия из набора. Только прямое ребро: иначе перенос по цепочке частей
    плодит ложную принадлежность.
    """
    for owner in k.related(noun, "PART_OF"):
        if owner in remaining:
            return {"lemma": owner}
    return None


def fill_frame(concepts: Sequence[str], knowledge) -> FrameFill:
    """Заполняет каркас из набора понятий, читая базу (knowledge).

    concepts — набор лемм (роль слова в предложении решает база);
    knowledge— StaticKnowledge; читается, НЕ пишется.
    """
    k = knowledge
    result = FrameFill()

    concepts = [str(c).strip().lower() for c in concepts if str(c).strip()]
    result.unknown = [c for c in concepts if not k.knows(c)]

    # Роли из набора: глаголы-сказуемые отдельно от существительных.
    verbs_in_set = [c for c in concepts if _is_verb_in_base(k, c)]
    nouns_in_set = [c for c in concepts if c not in verbs_in_set]
    remaining = set(nouns_in_set)

    # Существительные-деятели (что-то умеют) и существительные-объекты
    # (над ними что-то происходит).
    subj_nouns = [n for n in nouns_in_set if k.related(n, "CAN_DO")]
    obj_nouns = [n for n in nouns_in_set if _verbs_over(k, n)]

    # Кандидаты предиката: глагол V, который и деятель умеет, и над объектом
    # происходит. Предикат может быть как из набора, так и выведен.
    seen: set = set()
    for s in subj_nouns:
        for v in k.related(s, "CAN_DO"):
            for o in obj_nouns:
                if v not in _verbs_over(k, o):
                    continue
                key = (s, v, o)
                if key in seen:
                    continue
                seen.add(key)
                pred_basis = _EXPLICIT if v in verbs_in_set else _DERIVED
                result.candidates.append({
                    "subject": s, "predicate": v, "object": o,
                    "predicate_basis": pred_basis,
                })

    # Выбор исходов: ни одного кандидата — NO_PATH; ровно один — SUCCESS;
    # несколько равных — ASK.
    if result.candidates:
        # Ровно один предикат с данным субъектом и объектом — уверенно.
        best = result.candidates[0]
        if len(result.candidates) == 1:
            result.outcome = "SUCCESS"
        else:
            # Несколько равноправных: думалка спрашивает, а не выбирает.
            result.outcome = "ASK"
            return result
        result.frame["predicate"] = best["predicate"]
        result.basis["predicate"] = best["predicate_basis"]
        result.basis["subject"] = _EXPLICIT
        result.basis["object"] = _EXPLICIT
        result.frame["subject"] = best["subject"]
        remaining.discard(best["subject"])
        # Объект: простое существительное или с принадлежностью.
        obj_word = best["object"]
        remaining.discard(obj_word)
        poss = _possessor_of(k, obj_word, remaining)
        if poss is not None:
            result.frame["object"] = {"lemma": obj_word, "possessor": poss}
            result.basis["possessor"] = _DERIVED
            remaining.discard(poss["lemma"])
        else:
            result.frame["object"] = obj_word
        # Место и время по рёбрам базы от оставшихся понятий набора.
        place, time = _place_time(k, best, remaining)
        if place is not None:
            result.frame["place"] = place
            result.basis["place"] = _DERIVED
            remaining.discard(place)
        if time is not None:
            result.frame["time"] = time
            result.basis["time"] = _DERIVED
            remaining.discard(time)
        return result

    # Нет кандидатов. Построить честный ЧАСТИЧНЫЙ каркас из того, что
    # реально выводится, и назвать нехватку. Пропуск не выдумывается.
    # Предикат из набора (явный) остаётся в каркасе.
    for v in verbs_in_set:
        result.frame["predicate"] = v
        result.basis["predicate"] = _EXPLICIT
        break
    # Объект и его принадлежность, если выводимы.
    for o in obj_nouns:
        result.frame["object"] = o
        result.basis["object"] = _EXPLICIT
        remaining.discard(o)
        poss = _possessor_of(k, o, remaining)
        if poss is not None:
            result.frame["object"] = {"lemma": o, "possessor": poss}
            result.basis["possessor"] = _DERIVED
            remaining.discard(poss["lemma"])
        break
    # Нехватка: если предиката нет в наборе и не выводится — назвать ребро.
    if not verbs_in_set and obj_nouns:
        orphan = set()
        for o in obj_nouns:
            orphan.update(_verbs_over(k, o))
        assigned_objects = set(obj_nouns)
        # Кто мог бы быть деятелем: существительные набора, не занятые как
        # объекты. Каждому называем ребро «X CAN_DO V», которого не хватает.
        candidates_subj = [n for n in nouns_in_set if n not in assigned_objects]
        for s in candidates_subj:
            for v in orphan:
                if v not in k.related(s, "CAN_DO"):
                    result.gaps.append((s, v))
    elif verbs_in_set and not subj_nouns:
        result.gaps.append(("subject", result.frame.get("predicate", "")))
    result.outcome = "NO_PATH"
    return result


def _place_time(k, best: Dict[str, Any],
                remaining: set) -> Tuple[str, str]:
    """Место (LOCATED_IN от объекта/субъекта) и время (CIRCUMSTANCE предиката).

    Только понятия из оставшегося набора — иначе месту неоткуда взяться.
    """
    place = None
    for head in (best["object"], best["subject"]):
        for p in k.related(head, "LOCATED_IN"):
            if p not in remaining:
                continue
            # ДЕЯТЕЛЬ НЕ МЕСТО (пункт 9 диагноза): существительное с CAN_DO
            # — участник, а не пассивная локация. «сторож сторожит склад ОТ
            # ВОРОВ»: если база по ошибке дала «склад LOCATED_IN воры», а
            # воры умеют красть, это деятель, а не место — не ставить его
            # в слот места, иначе выйдет ложное «на ворах».
            if k.related(p, "CAN_DO"):
                continue
            place = p
            break
        if place is not None:
            break
    time = None
    for t in k.related(best["predicate"], "CIRCUMSTANCE"):
        if t in remaining:
            time = t
            break
    return place, time