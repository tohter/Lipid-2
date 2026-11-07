"""
lang/ru/ability.py — русская сторона вопроса «МОЖЕТ ЛИ X СДЕЛАТЬ Y».

ПОЧЕМУ ОТДЕЛЬНЫЙ ФАЙЛ, а не внутри brain/thinker.py
---------------------------------------------------
Думалка родилась в brain/ и сразу нарушила закон границы (experiments/
IR_BOUNDARY.md): она импортировала core.morph (шим S) и сама собирала русскую
фразу ответа (выход O). Измеритель поймал это в тот же день — не как придирку
к стилю, а как рост границы 6 -> 7. Правильный ответ на такой рост НЕ «поднять
потолок», а убрать язык из мозга. Здесь живут ОБА конца границы:

    русский текст -> parse_ability() -> роли (субъект, действие, объект)
    вердикт думалки (структура) -> render_verdict() -> русская фраза

ПОЧЕМУ ОДИН ФАЙЛ НА РАСПОЗНАВАНИЕ И ПОРОЖДЕНИЕ
----------------------------------------------
Словарь форм один и тот же: модель обязана уметь и спросить «как связаны X и
Y», и понять этот же вопрос (шаг 7 очереди хозяйки). Два словаря разъехались
бы за неделю — ровно то, что ЗАМЕРЕНО на двух маршрутизаторах вопросов
(lang/ru/questions.py и brain/question.py отвечали по-разному на один вопрос).

ЗНАЧЕНИЯ РОЛЕЙ — ЛЕММЫ, НЕ ФОРМЫ
--------------------------------
Думалка оперирует леммами («потушить», а не «потушил»), потому что узлы графа
хранятся леммами. Все русские слова, которые для этого нужны, — здесь.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

try:
    from core.morph import analyze
except ImportError:                              # pragma: no cover
    analyze = None

# ── СЛОВАРЬ ФОРМ ВОПРОСА О СПОСОБНОСТИ ──────────────────────
#
# Это РУССКИЕ строки. В core/ им место запрещено категорией C (потолок 0),
# в brain/ — швом категории S. Здесь они дома.
ABILITY_MARKERS = frozenset({
    "мочь", "смочь", "может", "могут", "смог", "смогла", "смогли",
    "сможет", "смогу", "сможем", "сможете",
    "способен", "способна", "способно", "способны",
    "в состоянии", "в силах", "получится", "получится", "удастся",
    "удастся",
})
# Модальный глагол НЕ является действием вопроса. Без этого «может» само
# становилось сказуемым и разбор рассыпался (замер первой версии думалки).
MODAL_LEMMAS = frozenset({"мочь", "смочь", "хотеть", "нужно", "надо",
                        "стать", "удаваться"})
SERVICE_WORDS = frozenset({
    "ли", "бы", "б", "же", "ведь", "разве", "сам", "сама", "самое", "эти",
    "этот", "эта", "это", "такой", "такая", "такое", "лишь", "как",
})
NEGATION_MARKERS = ("не", "ни", "никогда", "вряд", "едва", "разве")

VERB_POS = frozenset({"VERB", "INFN"})
NOUN_POS = "NOUN"


def parse_ability(question: str) -> Dict[str, Any]:
    """Русский вопрос -> {subject, action, object, negative, content}.

    Разбор РАЗБОРОМ (morphological analyze), а не перечем шаблонов: список
    шаблонов означает «на каждую форму свой костыль». Правила ровно три:

      действие — глагол, не являющийся модальным маркером;
      субъект  — существительное ДО этого глагола;
      объект   — существительное ПОСЛЕ него.

    Порядок слов в русском свободный, поэтому правила проверяются контрактом
    на конкретных формах, а не объявляются истинными по умолчанию.
    """
    out: Dict[str, Any] = {"subject": "", "action": "", "object": "",
                           "negative": False, "content": [], "markers": []}
    text = (question or "").strip()
    if not text:
        return out
    low = text.lower()
    out["negative"] = any(f" {m} " in f" {low} " for m in NEGATION_MARKERS)
    out["ability"] = any(m in low for m in ("может", "могут", "смог", "способ",
                                            "в состоянии", "в силах",
                                            "получится", "удастся"))

    if analyze is None:                          # pragma: no cover
        return out
    toks = [t for t in analyze(text)
            if t.get("content") and t.get("lemma")
            and t["lemma"] not in SERVICE_WORDS]
    out["content"] = [t["lemma"] for t in toks]
    if not toks:
        return out

    modal_at = -1
    for i, t in enumerate(toks):
        if t["lemma"] in MODAL_LEMMAS or t["lemma"] in ABILITY_MARKERS:
            modal_at = i
            break

    verb_at = -1
    for i, t in enumerate(toks):
        if t.get("pos") in VERB_POS and i != modal_at:
            verb_at = i
            break
    if verb_at >= 0:
        out["action"] = toks[verb_at]["lemma"]

    nouns = [i for i, t in enumerate(toks)
             if t.get("pos") == NOUN_POS and i != modal_at]
    before = [i for i in nouns if verb_at < 0 or i < verb_at]
    after = [i for i in nouns if verb_at >= 0 and i > verb_at]
    if before:
        out["subject"] = toks[before[0]]["lemma"]
    if after:
        out["object"] = toks[after[-1]]["lemma"]
    elif nouns and not out["subject"] and modal_at >= 0:
        out["object"] = toks[nouns[-1]]["lemma"]
    return out


def render_verdict(v: Any) -> str:
    """Структурный вердикт думалки -> русская фраза.

    Здесь собирается ТОЛЬКО язык. Ни одного решения «отвечать да или нет» тут
    нет: ok/code/links приходят из brain/thinker.py готовыми.
    """
    subject = getattr(v, "subject", "") or "это"
    action = getattr(v, "action", "") or ""
    obj = getattr(v, "object", "") or ""
    links = list(getattr(v, "links", []) or [])

    if getattr(v, "ok", False) and links:
        reasons: List[str] = []
        for link in links[:3]:
            source = getattr(link, "source", "")
            verb = getattr(link, "action", "")
            target = getattr(link, "target", "")
            reasons.append(f"{source} {verb} {target}".strip())
        text = f"{subject} может {action} {obj}".strip()
        if reasons:
            text += ": " + "; ".join(dict.fromkeys(reasons))
        depth = getattr(v, "depth_used", 0)
        if depth:
            text += f" (через {depth} шаг(ов) обобщения)"
        return text

    code = getattr(v, "code", "")
    if code == "UNKNOWN_WORDS":
        words = ", ".join(getattr(v, "unknown_words", []) or []) or "неизвестно"
        return f"не могу рассуждать: не знаю слов «{words}»"
    if code == "MISSING_ACTION":
        gaps: Sequence[Any] = getattr(v, "gaps", []) or []
        if gaps:
            first = gaps[0]
            return (f"не могу утверждать, что {subject} может {action} {obj}: "
                    f"в графе нет связи между «{first[0]}» и «{first[1]}»")
        return (f"не могу утверждать, что {subject} может {action} {obj}: "
                f"действие вопроса ни с чем не связано")
    if code == "MISSING_CARRIER":
        gaps = getattr(v, "gaps", []) or []
        who = gaps[0][1] if gaps and len(gaps[0]) > 1 else ""
        return (f"{subject} не является и не содержит того, что умеет "
                f"{who}: поэтому не могу сказать, что он может {action} {obj}")
    if code == "TOO_DEEP":
        return ("связка есть глубже, чем я разрешаю себе искать: отвечать «да»"
                " было бы выдумкой")
    if code == "LINK_FOUND_NEGATED":
        return (f"вопрос был «не может ли {subject} {action} {obj}», а связь"
                " нашлась — значит может")
    if "NEGATED_QUESTION_BUT_LINK_FOUND" in (getattr(v, "notes", []) or []):
        return f"связь нашлась, поэтому ответ — может"
    return f"не нашёл, чем {subject} может {action} {obj}"
