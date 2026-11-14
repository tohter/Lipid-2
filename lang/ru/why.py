"""
lang/ru/why.py — русская сторона вопроса «ПОЧЕМУ X сделал Y».

Зачем отдельный файл (тот же закон границы, что у lang/ru/ability.py)
--------------------------------------------------------------------
Русский текст обязан жить в lang/ru/, а не в brain/: ядро не импортирует
русские строки (категория C, потолок 0), мозг держит язык за швом (категория
S). Оба конца границы «почему» — распознавание и порождение — здесь:

    русский вопрос -> parse_why() -> роли {subject, action, object}
    причинная цепочка (структура) -> render_why() -> русская фраза

Один словарь на оба конца — чтобы модель и спросила «почему кошка сожрала
канарейку», и поняла этот же вопрос.

ПОЧЕМУ РАЗБОР, А НЕ ШАБЛОНЫ
---------------------------
parse_why() разбирает морфологическим разбором (core.morph.analyze), как и
parse_ability(). Правила три:

    маркер   — «почему»/«зачем» (ADVB, content=False) в начале фразы;
    действие — первый глагол-лемма ПОСЛЕ маркера;
    субъект  — существительное ДО этого глагола;
    объект   — существительное ПОСЛЕ него.

Порядок слов в русском свободный, поэтому правила проверяются контрактом на
конкретных формах, а не объявляются истинными по умолчанию.

ЧЕСТНОСТЬ (требование хозяина, усиленное обоими советчиками)
------------------------------------------------------------
«почему кошка рассказала секрет» — НЕВЕРНАЯ посылка: в рассказе секрет
рассказала канарейка. render_why() обязана уметь разделить:
    факт        — что ПРЯМО сказано в событиях рассказа;
    причинный вывод — цепочка событий (это не выдумка, она извлечена);
    гипотезу    — мотив («вероятно, восприняла как предательство») — только
                  с пометкой, никогда как факт.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    from core.morph import analyze
except ImportError:                              # pragma: no cover
    analyze = None

# Маркеры вопроса о причине. Лемма «почему»/«зачем» (ADVB), content=False —
# в графе узлом не становится, но фразу открывает.
WHY_MARKERS = frozenset({"почему", "зачем"})
# Служебные слова, которые не должны стать субъектом/объектом.
_SERVICE = frozenset({
    "ли", "бы", "б", "же", "ведь", "разве", "сам", "сама", "самое", "эти",
    "этот", "эта", "это", "такой", "такая", "такое", "лишь", "как", "кто",
    "что", "он", "она", "они", "оно", "я", "ты", "мы", "вы", "мой", "твой",
    "наш", "ваш", "его", "её", "ее", "их",
})
VERB_POS = frozenset({"VERB", "INFN"})
NOUN_POS = "NOUN"


def parse_why(question: str) -> Dict[str, Any]:
    """Русский вопрос «почему» -> {subject, action, object, marker}.

    Возвращает роли леммами. Если маркера причины нет — возвращает пустые
    роли (это вопрос не про «почему», говорит другой обработчик).
    """
    out: Dict[str, Any] = {"subject": "", "action": "", "object": "",
                           "marker": "", "content": []}
    text = (question or "").strip().lower()
    if not text:
        return out

    # Маркер обязан быть. Без него parse_why() не имеет права решать за
    # остальные обработчики (иначе мозг начнёт врать про причину на любой
    # фразе с глаголом).
    if not any(m in text.split() for m in WHY_MARKERS):
        return out
    marker = next((m for m in WHY_MARKERS if m in text.split()), "")

    if analyze is None:                          # pragma: no cover
        out["marker"] = marker
        return out
    toks = [t for t in analyze(text)
            if t.get("content") and t.get("lemma")
            and t["lemma"] not in _SERVICE]
    out["content"] = [t["lemma"] for t in toks]
    if not toks:
        out["marker"] = marker
        return out

    verb_at = -1
    for i, t in enumerate(toks):
        if t.get("pos") in VERB_POS:
            verb_at = i
            break
    if verb_at >= 0:
        out["action"] = toks[verb_at]["lemma"]

    nouns = [i for i, t in enumerate(toks) if t.get("pos") == NOUN_POS]
    before = [i for i in nouns if verb_at < 0 or i < verb_at]
    after = [i for i in nouns if verb_at >= 0 and i > verb_at]
    if before:
        out["subject"] = toks[before[0]]["lemma"]
    if after:
        out["object"] = toks[after[-1]]["lemma"]
    elif nouns and not out["subject"] and verb_at >= 0:
        out["object"] = toks[nouns[-1]]["lemma"]
    out["marker"] = marker
    return out


def render_why(subject: str, action: str, obj: str, chain: List[str],
               cause: str = "", found: bool = True,
               evidence: str = "") -> str:
    """Причинная цепочка (структура) -> русская фраза.

    chain — упорядоченные действия-леммы причинной цепочки («разозлиться»,
    «сожрать»). cause — прямо названная в тексте причина, если была.
    evidence — фраза-свидетельство из событий рассказа (та самая «кошка
    разозлилась и сожрала канарейку»): она ЗАМЕНЯЕТ собранную из лемм фразу,
    чтобы не собирать русскую грамматику из лемм (это шов, а не речь).
    found=False — честный отказ: события с таким глаголом/агентом в рассказе
    нет (неверная посылка вопроса).
    """
    if not found:
        return (f"В этом рассказе события «{action}» с агентом «{subject}» "
                f"нет — причина неизвестна. "
                f"Проверь, кто на самом деле совершал действие.")
    claim = (evidence or "").strip()
    if not claim:
        who = subject or "кто-то"
        claim = f"{who} {action}" + (f" {obj}" if obj else "")
    if cause:
        return (f"{claim}. Причина прямо названа в рассказе: {cause}.")
    if len(chain) >= 2:
        path = " → ".join(chain)
        return (f"{claim}. Причинная цепочка событий рассказа: {path}.")
    if chain:
        return (f"{claim}. Событие «{action}» есть в рассказе, но причина "
                f"в тексте не названа.")
    return (f"{claim}. События «{action}» в рассказе нет — причина неизвестна.")