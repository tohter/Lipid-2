"""
ЗАМЕР п.A, шаг 4: ПРАВИЛО ПОРЯДКА ИЗ ГРАФА РОДИТЕЛЕЙ vs ЗОЛОТО.

Исходные замеры, которые обязано обойти любое правило:
  * API отдаёт категории в АЛФАВИТНОМ порядке — 19 слов из 22 (p26b).
  * Ответ из hierarchy[0] звучит ТОЛЬКО когда definition пуст: у реальных
    текстовых сущностей так вышло 3 раза из 6, отсюда «флюгер — это
    измерительные приборы», «луга — это населённые пункты, основанные в xvi
    веке».
  * Ранжировать по весу/частоте нельзя: грубая категория подтверждается чаще
    именно потому, что она грубая (п.26, замер 9).

Золотая мера — не моё мнение: это род, выбранный РЕДАКТОРАМИ Википедии в
первой строке статьи («X — это Y»), снимок wiki_lead_snapshot.json.

Правила-кандидаты, которые сравниваются ОДНИМ прогоном:
  as_is      — как сейчас, порядок API;
  reverse    — обратный порядок (документировался в докстринге канала,
               проверим хотя бы утверждение «Викиидёт от абстрактных к
               конкретным»);
  spec       — самый специфичный кандидат (у которого больше кандидатов-
               предков в графе категорий);
  spec+shared— при равенстве тот, у кого больше общих родителей с остальными.

ВЫВОД ЗАМЕРА ОБЯЗАН БЫТЬ ЧИТАЕМ: по каждому слову печатается, какое правило
какой категорией ответило и совпало ли это с золотом.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CATS = os.path.join(ROOT, "experiments", "wiki_category_snapshot.json")
PAR = os.path.join(ROOT, "experiments", "wiki_parents_snapshot.json")
LEADS = os.path.join(ROOT, "experiments", "wiki_lead_snapshot.json")

from core.records import meaningful_categories                       # noqa: E402
from teachers.teacher_wiki import WikiTeacher                        # noqa: E402

_wiki = WikiTeacher.__new__(WikiTeacher)   # фильтр без сети и без памяти


def channel_candidates(title, raw):
    """КАНАЛЬНЫЕ кандидаты, а не сырые категории. Первая версия замера меряла
    только meaningful_categories и ранжировала мусор («Википедия:Cite web…»),
    который канал выбрасывает сам — вывод был бы неверен. Порядок обязан
    выбираться из того же списка, который доходит до онтологии."""
    hier = _wiki._categories_to_hierarchy(raw, title)
    return meaningful_categories(title, hier)

MORPH = None


def morph():
    global MORPH
    if MORPH is None:
        try:
            from pymorphy3 import MorphAnalyzer
            MORPH = MorphAnalyzer()
        except ImportError:
            MORPH = False
    return MORPH


def lems(s):
    """Леммы слов фразы. Сличение по леммам, а не строкам: «звёзды G-класса»
    против «звезда» — одна форма, разные строки."""
    m = morph()
    out = set()
    for tok in re.findall(r"[а-яёА-ЯЁa-zA-Z]{3,}", s or ""):
        tok = tok.lower()
        if m:
            out.add(m.parse(tok)[0].normal_form)
        else:
            out.add(tok)
    return out


# ─────────────────────────── золото из лида ───────────────────────────
SOFT_HYPHEN = "\u00ad"
PAREN = re.compile(r"\([^()]*\)")
NESTED = re.compile(r"\([^()]*\([^()]*\)[^()]*\)")
DASH = r"(?:\u2014|-{1,2}|\u2013)"
# Страницы-разрешения неоднозначностей: их «определение» — перечисление значений,
# брать из них род нельзя, иначе золото само становится мусором.
DISAMBIG = re.compile(r"значения|уточнен|неоднозначн", re.IGNORECASE)
# Придаток, после которого род уже назван.
TRAIL = re.compile(
    r",\s*(?:\u043a\u043e\u0442\u043e\u0440|\u0438\u043c\u0435\u044e\u0449|\u043d\u0430\u0437\u044b\u0432\u0430\u0435\u0442|\u043f\u043e\u0434\u0440\u0430\u0437\u0443\u043c\u0435\u0432|\u043f\u043e\u043d\u044f\u0442\u0438\u0435|\u043f\u043e\u0437\u0432\u043e\u043b\u044f\u044e\u0449|\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u043c|\u043a\u043e\u0442\u043e\u0440\u043e\u0435 \u043f\u0440\u0435\u0434\u043d)")


def clean(text):
    text = (text or "").replace(SOFT_HYPHEN, "")
    for _ in range(3):
        new = NESTED.sub(" ", text)
        if new == text:
            break
        text = new
    text = PAREN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def gold(word, lead):
    """Род из первой строки статьи. '' — если получить его честно нельзя."""
    sent = clean(lead).split(". ")[0].rstrip(".")
    if DISAMBIG.search(sent):
        return ""
    m = re.match(r"^(.{2,120}?)\s*" + DASH + r"\s*(\u044d\u0442\u043e\s+)?(.+)$",
                 sent, re.IGNORECASE)
    if not m:
        return ""
    subj, pred = m.group(1).strip(" ,;:"), m.group(3).strip(" ,;:."'"'"'«»")
    stem = re.sub(r"[^а-яё]", "", word.lower())[:4]
    if stem and stem not in subj.lower():
        return ""                       # это не дефиниция данного слова
    pred = TRAIL.split(pred)[0].strip(" ,;.")
    return pred if len(pred) >= 4 else ""


# ──────────────────────── граф родителей ────────────────────────────
def load_parents():
    raw = json.load(open(PAR, encoding="utf-8"))
    return {k.lower(): [p.lower() for p in v] for k, v in raw.items()}


def is_ancestor(a, b, parents, depth=3):
    """a — предок b? Обход в ширину на фиксированную глубину: граф категорий
    Вики содержит циклы (категории ссылаются друг на друга), поэтому без
    ограничения глубины и множества посещённых обход не заканчивается."""
    seen, frontier = {a}, {a}
    for _ in range(depth):
        nxt = set()
        for node in frontier:
            for p in parents.get(node, []):
                if p == b:
                    return True
                if p not in seen:
                    seen.add(p)
                    nxt.add(p)
        frontier = nxt
        if not frontier:
            break
    return False


def ancestors(c, parents, depth=3):
    seen, frontier = set(), {c}
    for _ in range(depth):
        nxt = set()
        for node in frontier:
            for p in parents.get(node, []):
                if p not in seen:
                    seen.add(p)
                    nxt.add(p)
        frontier = nxt
    return seen


def rank(cands, parents, mode):
    if mode == "as_is":
        return list(cands)
    if mode == "reverse":
        return list(reversed(cands))
    if mode in ("spec", "spec+shared"):
        def score(c):
            anc = ancestors(c, parents)
            spec = sum(1 for d in cands if d != c and d in anc)   # кандидатов-предков
            if mode == "spec":
                return (-spec, cands.index(c))
            others = set()
            for d in cands:
                if d != c:
                    others |= set(parents.get(d, []))
            shared = len(set(parents.get(c, [])) & others)
            return (-spec, -shared, cands.index(c))
        return sorted(cands, key=score)
    raise ValueError(mode)


def main():
    cats = json.load(open(CATS, encoding="utf-8"))
    leads = json.load(open(LEADS, encoding="utf-8"))
    parents = load_parents()

    modes = ["as_is", "reverse", "spec", "spec+shared"]
    score = {m: {"hit": 0, "miss": 0} for m in modes}
    with_gold = 0
    print("──────── ПОРЯДОК vs ЗОЛОТО (род из первой строки Вики) ────────")
    print("родителей в графе:", len(parents), "категорий\n")
    for w, rec in cats.items():
        cands = channel_candidates(rec["title"], rec["categories"])
        g = gold(w, leads.get(w, {}).get("lead", ""))
        if not g or len(cands) < 2:
            continue
        with_gold += 1
        print(f"◆ {w}   золото: {g[:70]}")
        gold_lem = lems(g)
        for m in modes:
            top = rank(cands, parents, m)[0]
            ok = bool(lems(top) & gold_lem)
            score[m]["hit" if ok else "miss"] += 1
            print(f"    {m:12s} -> {top[:34]:34s} {'✓' if ok else '✗'}")
    print(f"\nСлов с золотом и выбором из ≥2 категорий: {with_gold}")
    for m in modes:
        h, mi = score[m]["hit"], score[m]["miss"]
        print(f"  {m:12s} совпало {h} из {h + mi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
