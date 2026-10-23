"""
ЗАМЕР п.A, шаг 3: ЗОЛОТАЯ МЕРА ближайшего семейства — из ЛИДА ру-Вики.

Первая строка статьи пишется по шаблону «X — это Y»; Y и есть ближайшее
семейство, выбранное РЕДАКТОРАМИ Википедии. Это данные, а не моё мнение, и на
них можно проверять порядок кандидатов (док, п.10).

Извлечение v2. Первая версия давала мусор: «Felis catus) — домашнее
животное» (подлежащее содержит скобки и запятые, жадный шаблон резал не там).
Правила v2, каждое оправдано замером на 22 словах:
  1) снять скруглённые вставки — в них этимология и латынь, а не род;
  2) начинать с ПЕРВОГО предложения, контаминацию «X — это Y, а также Z»
     обрывать на первой запятой-перечислении лишь в том случае, если далее
     идёт придаток; иначе род теряется («город — населённый пункт, имеющий…»);
  3) отбросить предложения без дефиниционной связки: «в обиходной речи —
     „пёс“» не является определением, и молча принять его за род нельзя.
Слова, где связь не найдена, остаются в списке с пустым золотом — они не
участвуют в замере точности, но участвуют в замере покрытия меры.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LEADS = os.path.join(ROOT, "experiments", "wiki_lead_snapshot.json")
CATS = os.path.join(ROOT, "experiments", "wiki_category_snapshot.json")
OUT = os.path.join(ROOT, "experiments", "gold_genus.json")

PAREN = re.compile(r"\([^()]*\)")
DASH = "\u2014"
DEF_COPULA = re.compile(
    r"^\s*(?P<subj>.{2,90}?)\s*(?:\u2014|-|--|\u2013)\s*(?P<this>\u044d\u0442\u043e\s+)?"
    r"(?P<pred>.+)$", re.IGNORECASE)

# Слова, после которых дефиниция продолжается, а не кончается. Разделено
# замером: «предмет мебели, имеющий…» — род «предмет мебели», придаток отсечён;
# «экологическая система, биогеценоз, в которой…» — там перечисление СИНОНИМОВ,
# и отсечение по первой запятой даёт верный род.
SPLIT = re.compile(r",(?=\s*(?:\u043a\u043e\u0442\u043e\u0440|\u0438\u043c\u0435\u044e\u0449|\u043d\u0430\u0437\u044b\u0432\u0430|\u043f\u043e\u0434\u0440\u0430\u0437\u0443\u043c|\u043f\u043e\u043d\u044f\u0442\u0438\u0435))")


def first_sentence(text):
    text = re.split(r"(?<=[.;])\s", text.strip(), maxsplit=1)[0]
    return text.strip()


def gold(word, lead):
    sent = first_sentence(lead)
    sent = PAREN.sub("", sent)
    sent = re.sub(r"\s+", " ", sent).strip()
    m = DEF_COPULA.match(sent)
    if not m:
        return ""
    subj = m.group("subj").strip(" ,;:")
    pred = m.group("pred").strip(" ,;:.")
    if not pred:
        return ""
    # подлежащее обязано содержать слово статьи, иначе это не дефиниция
    if word.lower().split()[0][:4] not in subj.lower():
        return ""
    pred = SPLIT.split(pred)[0].strip(" ,;.")
    return pred


def main():
    leads = json.load(open(LEADS, encoding="utf-8"))
    cats = json.load(open(CATS, encoding="utf-8"))
    golds = {}
    for w, rec in leads.items():
        g = gold(w, rec["lead"])
        golds[w] = g
        mark = "✓" if g else "✗"
        print(f"{mark} {w:14s} золотo: {g[:90]}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(golds, fh, ensure_ascii=False, indent=1)
    n = sum(1 for v in golds.values() if v)
    print(f"\nЗолото извлечено для {n} слов из {len(golds)}")

    # Что канал отвечает СЕЙЧАС (алфавитный порядок API) против золота
    from core.records import meaningful_categories
    hit = miss = 0
    print("\n──────── ТЕКУЩИЙ ОТВЕТ КАНАЛА vs ЗОЛОТО ────────")
    for w, rec in cats.items():
        cands = meaningful_categories(rec["title"], rec["categories"])
        if not golds.get(w) or not cands:
            continue
        cur = cands[0]
        ok = overlap(cur, golds[w])
        hit += ok
        miss += (not ok)
        print(f"{'✓' if ok else '✗'} {w:12s} канал: {cur[:38]:38s} | золото: {gold(w, leads[w]['lead'])[:44]}")
    print(f"\nСовпадений первой категории с золотом: {hit}, промахов: {miss} "
          f"(доля: {hit / max(1, hit + miss):.0%})")
    return 0


def overlap(cat, golden):
    """Пересечение лемм категории и золотого рода. Сличение по леммам, а не по
    строкам: «звёзды G-класса» против «звезда»."""
    try:
        from pymorphy3 import MorphAnalyzer
        m = MorphAnalyzer()
    except ImportError:
        return cat.lower() in golden.lower()

    def lems(s):
        out = set()
        for tok in re.findall(r"[а-яёА-ЯЁa-zA-Z]{3,}", s):
            out.add(m.parse(tok)[0].normal_form.lower())
        return out
    a, b = lems(cat), lems(golden)
    return bool(a & b)


if __name__ == "__main__":
    raise SystemExit(main())
