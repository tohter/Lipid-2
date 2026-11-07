"""РАЗБОР ДОСТОВЕРНОСТИ ТРЕМЯ КАНАЛАМИ: локалка, Вики и я (пользователь).

ЗАЧЕМ. Хозяин спросил: можно ли учителю сразу выставлять высокую достоверность
за «заведомо правильное», и что делать, если локалка предложит много, а Вики
подтвердит. Прежде чем менять весовую шкалу, надо ИЗМЕРИТЬ, что каждый канал
действительно знает про одни и те же утверждения — и где они расходятся.

ТРИ НЕЗАВИСИМЫХ КАНАЛА (независимость — сегодняшний закон свидетельства):
  1. Гемма (localhost:8080) — просит вердикт, уверенность и РОД знания
     (аксиома/правило/факт). Это то, что модель готова заявить сама.
  2. РУССКАЯ ВИКИПЕДИЯ — не верит ни чьей уверенности. Текст статьи
     раскладывается OUR extractor (core.extract.SimpleExtractor), и мы смотрим,
     сказано ли в тексте ОТНОШЕНИЕ между теми же словами. Совпадение слов в
     абзаце — не подтверждение; подтверждение = найденный глагол-связка.
  3. Я (Amanda) в роли канала «спроси пользователя» — самое железное, что
     есть: рассуждение, которое можно прочитать и оспорить. Основание пишется
     словами, никогда числом из воздуха.

ЧТО ПЕЧАТАЕТ:
  * построчно: вердикт каждого канала, его уверенность, согласие каналов;
  * ИТОГ: сколько утверждений закрыто ДВУМЯ независимыми каналами, сколько
    разошлись (там и только там — вопрос к хозяину), где Вики молчит;
  * сколько «заведомо правильных» утверждений Вики подтвердила МЕХАНИЧЕСКИ,
    а где нам пришлось бы поверить модели на слово.

НИЧЕГО НЕ ПИШЕТ в живой мир: каждый канал в mktemp, база не трогается.

Запуск:  LIPID2_STATUS=0 .venv/bin/python experiments/confidence_triage.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lang.ru.lemma import install                      # noqa: E402
install()

from core.extract import SimpleExtractor               # noqa: E402
from core.morph import Morph                           # noqa: E402
from teachers.teacher_wiki import WikiTeacher          # noqa: E402

LLM_URL = "http://localhost:8080/v1/chat/completions"
LLM_MODEL = "gemma"

# ── МАТЕРИАЛ ───────────────────────────────────────────────────────────────
# subject, predicate, object, наш вердикт-ожидание, чьё это основание
#
# РОД (аксиома/правило/факт) задаю ЗАГОТОВКОЙ, чтобы сравнить с тем, что
# скажет Гемма: сходятся ли мы в том, ЧТО это за знание, а не только в том,
# истинно оно или ложно.
STATEMENTS = [
    # заведомо правильные: логика (не требует перепроверки)
    ("кит", "является", "животное", True, "правило",
     "таксономия: кит — млекопитающее, млекопитающее — животное; выводимо"),
    ("квадрат", "имеет", "сторона", True, "аксиома",
     "геометрия: 4 стороны следуют из определения квадрата"),
    ("часть", "входит", "целое", True, "аксиома",
     "логика mereology: транзитивно, перепроверять наблюдениями бессмысленно"),
    # заведомо правильные: мир (обобщение с огромным n, но не логика)
    ("солнце", "восходит", "восток", True, "факт",
     "астрономия: восход на востоке — следствие вращения Земли"),
    ("вода", "гасит", "огонь", True, "факт",
     "физика: вода охлаждает и перекрывает кислород"),
    ("человек", "дышит", "кислород", True, "факт",
     "биология: аэробное дыхание"),
    # СОМНИТЕЛЬНЫЕ и ложные: тут и проверяется, не отмывает ли уверенность
    ("кит", "дышит", "жабра", False, "факт",
     "кит — млекопитающее, дышит лёгкими; правдоподобная ловушка (живёт в воде)"),
    ("курица", "является", "птица", True, "факт",
     "курица — птица; проверяет, что отрицание полёта не сбивает класс"),
    ("пингвин", "летает", "самолёт", False, "факт",
     "бессмысленная пара: пингвин не летает, самолёт — не способ"),
    ("металл", "плавится", "холод", False, "факт",
     "плавление требует нагрева; «от холода» — ложное основание"),
    ("луна", "излучает", "свет", False, "факт",
     "луна отражает солнечный свет, сама его не производит"),
    ("земля", "вращается", "солнце", True, "факт",
     "гелиоцентризм: обращение Земли вокруг Солнца"),
]

PROMPT = (
    "Утверждение: {s} {p} {o}.\n"
    "Оцени его как носитель языка и верни СТРОГО JSON без пояснений:\n"
    '{{"истинно": true|false, "уверенность": 0.0-1.0, '
    '"род": "аксиома|правило|факт", "почему": "до 6 слов"}}'
)


def llm_judge(s: str, p: str, o: str) -> dict:
    """Verdict of the local model: truth, its confidence, and the KIND of knowledge."""
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {"role": "system",
             "content": "Ты аккуратный оценщик. Отвечай строго JSON, без текста после JSON."},
            {"role": "user", "content": PROMPT.format(s=s, p=p, o=o)},
        ],
        "max_tokens": 2500, "temperature": 0,
    }).encode()
    req = urllib.request.Request(LLM_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=180).read())["choices"][0]["message"]["content"]
    except Exception as exc:
        return {"ok": False, "why": f"сбой канала: {type(exc).__name__}"}
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        return {"ok": False, "why": "не JSON"}
    try:
        d = json.loads(raw[start:end + 1])
    except Exception as exc:
        return {"ok": False, "why": f"битый JSON: {exc}"}
    return {"ok": True, "truth": bool(d.get("истинно")),
            "conf": float(d.get("уверенность") or 0), "kind": str(d.get("род") or "?"),
            "why": str(d.get("почему") or "")}


# ── КАНАЛ 2: Википедия + НАШ экстрактор (не верит чужой уверенности) ────────
_wiki = WikiTeacher(memory_dir="/tmp/triage_wiki")
_ext = SimpleExtractor()


def _gram(word: str) -> str:
    """Лемма слова — чтобы «гасит» сошёлся с «гасить» из статьи."""
    try:
        return str(Morph.parse(word).get("lemma") or word).lower()
    except Exception:
        return word.lower()


def wiki_check(s: str, p: str, o: str) -> dict:
    """Does the article text actually STATE the relation, by our own extractor?

    Подтверждением считается НЕ соседство слов в абзаце, а извлечённая нашим
    экстрактором связь, у которой субъект и объект совпадают по лемме с
    запрошенными, а глагол-связка не пустая. Иначе «косвенные упоминания»
    выглядели бы подтверждением.
    """
    try:
        page = _wiki._fetch_page(s)
    except Exception as exc:
        return {"found": False, "why": f"сбой: {type(exc).__name__}"}
    if not page:
        return {"found": False, "why": "статья не найдена"}
    title, text, _cats = page
    if not text:
        return {"found": False, "why": "статья пустая"}
    tg = _gram(title)
    if tg != _gram(s):
        # подмена темы (замерено ранее: «невестка» -> статья «Родство»)
        return {"found": True, "states": None, "offtopic": True,
                "why": f"статья про «{title}», не про «{s}»", "chars": len(text)}
    sg, og = _gram(s), _gram(o)
    hits, relations = 0, set()
    for sent in text.replace("\n", " ").split(". "):
        if len(sent) < 12:
            continue
        try:
            facts = _ext.extract(sent)
        except Exception:
            continue
        facts = facts if isinstance(facts, list) else [facts]
        for f in facts:
            rel = getattr(f, "relation", None)
            if rel is None or not getattr(rel, "verb", ""):
                continue
            subj = _gram(str(getattr(rel, "subject", "") or ""))
            obj = _gram(str(getattr(rel, "object", "") or ""))
            if (subj == sg or subj in (s.lower(), tg)) and (obj == og or obj == o.lower()):
                hits += 1
                relations.add(str(rel.verb).lower())
    return {"found": True, "offtopic": False, "states": hits > 0, "hits": hits,
            "via": sorted(relations)[:4], "chars": len(text),
            "why": ("связь названа текстом" if hits else "текст связь не называет")}


def main() -> int:
    print("\n" + "=" * 78)
    print("РАЗБОР ДОСТОВЕРНОСТИ: Гемма | Вики (наш экстрактор) | я (пользователь)")
    print("=" * 78)

    rows = []
    for s, p, o, want, kind, reason in STATEMENTS:
        print(f"\n── {s} {p} {o} " + "─" * 40)
        gem = llm_judge(s, p, o)
        if gem.get("ok"):
            print(f"   Гемма : истинно={gem['truth']}  уверенность={gem['conf']:.2f}  "
                  f"род={gem['kind']}  — {gem['why']}")
        else:
            print(f"   Гемма : {gem['why']}")

        wik = wiki_check(s, p, o)
        mark = ("согласна" if wik.get("states") else
                ("не подтверждает" if wik.get("states") is False else
                 ("МОЛЧИТ: " + wik.get("why", ""))))
        if wik.get("offtopic"):
            mark = "ПОДМЕНА ТЕМЫ: " + wik.get("why", "")
        print(f"   Вики  : {mark}"
              + (f"  связок={wik.get('hits')} через={wik.get('via')}" if wik.get("hits") else ""))
        print(f"   Я     : истинно={want}  род={kind}  — {reason}")

        agree = gem.get("ok") and (gem["truth"] == want)
        rows.append({"s": f"{s} {p} {o}", "want": want, "kind": kind,
                     "gem_truth": gem.get("truth"), "gem_conf": gem.get("conf"),
                     "gem_kind": gem.get("kind"), "gem_ok": gem.get("ok"),
                     "gem_agrees": agree, "wiki": wik.get("states"),
                     "wiki_why": wik.get("why"), "offtopic": wik.get("offtopic")})

    print("\n" + "=" * 78)
    print("ИТОГ")
    print("=" * 78)
    n = len(rows)
    gem_good = sum(1 for r in rows if r["gem_agrees"])
    gem_conf_on_right = [r["gem_conf"] for r in rows if r["gem_agrees"]]
    gem_conf_on_wrong = [r["gem_conf"] for r in rows
                         if r["gem_ok"] and not r["gem_agrees"]]
    wiki_yes = sum(1 for r in rows if r["wiki"] is True)
    wiki_no = sum(1 for r in rows if r["wiki"] is False)
    wiki_silent = sum(1 for r in rows if r["wiki"] is None)
    both = sum(1 for r in rows if r["gem_agrees"] and r["wiki"] is True)
    clash = [r for r in rows
             if (r["gem_agrees"] and r["wiki"] is False)
             or (not r["gem_agrees"] and r["wiki"] is True)]
    print(f"   Гемма угадала истинность: {gem_good} из {n}")
    if gem_conf_on_right:
        print(f"      уверенность на верных: {min(gem_conf_on_right):.2f}"
              f"–{max(gem_conf_on_right):.2f} (среднее "
              f"{sum(gem_conf_on_right)/len(gem_conf_on_right):.2f})")
    if gem_conf_on_wrong:
        print(f"      !!! уверенность на ОШИБКАХ: "
              f"{max(gem_conf_on_wrong):.2f} — вот что значит доверить модели ставить вес")
    print(f"   Вики подтвердила механически: {wiki_yes}, "
          f"не подтвердила: {wiki_no}, молчит (нет статьи/подмена): {wiki_silent}")
    print(f"   Два независимых канала согласны: {both} — это настоящие 2 свидетельства")
    print(f"   РАЗОШЛИСЬ: {len(clash)}"
          + (" -> " + "; ".join(r["s"] for r in clash) if clash else ""))
    kinds = {}
    for r in rows:
        kinds.setdefault((r["kind"], r["gem_kind"]), 0)
        kinds[(r["kind"], r["gem_kind"])] += 1
    print("   РОД знания (мой -> Гемма): " + ", ".join(f"{a}->{b}:{c}" for (a, b), c in sorted(kinds.items())))
    print("\n   Вывод читается из чисел выше: там, где Вики молчит, высокая")
    print("   уверенность Геммы нечем подтвердить, а там, где каналы согласны,")
    print("   свидетельство ДЕЙСТВИТЕЛЬНО отдельное. Разногласие — это не среднее,")
    print("   это вопрос к хозяину.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
