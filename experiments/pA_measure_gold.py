"""
ЗАМЕР п.A, шаг 2: ЗОЛОТАЯ МЕРА «ближайшее семейство».

Порядок кандидатов нельзя выбирать «на глаз». Нужна мера, и она обязана быть
данными, а не моим мнением (док, п.10). Источник разметки — сама Википедия:
первая строка статьи на ру-Вики пишется по шаблону «X — это Y» / «X — Y», где
Y и есть ближайшее семейство, выбранное РЕДАКТОРАМИ, а не мной.

Отдельно Gemma @8080 размечает тот же список независимо. Меры сверяются; там,
где оба источника согласны, золото считается надёжным. LLM здесь РАЗМЕТЧИК
КОРПУСА, он не участвует в ответе мозга — подменой причинного ответа он стать
не может (запрет дока, п.2).

Снимок: experiments/wiki_lead_snapshot.json (Вики отвечает 429 на частые
обращения, вывод обязан переживать сеть).
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SNAP = os.path.join(ROOT, "experiments", "wiki_category_snapshot.json")
OUT = os.path.join(ROOT, "experiments", "wiki_lead_snapshot.json")
UA = {"User-Agent": "Lipid2Bot/1.0 (neuromorphic-learning-agent; contact: localhost)"}

# Дефиниционная связка первой строки. Тире может быть -, —, –, и стоять с
# обеими сторонами; «это» факультативно.
DEF_COPULA = re.compile(
    r"^\s*(?P<subj>[^,.;:]{2,80}?)\s*(?:—|-|--|–)\s*(?P<this>это\s+)?(?P<pred>.+?)\s*[.;,]\s*$",
    re.IGNORECASE)


def lead(title):
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": 1, "exintro": 1, "redirects": 1, "titles": title,
    }
    body = urllib.parse.urlencode(params).encode()
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                "https://ru.wikipedia.org/w/api.php", data=body, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.load(r)
            pages = data.get("query", {}).get("pages", {})
            for p in pages.values():
                return re.sub(r"\s+", " ", p.get("extract", "") or "").strip()
            return ""
        except urllib.error.HTTPError as exc:
            last = exc
            wait = 30 * (attempt + 1)
            print(f"  HTTP {exc.code}, жду {wait} с", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"Вики не отвечает: {last}")


def definition_phrase(text):
    """Первое предложение с дефиниционной связкой, либо ''."""
    for sent in re.split(r"(?<=[.;])\s", text):
        m = DEF_COPULA.match(sent)
        if m and m.group("pred").strip():
            return sent.strip()
    return ""


def main() -> int:
    with open(SNAP, encoding="utf-8") as fh:
        snap = json.load(fh)
    out = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            out = json.load(fh)
    for word, rec in snap.items():
        if word in out:
            continue
        text = lead(rec["title"])
        out[word] = {"title": rec["title"], "lead": text,
                     "definition": definition_phrase(text)}
        print(f"◆ {word}: {out[word]['definition'][:120] or 'НЕТ дефиниционной связки'}")
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        time.sleep(3)

    have = sum(1 for v in out.values() if v["definition"])
    print(f"\nСлов: {len(out)}, с дефиниционной связкой в лиде: {have}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
