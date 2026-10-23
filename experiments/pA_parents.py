"""
ЗАМЕР п.A, шаг 1: РОДИТЕЛИ категорий Википедии.

Зачем. Категории статьи отвечают на «X ПРИНАДЛЕЖИТ множеству C», а не на
«X является C». Ближайший РОД понятия X в данных Вики сидит на один уровень
выше: в РОДИТЕЛЯХ тех категорий, которым принадлежит X. Поэтому порядок
кандидатов можно строить по графу, а не по алфавиту (замер
p26b_wiki_noise.py: порядок, отдаваемый API, алфавитный — 19 слов из 22).

Снимок сохраняется в experiments/wiki_parents_snapshot.json: Вики отвечает
429 на частые обращения, вывод обязан переживать сеть.

Запрос: батч до 40 заголовков + prop=categories. Это ОДИН запрос на 40
категорий, что спасает от лимита (у меня уже был 429 на 22 одиночных запроса).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "experiments", "wiki_category_snapshot.json")
OUT = os.path.join(ROOT, "experiments", "wiki_parents_snapshot.json")
UA = {"User-Agent": "Lipid2Bot/1.0 (neuromorphic-learning-agent; contact: localhost)"}
BATCH = 25


BATCH = 25
API = "https://ru.wikipedia.org/w/api.php"


def parents_batch(titles):
    """Родители для КАЖДОГО заголовка батча.generator не нужен: prop=categories
    над списком titles вернёт страницы категорий и их категории.

    POST, а не GET: 40 кириллических заголовков в строке запроса дают HTTP 414
    (замерено), в теле запроса проходят."""
    # Префикс «Категория:» ОБЯЗАТЕЛЕН: без него заголовок «волки» разрешается
    # в СТАТЬЮ «Волки», и вместо родителей категории приходят категории статьи
    # (замерено на первом снимке: 'собаки' -> [], 'волки' -> ['Волки', ...]).
    # clshow=!hidden убран: родители категорий на Вики преимущественно скрытые
    # («Википедия:Категории…»), с фильтром граф глубины обрывается на 1 шаге.
    params = {
        "action": "query", "format": "json", "prop": "categories",
        "cllimit": "max", "redirects": 1,
        "titles": "|".join("Категория:" + t for t in titles),
    }
    body = urllib.parse.urlencode(params).encode()
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(API, data=body, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.load(r)
            out = {}
            for p in data.get("query", {}).get("pages", {}).values():
                title = p.get("title", "")
                out[title] = [c["title"].replace("Категория:", "").strip()
                              for c in p.get("categories", [])]
            return out
        except urllib.error.HTTPError as exc:      # 429 — единственная честная причина
            last = exc
            wait = 30 * (attempt + 1)
            print(f"  HTTP {exc.code}, жду {wait} с", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"Вики не отвечает после 6 попыток: {last}")


def main() -> int:
    with open(SNAP, encoding="utf-8") as fh:
        snap = json.load(fh)
    cats = sorted({c.lower().strip()
                   for rec in snap.values() for c in rec["categories"]})
    print(f"слов: {len(snap)}, уникальных категорий для снятия родителей: "
          f"{len(cats)}")

    parents = {}
    if os.path.exists(OUT):                         # докачиваем, а не начинаем заново
        with open(OUT, encoding="utf-8") as fh:
            parents = json.load(fh)
        print(f"  уже снято: {len(parents)}")

    todo = [c for c in cats if c not in parents]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        got = parents_batch(chunk)
        for title, par in got.items():
            parents[title.lower()] = [p for p in par]
        # категории, по которым Вики не вернула страницу, помечаем пустым
        # списком, иначе они вечно остаются в todo и запрос повторяется
        seen = {t.lower() for t in got}
        for c in chunk:
            parents.setdefault(c, parents.get(c, []))
        print(f"  батч {i // BATCH + 1}: страниц {len(got)}, всего {len(parents)}")
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(parents, fh, ensure_ascii=False, indent=1)
        time.sleep(6)

    empty = sum(1 for v in parents.values() if not v)
    print(f"ИТОГО родителей снято для {len(parents)} категорий, "
          f"из них без родителей: {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
