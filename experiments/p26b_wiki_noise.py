"""
ЗАМЕР п.26-б: чем именно является иерархия Вики и что из неё попадает в ответ.

Данные: experiments/wiki_category_snapshot.json — сырые категории 22 реальных
русских слов, полученные ТЕМ ЖЕ запросом, который шлёт канал
teachers/teacher_wiki.py::_fetch_page (prop=extracts|categories, exintro,
cllimit=50, redirects=1, без clshow). Снимок нужен потому, что Вики отвечает
429 при повторных замерах, а вывод обязан воспроизводиться офлайн.

Вопросы замера:
  1. В каком порядке Вики отдаёт категории? (проверка докстринга канала,
     который утверждает «от абстрактных к конкретным — обращаем порядок»)
  2. Какая доля категорий — служебный мусор, и проходит ли он фильтры?
  3. Кто оказывается hierarchy[0], то есть тем, чем мозг ОТВЕТИТ на
     «что такое X», если определения в записи нет?
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.records import meaningful_categories          # noqa: E402
from teachers.teacher_wiki import WikiTeacher           # noqa: E402

SNAP = os.path.join(ROOT, "experiments", "wiki_category_snapshot.json")


def main() -> int:
    with open(SNAP, encoding="utf-8") as fh:
        snap = json.load(fh)
    wt = WikiTeacher.__new__(WikiTeacher)

    n_words = n_multi = 0
    n_alphabet_first = 0        # visible[0] совпадает с алфавитным первым
    n_service_leak = 0          # в видимую иерархию просочилась разметка статьи
    n_empty = 0                 # отвечать нечем (иерархия съедена самоименем)
    rows = []

    for word, rec in sorted(snap.items()):
        title = rec["title"]
        raw = rec["categories"]
        hier = wt._categories_to_hierarchy(raw)          # фильтр канала
        vis = meaningful_categories(title, hier)         # фильтр ответа
        n_words += 1

        low = [c.lower() for c in vis]
        if low and low[0] == sorted(low)[0]:
            n_alphabet_first += 1
        if vis and len(vis) > 1:
            n_multi += 1
        if not vis:
            n_empty += 1
        # Служебная разметка = то, что описывает статью, а не предмет.
        service = [c for c in vis if any(
            k in c for k in ("стать", "ссылк", "викисловар", "викисклад",
                             "викитек", "викитатник", "isbn", "страниц",
                             "неупорядоченн", "шаблон"))]
        if service:
            n_service_leak += 1
        rows.append((word, title, len(raw), len(hier), vis, service))

    print("═" * 78)
    print("ЗАМЕР 1. Порядок категорий, данных Вики (22 слова)")
    print("═" * 78)
    for word, title, n_raw, n_hier, vis, service in rows:
        head = vis[0] if vis else "НЕТ ДАННЫХ (иерархия съедена формой самого слова)"
        mark = "  <== МУСОР" if service else ""
        print(f"  {word:<12} -> «{title} — это {head}»{mark}")
        if vis:
            print(f"  {'':12}    сырых={n_raw:<3} после фильтра канала={n_hier:<3} "
                  f"видно в ответе={len(vis)}: {vis}")

    print()
    print("═" * 78)
    print("СЧЁТ")
    print("═" * 78)
    print(f"  слов: {n_words}")
    print(f"  слов, где visible[0] = АЛФАВИТНО первая категория: "
          f"{n_alphabet_first}/{n_words}")
    print(f"  слов с >1 категорией (есть из чего выбирать): {n_multi}/{n_words}")
    print(f"  слов, где в ответ просочилась разметка статьи: "
          f"{n_service_leak}/{n_words}")
    print(f"  слов, где отвечать нечем (иерархия пуста после фильтра): "
          f"{n_empty}/{n_words}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
