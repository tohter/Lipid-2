"""ВИДИТ ЛИ КАНАЛ ВИКИ ТО, ЧТО НАПИСАНО В ВИКИПЕДИИ (честная декомпозиция).

ЗАЧЕМ. Хозяин хочет сравнивать локалку с Вики («а Вики подтвердит?»). Мерить
такое сравнение можно только после того, как известно, ПОЧЕМУ канал молчит.
Иначе мы припишем Википедии собственную слепоту — что я и сделала в первой
версии этого зонда (см. «ЧЕМ Я ОШИБЛАСЬ» ниже).

ПЕРВЫЙ ЗОНД МЕРЯЛ НЕ ТО. Он брал СЫРОЙ текст `_fetch_page` и делил его по ". ".
Живой канал так не работает: `fetch_concept` идёт через `_first_sentence`,
который (учителя/teacher_wiki.py:319) снимает ПОСТАНОВЫЕ УДАРЕНИЯ (U+0301) и
ищет конец предложения, а не первую точку. ЗАМЕРЕНО на статье «Курица»:
с ударениями экстрактор даёт «рица --HAS_PROPERTY--> шний», без них —
«курица --HAS_PROPERTY--> домашний». То есть 5 ударных знаков в статье
ломали разбор, а ломали их я, минуя живую обработку.

ЗДЕСЬ ТРИ ОТДЕЛЬНЫХ ВЕРДИКТА, и они означают разное:
  А) ТЕКСТА НЕТ      — статья не найдена, либо это подмена темы («пингвин» ->
                       страница значений «Пингвиновые»), либо объект в статье
                       не упоминается. Это ограничение ИСТОЧНИКА.
  Б) КАНАЛ НЕ ЗАПИСАЛ — живой fetch_concept + write_to_world прошли, а связи
                       в мире нет. Это ограничение НАШЕЙ обработки Вики.
  В) СВЯЗЬ ПОДТВЕРЖДАЕТ — связь есть в мире, который канал оставил. Это и есть
                       настоящее подтверждение, его и надо ставить рядом с
                       вердиктом локалки.
Дополнительно меряется СЫРОЙ путь (как делала я раньше), чтобы видеть, сколько
молчания было виной зонда, а не канала.

НИЧЕГО НЕ ПИШЕТ в живой мир: каждый прогон в mktemp.

Запуск:  LIPID2_STATUS=0 .venv/bin/python experiments/wiki_channel_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lang.ru.lemma import install                      # noqa: E402
install()

from core.extract import SimpleExtractor               # noqa: E402
from core.morph import split_sentences                 # noqa: E402
from teachers.teacher_wiki import WikiTeacher          # noqa: E402

# (субъект, корень объекта, что надеемся увидеть)
CHECKS = [
    ("кит", "млекопитающ", "IS_A"),
    ("курица", "птиц", "IS_A"),
    ("огонь", "туши", "CAN_DO"),
    ("солнце", "восход", "HAS_PROPERTY"),
    ("вода", "кипяти", "HAS_PROPERTY"),
]


def strip_stress(text: str) -> str:
    """Снимает постановые ударения (U+0301) — как это делает живой канал."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def world_links(world: dict, head: str) -> list:
    """Связи, которые РАССУЖДЕНИЕ увидит в мире после канала (тот же
    механизм, что StaticKnowledge: hierarchy -> IS_A, properties, actions)."""
    out = []
    ent = {}
    for key, val in (world.get("entities") or {}).items():
        if str(key).lower().startswith(head):
            ent = val or {}
            break
    for h in (ent.get("hierarchy") or []):
        out.append((head, "IS_A", str(h)))
    for p in (ent.get("properties") or []):
        out.append((head, "HAS_PROPERTY", str(p)))
    for a in (ent.get("actions") or []):
        name = a.get("name") if isinstance(a, dict) else str(a)
        if name:
            out.append((head, "CAN_DO", str(name)))
    for key, v in (world.get("observations") or {}).items():
        parts = str(key).split("|")
        if len(parts) == 3 and parts[0] == head:
            out.append((parts[0], parts[1], parts[2]))
    return out


def raw_verdict(ext: SimpleExtractor, text: str, subj: str, stem: str):
    """Что экстрактор видит в СЫРОМ тексте (так меряла первая версия зонда)."""
    for sent in split_sentences(text):
        if len(sent) < 15:
            continue
        for f in ext.extract(sent).facts:
            if (subj in f.head.lower() and stem in f.tail.lower()):
                return True, f"{f.head} --{f.relation}--> {f.tail}"
    return False, ""


def main() -> int:
    ext = SimpleExtractor()
    tally = {"A": 0, "B": 0, "C": 0, "raw_only": 0}
    print("\n" + "=" * 76)
    print("КАНАЛ ВИКИ: источник молчит, или молчим мы")
    print("=" * 76)

    for subj, stem, want in CHECKS:
        # КАЖДЫЙ прогон в своём каталоге: живой мир не трогается, и канал не
        # видит запись, оставленную предыдущей проверкой.
        room = tempfile.mkdtemp()
        wiki = WikiTeacher(memory_dir=room)
        page = wiki._fetch_page(subj)
        print(f"\n── {subj} ~ «{stem}» (ожидаем {want})")
        if not page:
            tally["A"] += 1
            print("   А) ИСТОЧНИК: статья не найдена")
            continue
        title, raw, _ = page
        raw = raw or ""
        # подмена темы: страница значений или статья о другом
        if subj[:4].lower() not in str(title).lower():
            tally["A"] += 1
            print(f"   А) ИСТОЧНИК: подмена темы — статья «{title}»")
            continue

        got_raw, why_raw = raw_verdict(ext, raw, subj, stem)
        got_clean, why_clean = raw_verdict(ext, strip_stress(raw), subj, stem)

        learned = wiki.fetch_concept(subj)
        wrote = bool(learned and wiki.write_to_world(learned))
        world = json.load(open(os.path.join(room, "lipid_world.json"))) \
            if wrote and os.path.exists(os.path.join(room, "lipid_world.json")) \
            else {}
        links = world_links(world, subj)
        confirm = [l for l in links if stem in l[2].lower()]

        print(f"   сырой текст -> экстрактор: "
              f"{'ЕСТЬ ' + why_raw if got_raw else 'нет'}")
        print(f"   тот же текст БЕЗ ударений -> экстрактор: "
              f"{'ЕСТЬ ' + why_clean if got_clean else 'нет'}")
        print(f"   живой канал записал: {wrote}; связей у «{subj}»: {len(links)}")
        for l in links[:4]:
            print(f"      {l[0]} --{l[1]}--> {l[2]}")
        if confirm:
            tally["C"] += 1
            print(f"   В) ПОДТВЕРЖДАЕТ: {confirm[0]}")
        elif got_clean or got_raw:
            tally["B"] += 1
            print("   Б) ТЕКСТ ГОВОРИТ, А КАНАЛ НЕ ЗАПИСАЛ — чиним наш разбор")
        else:
            tally["A"] += 1
            print("   А) ИСТОЧНИК: утверждения в статье нет")
        if got_clean and not got_raw:
            tally["raw_only"] += 1
            print("   !!! первая версия зонда ошибалась: молчание было виной "
                  "УДАРЕНИЙ в моём тексте, не канала")

    print("\n" + "=" * 76)
    print(f"   А) источник молчит: {tally['A']}")
    print(f"   Б) текст есть, мы не записали: {tally['B']}")
    print(f"   В) канал ПОДТВЕРДИЛ: {tally['C']}")
    print(f"   из них молчание было артефактом моего зонда (ударения): "
          f"{tally['raw_only']}")
    print("   Вывод: сравнивать каналы можно только по строке В; строка Б —")
    print("   работа над нами, строка А — честно про источник.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
