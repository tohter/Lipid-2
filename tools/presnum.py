#!/usr/bin/env python3
"""Цифры витрины PRESENTATION.md снимаются здесь, а не в голове у автора.

ЗАЧЕМ. Очередь (п. ПРЕЗ, 31.10) зафиксировала: витрина была написана 23.10
«по 516 зелёным тестам» и с тех пор чисел не видела. Замер 31.10 дал 748
тестов и 87 коммитов, в витрине же сидели 516, «503 в 25 файлах», «12 586
строк ядра», «27 292 всего», «83 файла», «50 коммитов» и список из 13
отношений при 17 в реестре. Витрина продавала состояние десятидневной
давности как сегодняшнее. Тот же класс вранья, из-за которого в проекте
стоит трещотка границы языка: «число, которого нет в законе, всегда можно
сочинить» (memory.txt, урок 28).

ЛЕЧИТСЯ НЕ ручной подгонкой. Цифры переснимаются этим инструментом, а
tests/test_presentation_numbers_contract.py требует, чтобы у каждого числа
блока был измеритель здесь. Вписал в витрину число без замера -> тест
красный. То есть сочинённое число стало физически невписуемым.

ОДНО МЕСТО ОДНОЙ ПРАВДЕ. Измеренное число живёт ровно в одном месте. «516»
стояло и в шапке, и в блоке «Цифры» — и на следующем шаге устарели ОБЕ
копии, причём правка одной вторую не лечит. Поэтому шапка имеет право только
на дату, а размер в прозе вне блока запрещён проверкой.

МАРКЕРЫ, А НЕ ЗАГОЛОВКИ. Генерируемые куски ищутся по HTML-комментариям
`<!--presnum:...-->`: заголовок «## 9. Цифры» можно переформулировать, и
тогда проверка тихо перестанет что-либо видеть (этот случай есть в тесте
как отдельная мутация). Нет маркера -> проверка красная, а не зелёная.

    ПОРЯДОК ОБЯЗАТЕЛЕН: сначала правится код, ПОТОМ `--fix`, ПОТОМ прогон
    тестов. Наоборот нельзя: витрина мерит дерево, а правка любого .py это
    дерево меняет — зафиксированное число устареет в тот же миг. Это не
    придирка к порядку, а следствие того, что витрина мерит живой
    код, а не воспоминание о нём. Отсюда же, чего в блоке нет: числа КОММИТОВ —
    оно тикает от самого коммита, которым его вписали (зафиксировал 87 → стал
    88), а починка требует нового коммита, который сдвигает число снова. Замык,
    который нельзя сделать зелёным, эквивалентен всегда красному и просто
    отключается.

    .venv/bin/python tools/presnum.py            # что мерится сегодня
    .venv/bin/python tools/presnum.py --check    # врёт ли витрина (rc=1 = врёт)
    .venv/bin/python tools/presnum.py --fix      # перегенерировать блоки
    .venv/bin/python tools/presnum.py --print    # готовый блок чисел
"""
import os
import re
import sys
import unittest
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DOC = os.path.join(ROOT, "PRESENTATION.md")

# Куда не смотрим при подсчёте строк/файлов. Совпадает со SKIP_DIRS в
# tests/test_static_hygiene.py: одна мера на проект, а не две похожие.
# legacy/ вне счёта сознательно — мёртвый код (там же, в test_static_hygiene,
# живёт и правило).
SKIP_DIRS = {".venv", ".git", "__pycache__", "data", "legacy", "build", "dist"}


# ────────────────────────── МЕРЫ ──────────────────────────
#
# Мемоизация. Замер unittest-сборки и AST-обход дерева дороги: без кэша одна
# сверка витрины стоила ~4 с, а весь замок — 406 с, т.е. ДОРОЖЕ полного прогона
# тестов. Замок дороже того, что охраняет, не будут запускать вообще — это
# второй способ убить проверку после «зелёного от незнания». В пределах одного
# процесса состояние проекта не меняется, поэтому замеры кэшируются.


def memo(fn):
    cache = {}

    def wrapper(*a):
        if a not in cache:
            cache[a] = fn(*a)
        return cache[a]
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper

# Единственный источник чисел для витрины. Ключ словаря = строка в блоке
# «Цифры»; нет ключа -> нет и права на число в документе.

@memo
def py_tree(sub=None):
    """(строк, файлов) в python-дереве.

    Построчный подсчёт, как в memory.txt: `cat *.py | wc -l` недосчитывает
    файлы без перевода строки в конце (для core расхождение 8076/8080 —
    ровно тот случай, когда «та же команда» даёт другую правду).
    """
    files = []
    base = os.path.join(ROOT, sub) if sub else ROOT
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                files.append(os.path.join(dirpath, name))
    lines = 0
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            lines += sum(1 for _ in fh)
    return lines, len(files)


@memo
def n_tests(pattern="test_*.py"):
    """countTestCases БЕЗ прогона: сборка занимает доли секунды, прогон —
    130 с. Это замер сегодняшнего дня, а не воспоминание о вчерашнем прогоне.
    «OK» прогона заперт остальными тестами."""
    suite = unittest.TestLoader().discover(os.path.join(ROOT, "tests"),
                                           pattern=pattern)
    return suite.countTestCases()


def n_contract_files():
    return len([f for f in sorted(os.listdir(os.path.join(ROOT, "tests")))
                if f.startswith("test_") and f.endswith("_contract.py")])


@memo
def relation_names():
    from core.storage import RelationType
    return [e.name for e in RelationType]


@memo
def relations():
    """(всего в реестре, порождено экстрактором).

    Порождаемость берётся из МЕРЫ tools/relcov.py, а не из README: память
    проекта знает случай, когда документ говорил «13 отношений», реестр — 17,
    а порождалось 14.
    """
    import tools.relcov as relcov
    # Вызывается МЕРА relcov, а не пересказывается здесь. Второй цикл по тем
    # же CASES был бы вторым измерителем одной границы — а две правды в этом
    # проекте уже были (документ «13 отношений», реестр 17). Мера уточнена
    # 31.10: ABOUT рождается, SYNONYM_OF приезжает каналом aliases, и обоим
    # presnum обязан поверить, а не сохранить прежние 14.
    return len(relation_names()), len(relcov.produced_relations())


@memo
def langborder(key):
    """Одна цифра границы языка. Переиспользует tools/langborder.py.

    Второй измеритель на ту же границу был бы второй правдой — а две правды
    в проекте уже были: документ говорил «13 отношений», реестр — 17.
    """
    from tools.langborder import measure
    return measure()["totals"][key]


def core_seams():
    """Швов языка ВНУТРИ ядра. Ноль — это итог A2.4, а не пустая мера."""
    return langborder("S_seams_inside_core")


MEASURES = {
    "тестов всего":                  n_tests,
    "контрактных тестов":            lambda: n_tests("test_*_contract.py"),
    "файлов с тестами":              lambda: py_tree("tests")[1],
    "файлов контрактных тестов":     n_contract_files,
    "отношений в реестре":           lambda: relations()[0],
    "отношений порождает экстрактор": lambda: relations()[1],
    "шов языка внутри ядра":         core_seams,
    "файлов ядра со следами языка":  lambda: langborder("D_debt_files"),
    "русских литералов в core/":     lambda: langborder("R_core_ruslits"),
    "методов ядра с русским выводом": lambda: langborder("O_output_methods"),
    "файлов языка в lang/":          lambda: py_tree("lang")[1],
    "строк в core/":                 lambda: py_tree("core")[0],
    "строк в brain/":                lambda: py_tree("brain")[0],
    "строк ядра (core+brain)":       lambda: py_tree("core")[0]
                                        + py_tree("brain")[0],
    "строк в tests/":                lambda: py_tree("tests")[0],
    "строк Python всего":            lambda: py_tree()[0],
    "файлов Python всего":           lambda: py_tree()[1],
}

# ПОЧЕМУ ЗДЕСЬ НЕТ ЧИСЛА КОММИТОВ. Оно тикает от самого коммита, которым ты
# его вписал: зафиксировал 87 — стал 88, витрина снова красная. Починка
# требует нового коммита, который снова сдвигает число: регресс, а не проверка.
# Любой СОБЫТИЙНЫЙ показатель делает замок невыполнимым, а невыполнимый замок
# эквивалентен всегда красному и отключается. Витрина обязана измеряться
# СОДЕРЖИМЫМ дерева — его коммит не меняет, поэтому цифры остаются зелёными и
# до, и после коммита. Число коммитов живёт в `git log` и в memory.txt.


# ──────────────────── ГЕНЕРИРУЕМЫЕ КУСКИ ────────────────────

MARK_NUM = "<!--presnum:numbers-->"
MARK_REL = "<!--presnum:relations-->"

HEADER_RE = re.compile(r"\*[\s\S]{0,120}?составлено[^*]*\*", re.IGNORECASE)
# ЧИСЛА ТЕСТОВ В ШАПКЕ БЫТЬ НЕ МОЖЕТ — не «устарело», а запрещено. Дубль
# «516» в шапке и в блоке протух целиком; правка одной копии вторую не лечит.
HEADER_WANT = ("*составлено {date}. Числа витрины снимает измерением "
               "`tools/presnum.py`; правь инструмент, а не цифру глазами*")


def fmt(n):
    """35 435 — разрядный разделитель, как принято в витрине."""
    return f"{n:,}".replace(",", " ")


def today_ru():
    return date.today().strftime("%d.%m.%Y")


def render_block():
    """Fenced-блок с числами. Витрина обязана содержать его побайтно.

    Побайтность — не педантизм: правка одной цифры на глаз это ровно то, как
    витрина и протухла.
    """
    vals = {k: m() for k, m in MEASURES.items()}
    w = max(len(k) for k in vals)
    body = "\n".join(f"{k + ':':{w}} {fmt(v)}" for k, v in vals.items())
    return "```\n" + body + "\n```"


def render_relations():
    """Fenced-блок со списком отношений. Перечень берёт из реестра.

    До 31.10 в витрине лежало 13 имён при 17 в реестре (не хватало SYNONYM_OF,
    CAUSES, IS_ROLE, HAS_PART), и число «17 отношений» этого НЕ выдавало:
    поэтому сверяется САМ СПИСОК, а не длина рядом с ним.
    """
    return "```\n`" + " / ".join(relation_names()) + "`\n```"


# ──────────────────── ЧТЕНИЕ ДОКУМЕНТА ────────────────────

LINE_RE = re.compile(r"^([^\s:][^:]*?):\s+(\d[\d \u00a0]*)(?:\s+\S.*)?$")


def doc_text():
    with open(DOC, encoding="utf-8") as fh:
        return fh.read()


def fenced_span(text, marker):
    """(start, end) блока ``` сразу за маркером. None, если блока там нет.

    «Сразу» принципиально: первая версия fix() искала ближайший ``` в documento
    и, когда после маркера решётки не оказывалось, молча ПЕРЕЗАПИСЫВАЛА соседний
    блок — инструмент, который портит документ вместо того чтобы отказаться,
    опаснее отсутствия инструмента.
    """
    i = text.find(marker)
    if i < 0:
        return None
    j = i + len(marker)
    while j < len(text) and text[j] in " \t\r\n":
        j += 1
    if not text.startswith("```", j):
        return None
    end = text.find("```", j + 3)
    if end < 0:
        return None
    return j, end + 3


def block_after(text, marker):
    """Текст блока после маркера, None если чего-то нет.

    None, а не исключение: check() обязан ДИАГНОСТИРОВАТЬ пропавший маркер, а
    не падать на нём.
    """
    span = fenced_span(text, marker)
    return None if span is None else text[span[0]:span[1]]


def doc_numbers():
    """{ключ: число} из блока «Цифры». {} если блока нет."""
    text = doc_text()
    blk = block_after(text, MARK_NUM)
    if blk is None:
        return {}
    found = {}
    for line in blk.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            found[m.group(1).strip()] = int(
                m.group(2).replace(" ", "").replace("\u00a0", ""))
    return found


def doc_relations():
    blk = block_after(doc_text(), MARK_REL)
    return blk


DATE_RE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{2,4}")


def doc_header_count():
    """Число в шапке, если шапка его несёт. Наличие — всегда нарушение.

    Дату («31.10.2026») вырезаем до поиска: иначе проверка краснела бы на
    легальную дату, а трещотка на правду начинает обходиться.
    """
    m = HEADER_RE.search(doc_text())
    if not m:
        return None
    head = DATE_RE.sub("Дата", m.group(0))
    n = re.search(r"(?<![\d.])(\d[\d \u00a0]{0,9}\d)(?![\d])", head)
    if not n:
        return None
    return int(n.group(1).replace(" ", "").replace("\u00a0", ""))


# ──────────────────── ПРОВЕРКА ────────────────────

# Число вне блока подозрительно только ТОГДА, когда стоит рядом с тем, что
# его мерит. Без этого правила проверка ловила «Gemma @8080» как «8 080
# строк ядра» — ложное срабатывание на номер порта. Трещотка, которая орёт
# на косметику, начинает обходиться (memory.txt, урок A0), поэтому правило
# привязано к существительному, а не к совпадению цифр вообще.
COUNT_WORDS = ("тест", "строк", "строки", "файлов", "файла", "коммитов",
               "коммита", "отношений", "отношения", "швов", "шов",
               # латиница: в разделе «как проверить» число пряталось в
               # комментарии `# 516 OK`, и правило на кириллице его не видело
               "test", "tests", "line", "lines", "file", "files", "commit")

# Ниже этого числа копии размеров не ищем: 20 активных бит, порог 8, вес 150,
# пороги 13/17 — замеры по смыслу, а не витринные размеры.
STALE_MIN = 100


def stale_copies():
    """[(строка прозы, число)] — размер в прозе вне генерируемых блоков.

    Значение СОЗНАТЕЛЬНО не сверяется с замером: сам факт второй копии — уже
    нарушение, и чем копия меньше, тем быстрее станет враньём.
    """
    text = doc_text()
    body = text.replace(render_block(), "")
    rel = render_relations()
    body = body.replace(rel, "")
    hits, seen = [], set()
    num = re.compile(r"(?<![\d.\u0430-я])"
                     r"(\d[\d \u00a0]{0,11}\d|\d)")
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        for m in num.finditer(s):
            raw = m.group(1).strip(" \u00a0")
            try:
                val = int(raw.replace(" ", "").replace("\u00a0", ""))
            except ValueError:
                continue
            if val < STALE_MIN:
                continue
            around = (s[max(0, m.start() - 16):m.end()] +
                      s[m.end():m.end() + 16]).lower()
            if any(w in around for w in COUNT_WORDS) and s not in seen:
                seen.add(s)
                hits.append((s[:110], val))
    return hits


def check():
    """(список расхождений, {ключ: (в витрине, замерено)})."""
    live = {k: m() for k, m in MEASURES.items()}
    got = doc_numbers()
    bad, table = [], {}

    text = doc_text()
    for mark, name in ((MARK_NUM, "чисел"), (MARK_REL, "отношений")):
        if mark not in text:
            bad.append(f"в витрине нет маркера {mark} — блок {name} больше "
                       f"не генерируется и не проверяется. Тихая потеря "
                       f"защиты опаснее красного теста.")

    for k, v in live.items():
        table[k] = (got.get(k), v)
        if k not in got:
            bad.append(f"«{k}»: в витрине числа НЕТ, замерено {fmt(v)}")
        elif got[k] != v:
            bad.append(f"«{k}»: витрина говорит {fmt(got[k])}, "
                       f"замер — {fmt(v)}")
    for k in got:
        if k not in live:
            bad.append(f"«{k}»: число в витрине без измерителя в "
                       f"tools/presnum.py — такого числа быть не может")

    if block_after(text, MARK_NUM) != render_block():
        bad.append("блок «Цифры» побайтно отличается от вывода "
                   "`tools/presnum.py --print`: перегенерируй блок целиком "
                   "(--fix), а не правь одну цифру на глаз")

    if block_after(text, MARK_REL) != render_relations():
        bad.append("список отношений в витрине расходится с реестром "
                   "core.storage.RelationType — перегенерируй (--fix). "
                   "Если причина в том, что блока после маркера нет, так и "
                   "верни ему блок, а не правь список глазами")

    n = doc_header_count()
    if n is not None:
        bad.append(f"шапка витрины несёт число ({fmt(n)}): измеренное число "
                   f"живёт ровно в одном месте, в блоке «Цифры». В шапке — "
                   f"только дата (--fix)")

    for line, val in stale_copies():
        bad.append(f"в прозе вне блоков стоит размер {val}: «{line}» — "
                   f"измеренное число живёт в одном месте, иначе следующим "
                   f"шагом устареет вторая копия")
    return bad, table


def fix():
    """Перегенерировать витрину по живому замеру: блоки и дату шапки.

    Только те места, где текст является ВЫВОДОМ замера. Прозу не трогаем:
    убрать число из предложения — решение человека, отдашь это машине и
    инструмент начнёт незаметно портить смысл.
    """
    text = doc_text()
    for marker, renderer in ((MARK_NUM, render_block),
                             (MARK_REL, render_relations)):
        if marker not in text:
            continue                      # отсутствие ловит check(), не fix()
        span = fenced_span(text, marker)
        if span is None:
            raise SystemExit(
                f"ОСТАНОВКА: сразу за {marker} нет блока ```. Ничего не "
                f"переписано. Вставлять блоки нужно в правильных местах — "
                f"иначе правка ушла бы в соседний блок.")
        start, end = span
        text = text[:start] + renderer() + text[end:]
    if HEADER_RE.search(text):
        text = HEADER_RE.sub(
            HEADER_WANT.replace("{date}", today_ru()), text, count=1)
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--print":
        print(render_block())
        return 0
    if mode == "--fix":
        fix()
        bad, _ = check()
        print("блоки витрины перегенерированы по живому замеру")
        if bad:
            print(f"осталось расхождений: {len(bad)} "
                  f"(это проза — правится руками)")
            for b in bad:
                print("  -", b)
            return 1
        print("расхождений нет")
        return 0
    if mode == "--check":
        bad, _ = check()
        if bad:
            print("ВИТРИНА ВРЁТ:")
            for b in bad:
                print("  -", b)
            return 1
        print("витрина сверена с замером: расхождений нет")
        return 0
    for k, m in MEASURES.items():
        print(f"{k:38s} {fmt(m())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
