#!/usr/bin/env python
"""
ГРАНИЦА ЯЗЫКА: кто и где смешивает русский с языконейтральным ядром.

ЗАКОН ГРАНИЦЫ (формулировка — experiments/IR_BOUNDARY.md, здесь только её
проверяемая часть):

    текст -> lang/<xx> -> IR-НА-ВХОД -> core -> IR-С-ВЫХОДА -> lang/<xx> -> текст

core обязан знать ТОЛЬКО IR. Русский язык для него — чужие байты. Всё, что
знает о падежах, предлогах и окончаниях, обязано жить в lang/ru.

ПОЧЕМУ ИНСТРУМЕНТ, А НЕ СПИСОК В ДОКУМЕНТЕ. Список умирает молча: правка
добавляет импорт, и никто не замечает, пока не выясняется, что ядро нельзя
переставить на другой язык. Здесь же число по каждой категории ПЕЧАТАЕТСЯ, а
замок (tests/test_langborder_contract.py) даёт ему ТРЕЩОТКУ: число может УМЕНЬ-
ШИТЬСЯ без вопросов, но РАСТИ не может — рост это падение теста.

ПОЧЕМУ AST, А НЕ GREP. Тот же класс подмены, что уроки 20 и 24: grep по слову
"morph" считает и комментарии, и докстринги, и локальные импорты внутри функ-
ций, и строки-сообщения. Он не отличает «код дёрнул русский модуль» от «в
докстринге упомянут русский модуль». Отличает только дерево разбора.

ЧЕТЫРЕ КАТЕГОРИИ (важно: они НЕ равноценны, и сваливать их в один общий
индекс нельзя — это ровно та ошибка, что п.21 с notes):

  1. DEBT   — языковой модуль лежит ВНУТРИ core (core.morph, core.lemma,
              core.extract). Это и есть переносимая часть будущего lang/ru.
  2. SEAM   — ядро или мозг ИМПОРТИРУЕТ язык. Ровно те места, которые обязан
              закрыть IR; их число и есть размер работы.
  3. OUTPUT — метод возвращает RUССКУЮ СТРОКУ потребителю. Опаснее импортов:
              импорт видно глазами, а строка-ответ распространяется тестами и
              шаблонами, и однажды «core возвращает ответ» становится неотли-
              чим от «core думает по-русски».
  4. RUSLIT — русские литералы в core/ (не докстринги). Мера инерции: пока их
              сотни, «core языконейтрален» — утверждение без опоры.

Запуск:
    .venv/bin/python tools/langborder.py            # отчёт
    .venv/bin/python tools/langborder.py --json /tmp/b.json
"""
import argparse
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Кириллица в строке. Именно наличие кириллицы, а не «не ASCII»: латинские
# термы (RelationType, LOCATED_IN) — это НЕ язык, они и должны остаться.
RU = re.compile(r"[\u0430-\u044f\u0410-\u042f\u0451\u0401]")

# Языковые модули. Явный список, а не «в имени есть morph»: молчаливое
# рассширение списка под видом рефакторинга — тот же обман, что молчаливое
# расширение словаря отношений.
LANGUAGE_MODULES = {
    "core.morph", "core.lemma", "core.extract",
    # внешний морфологический движок — язык в чистом виде
    "pymorphy3", "pymorphy3.unit",
}
# Ядро НЕ должно их знать. brain/ имеет право (это его работа — говорить с
# человеком), но обязан делать это ЧЕРЕЗ lang/, а не руками.
LANG_DIR = "lang/"
CORE_PACKAGES = ("core",)
BRAIN_PACKAGES = ("brain",)
# Куда язык ИМЕЕТ ПРАВО лежать (шаг A1/A2). Файлы отсюда — язык по месту
# жительства, а не должник внутри ядра.
LANGUAGE_PACKAGES = ("lang",)
# ИНСТРУМЕНТ: не ядро и не мозг, но и НЕ самопроверка. У tools/ и teachers/
# main() — рабочий путь (его зовут из командной строки), поэтому demo_spans
# вызывается для них со script=True. Сканируются СПРАВОЧНО, категории U, без
# потолка: инструмент имеет право знать язык, но шов, переехавший в tools/, не
# обязан становиться невидимым (ровно так A1 сделал невидимым импорт, переехав
# в lang/). Поднимать из-за этого S_seam_sites нельзя: рост потолка — решение
# хозяина, не моё.
SCRIPT_PACKAGES = ("tools", "teachers")


def py_files(*packages):
    for pkg in packages:
        base = os.path.join(ROOT, pkg)
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for f in sorted(fns):
                if f.endswith(".py"):
                    yield os.path.relpath(os.path.join(dp, f), ROOT)


def parse(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        src = fh.read()
    return ast.parse(src, filename=rel)


def docstring_ids(tree):
    """id() узлов-докстрингов: они НЕ являются данными и не считаются."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            body = getattr(n, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


def imports_of(tree):
    """(имя_модуля, номер_строки, внутри_ли_функции) по всем import-формам."""
    out = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0

        def visit_FunctionDef(self, node):
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1
        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for a in node.names:
                out.append((a.name, node.lineno, self.depth > 0))
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            mod = node.module or ""
            if node.level:                       # from . import morph
                mod = "." * node.level + mod
            out.append((mod, node.lineno, self.depth > 0))
            # from core.morph import Morph -> имя модуля уже в node.module
            self.generic_visit(node)
    V().visit(tree)
    return out


def module_root(name, current_file):
    """Привести относительный импорт к полному имени пакета."""
    if not name.startswith("."):
        return name
    pkg = os.path.dirname(current_file).replace(os.sep, ".")
    stripped = name.lstrip(".")
    up = len(name) - len(stripped) - 1
    parts = pkg.split(".")
    return ".".join(parts[:len(parts) - up] + ([stripped] if stripped else []))


def lang_imports(rel, script=False):
    """
    Швы: где не-языковой код тянет язык.

    КАЖДЫЙ шов помечается, лежит ли он ВНУТРИ демо-самопроверки
    (`if __name__ == "__main__"`). Это не снисхождение, а та же логика, что
    уже применена к кириллице: цикл обязан кормиться русским текстом, иначе
    самопроверка ничего не проверяет, но в библиотечный путь этот импорт не
    попадает. Без разделения A2 физически не смог бы довести S_seams_inside_core
    до нуля: в core/cortex.py русский текст стоит именно в демо.
    """
    tree = parse(rel)
    spans = demo_spans(tree, script=script)
    hits = []
    for mod, lineno, nested in imports_of(tree):
        full = module_root(mod, rel)
        # pymorphy3 может быть вложен в составное имя (pymorphy3.unit)
        cand = {full, full.split(".")[0]}
        if cand & LANGUAGE_MODULES or any(
                c.startswith("pymorphy3") for c in cand):
            hits.append({"module": full, "line": lineno, "nested": nested,
                         "demo": in_span(lineno, spans)})
    return hits


def is_language_file(rel):
    """Язык по имени модуля ИЛИ по месту жительства (lang/<xx>/...)."""
    norm = rel.replace(os.sep, "/")
    if norm.startswith(LANG_DIR):
        return True
    return module_root(rel[:-3].replace(os.sep, "."), rel) in LANGUAGE_MODULES \
        or rel.replace(os.sep, ".")[:-3] in LANGUAGE_MODULES


def ru_strings(node, docs):
    """Есть ли внутри выражения русская строка, не являющаяся докстрингом."""
    for k in ast.walk(node):
        if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                and RU.search(k.value) and id(k) not in docs):
            return k.value
    return None


def output_methods(rel):
    """
    ПУБЛИЧНЫЕ методы, возвращающие русскую строку. Приватные (_...): исключены
    намеренно: они не граница, а внутренность, и ловить их — значит раздуть
    отчёт до бесполезного. Публичный метод, отдающий текст, — это и есть дыра
    в границе: её видно пользователю.
    """
    tree = parse(rel)
    docs = docstring_ids(tree)
    out = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [m for m in cls.body
                   if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            if fn.name.startswith("_"):
                continue
            for r in [x for x in ast.walk(fn)
                      if isinstance(x, ast.Return) and x.value is not None]:
                if ru_strings(r.value, docs):
                    out.append({"cls": cls.name, "method": fn.name,
                                "line": r.lineno})
                    break
    return out


# Функции, в которые русская строка приходит как СООБЩЕНИЕ ЧЕЛОВЕКУ. Строка
# внутри них на поведение программы НЕ ВЛИЯЕТ: её перевод — косметика.
DIAGNOSTIC_FUNCS = {"print", "warn", "info", "error", "debug", "_status",
                    "log", "warning", "exception"}


def demo_spans(tree, script=False):
    """
    Диапазоны самопроверки: `if __name__ == "__main__"` и def main()/demo().

    ПОЧЕМУ ЗДЕСЬ ПАРАМЕТР, а не одно правило на всех. Блок `__name__` — это
    самопроверка в ЛЮБОМ файле. А `def main()` — самопроверка ТОЛЬКО внутри
    библиотеки (core/, brain/): у ИНСТРУМЕНТА (tools/, teachers/) main() есть
    рабочий путь, его вызывают люди командной строкой, и «перенести импорт в
    main()» там не делает код чистым. Помечать golden_corpus.main() демо-блоком
    значило бы нарисовать ноль швов блоком, который реально работает, — ровно
    тот дефект, от которого эта категория защищает.
    """
    spans = []
    for n in ast.walk(tree):
        if isinstance(n, ast.If):
            names = {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)}
            vals = {x.value for x in ast.walk(n.test)
                    if isinstance(x, ast.Constant)}
            if "__name__" in names and "__main__" in vals:
                spans.append((n.lineno, n.end_lineno))
        elif (not script and isinstance(n, ast.FunctionDef)
              and n.name in ("main", "demo")):
            spans.append((n.lineno, n.end_lineno))
    return spans


def in_span(lineno, spans):
    return any(a <= lineno <= b for a, b in spans)


def classify_ruslits(rel):
    """
    Русские литералы core/, разложенные по ПРИРОДЕ. Одна цифра «сколько
    кириллицы в ядре» была бы сочинённой мерой (урок 22): замер показал, что
    из 390 строк 93 — это print/log, то есть ничего не значат для границы,
    а трещотка на такую цифру орала бы на каждую добавленную строчку лога.

    ДАННЫЕ (data) — влияют на поведение: суффиксы, предлоги, ключи словарей,
    сравниваемые строки. ИМЕННО ОНИ нарушают границу: сравнение «если слово
    кончается на -ся» внутри core — это решение, принятое по-русски.

    ДЕМО (demo) — строки внутри `if __name__ == "__main__"` / def main(). В
    библиотечный путь они не попадают и нарушением НЕ являются. Их ОБЯЗАНО
    быть видно ОТДЕЛЬНЫМ числом: первая версия меры посчитала 117 строк
    самопроверок «русскими данными в ядре» и завысила нарушение примерно
    вдвое (урок 24 в новом месте: категория объявлена, поле не прочитано).
    """
    tree = parse(rel)
    docs = docstring_ids(tree)
    parent = {}
    for n in ast.walk(tree):
        for ch in ast.iter_child_nodes(n):
            parent[id(ch)] = n
    spans = demo_spans(tree)

    def in_demo(lineno):
        return in_span(lineno, spans)

    data, msg, demo = 0, 0, 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and RU.search(node.value) and id(node) not in docs):
            continue
        p = parent.get(id(node))
        # f-строка и её части — обёртки, смотрим на деда
        for _ in range(2):
            if isinstance(p, (ast.JoinedStr, ast.FormattedValue)):
                p = parent.get(id(p))
            else:
                break
        # строка, на которую ссылаются как на аргумент print(...) / raise ... —
        # сообщение; всё прочее (set/tuple/dict/сравнение/присваивание) — данные
        call = p if isinstance(p, ast.Call) else (
            parent.get(id(p)) if isinstance(p, ast.keyword) else None)
        name = None
        if isinstance(call, ast.Call):
            name = getattr(call.func, "id", None) or getattr(call.func, "attr",
                                                             None)
        is_msg = name in DIAGNOSTIC_FUNCS or isinstance(p, ast.Raise) or (
            isinstance(parent.get(id(p)), ast.Raise))
        if in_demo(node.lineno):
            demo += 1
        elif is_msg:
            msg += 1
        else:
            data += 1
    return data, msg, demo


# ── ЦЕНА ВЫХОДНОЙ ГРАНИЦЫ ───────────────────────────────────────────
# Сколько проверок СРАВНИВАЮТ русскую строку. Это не «нарушение» и не
# должок: это ЦЕНА шага A4 («core отдаёт IR вместо строки»). Без неё
# «осталось дописать lang/» — сочинённое обещание; с ней цена стоит в
# отчёте и в тесте.
#
# Считать обязан AST, а не grep. Grep по «assert.*кириллица» дал 628 и был
# неправ дважды: он не отличает СРАВНИВАЕМУЮ строку от текста ошибки
# (а таких, замерено, 424), и не знает, сколько у конкретного assert
# аргументов: у assertTrue(a, "сообщение") кириллица в втором аргументе —
# это сообщение, а у assertEqual(a, b) во втором — проверка.
CMP_ARGS = {
    "assertTrue": 1, "assertFalse": 1, "assertIs": 2, "assertIsNone": 1,
    "assertIsNotNone": 1, "assertIn": 2, "assertNotIn": 2, "assertEqual": 2,
    "assertNotEqual": 2, "assertGreater": 2, "assertGreaterEqual": 2,
    "assertLess": 2, "assertLessEqual": 2, "assertAlmostEqual": 2,
    "assertNotAlmostEqual": 2, "assertListEqual": 2, "assertSetEqual": 2,
    "assertTupleEqual": 2, "assertDictEqual": 2, "assertIsInstance": 2,
    "assertRegex": 2, "assertCountEqual": 2, "assertMultiLineEqual": 2,
    "assertSequenceEqual": 2, "fail": 0,
}


def ru_inside(node):
    for k in ast.walk(node):
        if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                and RU.search(k.value):
            return True
    return False


def price_of_ir_output():
    """(проверок с русским сравнением, файлов, проверок только-с-сообщением)."""
    checked = msg_only = 0
    files = set()
    for rel in py_files("tests"):
        try:
            tree = parse(rel)
        except SyntaxError:
            continue
        for x in ast.walk(tree):
            if not isinstance(x, ast.Call):
                continue
            name = getattr(x.func, "attr", None)
            if not name or not (name.startswith("assert") or name == "fail"):
                continue
            if name == "assertRaises":
                continue
            n = CMP_ARGS.get(name, len(x.args))
            compared = x.args[:n]
            kw_data = [k.value for k in x.keywords if k.arg not in ("msg", None)]
            kw_msg = [k.value for k in x.keywords if k.arg == "msg"]
            if any(ru_inside(a) for a in compared) \
                    or any(ru_inside(v) for v in kw_data):
                checked += 1
                files.add(rel)
            elif any(ru_inside(a) for a in x.args[n:]) \
                    or any(ru_inside(v) for v in kw_msg):
                msg_only += 1
    return checked, len(files), msg_only


def core_reaches_lang(rel):
    """
    ЯДРО, которое импортирует lang/, — это та же дыра, что импорт core.morph,
    только ставшая НЕВИДИМОЙ после шага A1. Пока язык лежал в core/morph.py,
    импорт ловила категория S. Стоит переехать ему в lang/ru/morph.py, как
    `from lang.ru.morph import Morph` перестал бы быть языковым импортом по
    имени, и закон осталось бы только на бумаге. Поэтому потолок C = 0 и он
    не меняется никогда: через lang/ имеют право ходить brain/ и ui/, ядро —
    только через IR.
    """
    hits = []
    tree = parse(rel)
    spans = demo_spans(tree)
    for mod, lineno, nested in imports_of(tree):
        full = module_root(mod, rel)
        if full.split(".")[0] == LANGUAGE_PACKAGES[0]:
            hits.append({"module": full, "line": lineno, "nested": nested,
                         "demo": in_span(lineno, spans)})
    return hits


def lang_consumers(rel, script=False):
    """
    G: кто НЕ-языковой код импортирует lang/.

    Зачем отдельная категория, а не «и так видно». A1 уже показывала этот класс
    слепоты: пока язык лежал в core/morph.py, импорт ловила категория S; стоило
    ему переехать в lang/, как шов стал невидим и закон остался бы только на
    бумаге (для ядра это закрыто категорией C = 0). Но для brain/, tools/ и
    teachers/ язык ИМЕЕТ право жить в lang/, значит их импорты туда не
    нарушают ничего и не попадают ни в S, ни в C — то есть сегодня ни один
    замер не показывает, СКОЛЬКО мест вообще способны принести язык в ядро.

    Это ровно множество «кто может внедрить службу», а forgotten installer —
    главный риск шагов A2.1–A2.3 (мутант «мозг не передал резолвер» и мутант
    «install() забыл вторую службу»). Число без этой меры гнило бы молча:
    новый производственный вход, не принёсший службу, выглядел бы как
    существующий.

    ПОТОЛКА НЕТ намеренно: импортировать lang/ разрешено. Число нужно, чтобы
    переезд шва из ядра наружу не превратился в исчезновение работы.
    """
    tree = parse(rel)
    spans = demo_spans(tree, script=script)
    hits = []
    for mod, lineno, nested in imports_of(tree):
        full = module_root(mod, rel)
        if full.split(".")[0] == LANGUAGE_PACKAGES[0]:
            hits.append({"module": full, "line": lineno, "nested": nested,
                         "demo": in_span(lineno, spans)})
    return hits


def measure():
    report = {
        "language_files": [],          # D: долговой язык внутри core
        "seams": {},                   # S: кто импортирует язык
        "lang_consumers": {},          # G: кто импортирует lang/ (без потолка)
        "output_methods": {},          # O: кто отдаёт русскую строку
        "core_ruslits": {},            # R: русские литералы в НЕязыковом core
        "language_ruslits": {},        # справочно: кириллица в файлах-языках
        "core_messages": {},           # print/log/raise — косметика, НЕ граница
        "core_demo": {},               # строки самопроверок __main__
        "core_reaches_lang": {},       # C: ядро тянет lang/ мимо IR
        "lang_files": [],              # справочно: где язык живёт сейчас
        "outer_seams": {},             # U: швы в инструменте (справочно)
    }
    for rel in py_files(*CORE_PACKAGES, *BRAIN_PACKAGES, *LANGUAGE_PACKAGES):
        if rel.startswith(LANG_DIR):
            report["lang_files"].append(rel)
            continue        # lang/ — язык по месту жительства: не шов, не долг
        if is_language_file(rel):
            report["language_files"].append(rel)
        hits = lang_imports(rel)
        # сам языковой модуль, знающий язык, — не нарушение
        if hits and not is_language_file(rel):
            report["seams"][rel] = hits
        # G: НЕ нарушая ничего, импорты lang/ обязаны быть видны — иначе
        # множество «кто способен принести язык в ядро» не знает никто
        g = lang_consumers(rel)
        if g:
            report["lang_consumers"][rel] = g
        om = output_methods(rel)
        if om:
            report["output_methods"][rel] = om
        if rel.startswith("core" + os.sep):
            reach = core_reaches_lang(rel)
            if reach:
                report["core_reaches_lang"][rel] = reach
            rl, msg, demo = classify_ruslits(rel)
            if not (rl or msg or demo):
                continue
            # ЯЗЫКОВОЙ файл, полный кириллицы, — НЕ нарушение и НЕ инерция
            # ядра: он переедет в lang/ru целиком. Мерить по нему «сколько
            # русского в нейтральном ядре» — тот же сочинённый знаменатель,
            # что «14 из 20 отношений» (урок 22): цифра есть, смысла нет.
            # Поэтому R считает ТОЛЬКО НЕязыковые файлы core/, а долг
            # языковых показывается отдельно, справочно.
            if is_language_file(rel):
                report["language_ruslits"][rel] = rl
            else:
                report["core_ruslits"][rel] = rl
                report["core_messages"][rel] = msg
                if demo:
                    report["core_demo"][rel] = demo
    for rel in py_files(*SCRIPT_PACKAGES):
        if rel.startswith(LANG_DIR) or is_language_file(rel):
            continue
        try:
            hits = lang_imports(rel, script=True)
        except SyntaxError:                 # pragma: no cover
            continue
        hits = [h for h in hits if not h.get("demo")]
        if hits:
            report["outer_seams"][rel] = hits
        try:
            g = lang_consumers(rel, script=True)
        except SyntaxError:                 # pragma: no cover
            g = []
        if g:
            report["lang_consumers"][rel] = g

    checked, nfiles, msg_only = price_of_ir_output()
    report["tests_touching_russian"] = {
        "checks_comparing_russian": checked, "files": nfiles,
        "checks_with_russian_message_only": msg_only}
    report["totals"] = {
        "D_debt_files": len(report["language_files"]),
        # C обязан остаться НУЛЁМ навсегда: см. core_reaches_lang
        "C_core_reaches_lang": sum(len(v)
                                   for v in report["core_reaches_lang"].values()),
        "L_lang_files": len(report["lang_files"]),
        # S считает ТОЛЬКО рабочие швы. Демо-швы — справочно (как M и X):
        # они не попадают в библиотечный путь, но обязаны быть видны числом,
        # чтобы «ноль швов» не оказался нулём, нарисованным демо-блоком.
        "S_seam_sites": sum(1 for v in report["seams"].values()
                            for h in v if not h.get("demo")),
        "S_demo_seam_sites": sum(1 for v in report["seams"].values()
                                 for h in v if h.get("demo")),
        # файлом считается только тот, где есть хотя бы один РАБОЧИЙ шов:
        # иначе «9 файлов» осталось бы, даже когда все швы уехали в демо
        "S_seam_files": sum(1 for v in report["seams"].values()
                            if any(not h.get("demo") for h in v)),
        # ШВЫ ВНУТРИ core — единственные из двенадцати, что нарушают закон
        # напрямую. Мозг имеет право говорить с человеком; ядро — нет. Без
        # отдельной меры эти четыре тонут в общем числе и рефакторинг можно
        # объявить готовым, не тронув их.
        "S_demo_seams_inside_core": sum(
            1 for f, v in report["seams"].items()
            if f.startswith("core" + os.sep)
            for h in v if h.get("demo")),
        "S_seams_inside_core": sum(
            1 for f, v in report["seams"].items()
            if f.startswith("core" + os.sep)
            for h in v if not h.get("demo")),
        "O_output_methods": sum(len(v)
                                for v in report["output_methods"].values()),
        "R_core_ruslits": sum(report["core_ruslits"].values()),
        "M_core_messages": sum(report["core_messages"].values()),
        "X_core_demo": sum(report["core_demo"].values()),
        "R_neutral_files": sum(1 for v in report["core_ruslits"].values()
                               if v),
        "Z_language_ruslits": sum(report["language_ruslits"].values()),
        "P_ir_output_price": checked,
        "P_ir_output_files": nfiles,
        # U — СПРАВОЧНАЯ величина, потолка у неё НЕТ и быть не может без
        # решения хозяина: инструмент языконезависимым обязанности не имеет.
        # Смысл категории один: шов, переехавший из ядра в tools/, обязан
        # остаться ВИДНЫМ числом, а не исчезнуть из меры.
        "U_outer_seam_sites": sum(len(v)
                                  for v in report["outer_seams"].values()),
        "U_outer_seam_files": len(report["outer_seams"]),
        # G — СПРАВОЧНАЯ величина, потолка НЕТ (см. lang_consumers): импорт
        # lang/ не-языковому коду разрешён. Но молча гнить он не обязан:
        # это список мест, где можно ЗАБЫТЬ принести службу в ядро.
        "G_lang_import_sites": sum(len(v)
                                   for v in report["lang_consumers"].values()),
        "G_lang_import_files": len(report["lang_consumers"]),
    }
    return report


def print_report(rep):
    t = rep["totals"]
    print("=" * 72)
    print("ГРАНИЦА ЯЗЫКА:  текст -> lang/xx -> IR -> core -> IR -> lang/xx -> текст")
    print("core обязан знать ТОЛЬКО IR; русский язык обязан жить в lang/ru")
    print("=" * 72)

    print("\n[1. ДОЛГ] языковые модули ВНУТРИ core — это будущий lang/ru:")
    for f in rep["language_files"]:
        print(f"   {f}")

    if rep["core_reaches_lang"]:
        print("\n[1б. ЯДРО ТЯНЕТ lang/ РУКАМИ — это тот же импорт языка, "
              "ставший невидимым после переезда. ПОТОЛОК = 0:")
        for f, hits in sorted(rep["core_reaches_lang"].items()):
            mods = ", ".join(h["module"] for h in hits)
            lines = ",".join(str(h["line"]) for h in hits)
            print(f"   {f:34} x{len(hits)}  {mods}  (стр {lines})")
    print(f"\n   язык живёт здесь ({t['L_lang_files']}): "
          f"{', '.join(rep['lang_files']) or 'пока нигде'}")

    print(f"\n[2. ШВЫ] РАБОЧИХ импортов языка не-языковым кодом: "
          f"{t['S_seam_sites']} в {t['S_seam_files']} файлах —"
          f" ровно столько мест обязан закрыть IR")
    print(f"   (справочно: ещё {t['S_demo_seam_sites']} швов стоят в "
          f"демо-самопроверках `__main__` — в библиотечный путь не попадают, "
          f"нарушением НЕ считаются, но число обязано быть видно: «ноль швов», "
          f"дорисованный демо-блоком, это не ноль)")
    for f, hits in sorted(rep["seams"].items(),
                          key=lambda kv: -len(kv[1])):
        mods = ", ".join(sorted({h["module"] for h in hits}))
        lines = ",".join(str(h["line"]) for h in hits)
        mark = "  <- НАРУШЕНИЕ ЗАКОНА: ядро знает язык" if f.startswith(
            "core" + os.sep) else ""
        print(f"   {f:34} x{len(hits)}  {mods}  (стр {lines}){mark}")

    print(f"\n[3. ВЫХОД] публичных методов, возвращающих РУССКУЮ СТРОКУ: "
          f"{t['O_output_methods']}")
    if not rep["output_methods"]:
        print("   нет — ядро отдаёт IR")
    for f, hits in rep["output_methods"].items():
        for h in hits:
            print(f"   {f:34} {h['cls']}.{h['method']}()  :{h['line']}")

    print(f"\n[4. ИНЕРЦИЯ] ДАННЫХ на русском в НЕЯЗЫКОВОМ core/: "
          f"{t['R_core_ruslits']} в {t['R_neutral_files']} файлах  "
          f"(это границу НАРУШАЕТ: решение по окончанию/предлогу внутри ядра)")
    print(f"   самопроверок (__main__) на русском: {t['X_core_demo']} — в "
          f"библиотечный путь не попадают, трещоткой НЕ ЗАПРЕЩЕНЫ")
    print(f"   сообщений print/log/raise на русском: {t['M_core_messages']} — "
          f"на поведение не влияет, это косметика, трещоткой НЕ ЗАПРЕЩЕНА")
    print("   (в core/morph.py, core/extract.py, core/lemma.py кириллицы "
          f"{t['Z_language_ruslits']} — они переедут в lang/ru целиком, это НЕ "
          f"нарушение)")
    for f, n in sorted(rep["core_ruslits"].items(), key=lambda kv: -kv[1]):
        if n == 0:
            continue          # пустых строк не показываем: они размывают список
        print(f"   {f:34} {n}")

    tt = rep["tests_touching_russian"]
    print(f"\n[5. ЦЕНА A4] проверок, СРАВНИВАЮЩИХ русскую строку: "
          f"{tt['checks_comparing_russian']} в {tt['files']} файлах tests/ "
          f"(из 40)")
    print(f"   ещё {tt['checks_with_russian_message_only']} содержат русский "
          f"только в тексте ошибки — их правка бесплатна")
    print("   Это НЕ должок и НЕ нарушение: это ЦЕНА шага «core отдаёт IR "
          "вместо строки». Без этой цифры обещание «осталось дописать lang/» "
          "— сочинённое.")

    print(f"\n[6. ВНЕШНИЕ ШВЫ] язык импортируют инструменты: "
          f"{t['U_outer_seam_sites']} в {t['U_outer_seam_files']} файлах "
          f"(tools/, teachers/)")
    for f, hits in sorted(rep["outer_seams"].items()):
        print(f"   {f:34} x{len(hits)}  "
              f"{','.join(h['module'] for h in hits)}")
    print("   Это НЕ нарушение и ПОТОЛКОМ НЕ ЗАКРЫТО: учитель и инструмент")
    print("   имеют право знать язык. Число нужно, чтобы шов, ВЫВЕЗЕННЫЙ из")
    print("   ядра в tools/, остался виден, а не исчез из меры — ровно так")
    print("   же A1 сделал невидимым импорт, переехав в lang/.")

    print(f"\n[7. НОСИТЕЛИ ЯЗЫКА] не-языковой код импортирует lang/: "
          f"{t['G_lang_import_sites']} в {t['G_lang_import_files']} файлах")
    for f, hits in sorted(rep["lang_consumers"].items()):
        print(f"   {f:34} x{len(hits)}  "
              f"{','.join(h['module'] for h in hits)}")
    print("   Разрешено и ПОТОЛКОМ НЕ ЗАКРЫТО. Это список мест, где обязан")
    print("   стоять внедритель языковой службы: забыть его — значит оставить")
    print("   ядро деградировать молча (шаги A2.1–A2.3 падали именно на этом).")

    print("\n" + "=" * 72)
    print("ИТОГО  ", "  ".join(f"{k}={v}" for k, v in sorted(t.items())))
    print("ТРЕЩОТКА: любое из этих чисел МОЖЕТ упасть без вопросов; РАСТИ не")
    print("может — рост это падение tests/test_langborder_contract.py.")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="куда положить снимок")
    args = ap.parse_args()
    rep = measure()
    print_report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
        print(f"снимок: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
