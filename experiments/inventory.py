#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ИНВЕНТАРИЗАЦИЯ: что в движке работает, а что лежит на полке.

ЗАЧЕМ. В проекте дважды находили одно и то же: функция написана, задокументирована,
местами протестирована — и НИ РАЗУ не вызвана движком. Классика: гормоны долго
жили как украшение GUI, пока не стали управляющим сигналом. Проверка глазами
это не ловит: имя встречается в докстрингах, и глаз видит «упомянуто».

Мера одна и та же для всего: ССЫЛКА В КОДЕ, а не упоминание в тексте. Считаем
после разбора дерева, поэтому докстринги, комментарии и строки сообщений
счёт не увеличивают — ровно та же логика, что у tools/langborder.py.

ТРИ КОРЗИНЫ:
  * ДВИЖОК — есть вызов из рабочего пакета (brain/core/teachers/lang);
  * ПОЛКА — определено и вызывается ТОЛЬКО тестами или не вызывается вовсе;
  * ШУМ — инструмент, точка входа, тест: их незачем вызывать из движка.

ЗАМЕР НИЧЕГО НЕ ПЕРЕМЕНЯЕТ. Он печатает список кандидатов, чтобы решение
«выкинуть / задействовать» принималось по числу, а не по ощущению.

Запуск: .venv/bin/python experiments/inventory.py [--json]
"""
import argparse
import ast
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пакеты движка: то, что исполняется внутри программы.
ЯДРО = ("brain", "core", "teachers", "lang")
# Слой пользователя: GUI и точка входа. Вызов оттуда — НАСТОЯЩИЙ вызов
# программы, а не тестовая подпорка. Замер до этой строки записывал в полку
# process_input, который зовёт окно: инвентарь врал ровно тем же способом,
# каким проект воюет с докстрингами — считал контекст там, где надо смотреть
# на адрес.
ИНТЕРФЕЙС = ("ui",)
# Всё остальное в .py: инструменты, эксперименты, тесты, запуск.
ОБОС = ("tools", "experiments", "tests")
ИСКЛЮЧИТЬ = {"__pycache__", ".git", "data", "build", "dist", ".venv",
            "node_modules", ".venv312"}


def файлы_пакетов(пакеты):
    for pkg in пакеты:
        base = os.path.join(ROOT, pkg)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in ИСКЛЮЧИТЬ]
            for f in sorted(fns):
                if f.endswith(".py"):
                    yield os.path.join(dp, f)


def файлы_интерфейса():
    """ui/ плюс точка входа run.py одним списком."""
    base = os.path.join(ROOT, "ui")
    if os.path.isdir(base):
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d not in ИСКЛЮЧИТЬ]
            for f in sorted(fns):
                if f.endswith(".py"):
                    yield os.path.join(dp, f)
    root_run = os.path.join(ROOT, "run.py")
    if os.path.exists(root_run):
        yield root_run


def все_py():
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in ИСКЛЮЧИТЬ]
        for f in sorted(fns):
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def rel(path):
    return os.path.relpath(path, ROOT)


class Сборщик:
    """Собирает ОБЪЯВЛЕНИЯ и ССЫЛКИ одного файла по одному обходу дерева.

    Объявление = def/класс верхнего уровня, метод, КОНСТАНТА_ВЕРХНЕГО_УРОВНЯ.
    Ссылка = Name.id, Attribute.attr, строка внутри getattr/hasattr.

    Обход итеративный (ast.walk), а не рекурсивный NodeVisitor: в проекте
    есть глубоко вложенные литералы, и на рекурсии сборщик падал
    RecursionError — то есть врал бы про целый файл.

    Строки и комментариев в дереве нет, поэтому упоминание в докстринге
    ссылкой НЕ считается. В этом весь смысл меры: глаз по упоминанию
    отличить «вызывается» от «описано» не может.
    """

    def __init__(self, path, в_ядру):
        self.path = path
        self.в_ядру = в_ядру
        self.объявления = []          # (имя, вид, строка)
        self.ссылки = collections.Counter()

    def _собираем_алиасы(self, tree):
        """import x as y / from m import f as g: связываем локальное имя с далёким."""
        for узел in ast.walk(tree):
            if isinstance(узел, ast.Import):
                for a in узел.names:
                    if a.asname:
                        # import brain.sleep as bs: bs.метод() -> Attribute,
                        # его ловит другая ветка; сам модуль помечем ссылкой
                        self._алиасы[a.asname] = a.name.split(".")[-1]
            elif isinstance(узел, ast.ImportFrom):
                for a in узел.names:
                    if a.asname:
                        self._алиасы[a.asname] = a.name

    def собрать(self, tree):
        self._алиасы = {}          # локальное имя -> имя в исходном модуле
        self._собираем_алиасы(tree)
        # 1) объявления верхнего уровня (функции, классы, КОНСТАНТЫ)
        for узел in tree.body:
            if isinstance(узел, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.объявления.append((узел.name, "функция", узел.lineno))
            elif isinstance(узел, ast.ClassDef):
                self.объявления.append((узел.name, "класс", узел.lineno))
                for m in узел.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.объявления.append((m.name, "метод", m.lineno))
                    elif isinstance(m, ast.Assign):
                        for t in m.targets:
                            if isinstance(t, ast.Name) and t.id.isupper():
                                self.объявления.append((t.id, "константа", m.lineno))
            elif isinstance(узел, ast.Assign):
                for t in узел.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        self.объявления.append((t.id, "константа", узел.lineno))
        # 2) все ссылки одним обходом
        for узел in ast.walk(tree):
            if isinstance(узел, ast.Name):
                имя = узел.id
                self.ссылки[имя] += 1
                # ПСЕВДОНИМ. `from lang.ru.lemma import install as _i` и
                # вызов `_i()` — это вызов install, но под другим именем.
                # Без этого шага инвентарь записал install в полку, хотя
                # движок вызывает его в четырёх местах. Мера, которая так
                # врёт, опаснее отсутствия меры: она закрывает живое.
                далёкое = self._алиасы.get(имя)
                if далёкое:
                    self.ссылки[далёкое] += 1
            elif isinstance(узел, ast.Attribute):
                self.ссылки[узел.attr] += 1
            elif isinstance(узел, ast.Constant) and isinstance(узел.value, str):
                # getattr(obj, "имя"): имя живёт в строке, а движок по нему
                # работает. Без этого шага такая функция выглядела бы полкой.
                if len(узел.value) < 64 and узел.value.isidentifier():
                    self.ссылки[узел.value] += 1


# Ключи данных, чьи ЗНАЧЕНИЯ — имена кода. Метод можно позвать не только
# из .py: brain/question.py зовёт getattr(sim, scheme.handler), а handler
# приходит из JSON рассказа (experiments/cat_canary_world.json). Мера,
# которая смотрит только в .py, записывает такой метод в мёртвые.
КЛЮЧИ_АДРЕСОВ = {"handler", "class", "cls", "module", "method", "writer",
                 "reader", "factory", "hook", "callback"}


def адреса_из_данных():
    """Имена кода, на которые указывают значения JSON-файлов проекта."""
    имена = collections.Counter()
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in ИСКЛЮЧИТЬ]
        for f in sorted(fns):
            if not f.endswith(".json"):
                continue
            path = os.path.join(dp, f)
            try:
                данные = json.load(open(path, encoding="utf-8"))
            except (OSError, ValueError):
                continue

            def обойти(узел):
                if isinstance(узел, dict):
                    for к, зн in узел.items():
                        if к in КЛЮЧИ_АДРЕСОВ and isinstance(зн, str) and \
                                зн.isidentifier():
                            имена[зн] += 1
                        else:
                            обойти(зн)
                elif isinstance(узел, list):
                    for эл in узел:
                        обойти(эл)
            обойти(данные)
    return имена


def динамические_имена():
    """Имена, которые в ядре живут СТРОКАМИ: getattr(obj, name), name="x".

    Третья дыра меры (после .venv и псевдонимов импорта): шлюз гормонов
    читает шкалы как getattr(self._h, name), и я по AST-подсчёту
    идентификаторов объявила их «пишут 15, читают 3» — ложь, которую
    заметил только прогоном живого процесса. Если имя встречается в ядре
    как строковый литерал, мера обязана САМА поднять флаг, а не ждать,
    что я вспомню про динамический вызов.
    """
    строки = collections.Counter()
    for path in файлы_пакетов(ЯДРО):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for узел in ast.walk(tree):
            if isinstance(узел, ast.Call) and \
                    isinstance(узел.func, ast.Name) and \
                    узел.func.id in ("getattr", "setattr", "hasattr") and \
                    len(узел.args) >= 2:
                арг = узел.args[1]
                if isinstance(арг, ast.Constant) and isinstance(арг.value, str) \
                        and арг.value.isidentifier():
                    строки[арг.value] += 1
    return строки


def собрать():
    определения = {}     # имя -> {"вид":, "файлы": set, "строка":, "в_ядру": bool}
    ссылки_ядро = collections.Counter()
    ссылки_обос = collections.Counter()
    ссылки_интерфейс = collections.Counter()
    ссылки_вн = collections.Counter()      # ссылки внутри собственного файла

    for path in все_py():
        r = rel(path)
        pkg = r.split(os.sep)[0]
        в_ядру = pkg in ЯДРО
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        sb = Сборщик(path, в_ядру)
        sb.собрать(tree)
        имена_файла = {и for и, _в, _с in sb.объявления}
        for имя, вид, стр in sb.объявления:
            запись = определения.setdefault(
                имя, {"вид": вид, "файлы": set(), "строка": стр,
                      "в_ядру": False})
            запись["файлы"].add(r)
            запись["в_ядру"] = запись["в_ядру"] or в_ядру
        for имя, n in sb.ссылки.items():
            if в_ядру:
                ссылки_ядро[имя] += n
            elif r == "run.py" or pkg in ИНТЕРФЕЙС:
                ссылки_интерфейс[имя] += n
            elif pkg in ОБОС:
                ссылки_обос[имя] += n
            else:
                ссылки_вн[имя] += n

        # ссылки внутри того же файла, где объявлено, храним отдельно:
        # self.метод() внутри класса не делает метод живым для движка,
        # если сам класс никем не вызывается, но молчаливые внутренники
        # надо видеть.
        for имя in имена_файла:
            внутри = sb.ссылки.get(имя, 0)
            if внутри:
                ссылки_вн[f"внутри:{r}:{имя}"] = внутри
    return определения, ссылки_ядро, ссылки_интерфейс, ссылки_обос, ссылки_вн


def файл_ядра(файлы):
    """Показать файл ЯДРА, а не первый алфавитный.

    Без этого имя, объявленное и в core/, и в .venv (orjson, torch),
    печаталось как живущее в чужом пакете, и инвентарь врал про то, что
    именно лежит на полке.
    """
    for f in sorted(файлы):
        if f.split(os.sep)[0] in ЯДРО:
            return f
    return sorted(файлы)[0]


def классифицировать(определения, ссылки_ядро, ссылки_интерфейс,
                    ссылки_обос, ссылки_вн, адреса=None, дин=None,
                    слепые=None):
    """Корзины. Служебные имена (__init__, main, test_*) не считаем полкой."""
    результат = {"движок": [], "полка_только_тесты": [], "полка_мертво": [],
                 "полка_только_своим_файлом": [], "полка_под_вопросом": []}
    for имя, запись in sorted(определения.items()):
        if имя.startswith("__") and имя.endswith("__"):
            continue
        if имя.startswith("test_") or имя.startswith("Test"):
            continue
        if имя in ("main", "run", "cli"):
            continue
        r = файл_ядра(запись["файлы"])
        if not запись["в_ядру"]:
            continue                      # инструменты и тесты судить нечем
        ядро = ссылки_ядро.get(имя, 0)
        обос = ссылки_обос.get(имя, 0)
        интер = ссылки_интерфейс.get(имя, 0)
        # Вызов из GUI/точек входа — это работа программы, а не подпорка.
        адреса = адреса or collections.Counter()
        дин = дин or collections.Counter()
        файлы_слепых = {з[0] for з in (слепые or [])}
        if ядро > 0 or интер > 0:
            результат["движок"].append((имя, запись["вид"], r, ядро, обос))
        elif r in файлы_слепых:
            # Файл передаёт имена коду переменной: ни «мертво», ни «только
            # тесты» тут доказать нельзя — тест тоже мог позвать динамически.
            результат["полка_под_вопросом"].append(
                (имя, запись["вид"], r, адреса.get(имя, 0),
                 дин.get(имя, 0) or "слепая зона в файле"))
        elif адреса.get(имя) or дин.get(имя):
            # МЁРТВЫМ НАЗЫВАТЬ НЕЛЬЗЯ. Имя подают коду строкой: адресом из
            # JSON-схемы или аргументом getattr(..., name)
            # не отличает живой вызов от случайного слова, и честный ответ
            # здесь — «не доказано», а не «мертво».
            результат["полка_под_вопросом"].append(
                (имя, запись["вид"], r, адреса.get(имя, 0), дин.get(имя, 0)))
        elif обос > 0:
            результат["полка_только_тесты"].append(
                (имя, запись["вид"], r, ядро, обос))
        else:
            внутри = any(k.endswith(f":{имя}") and k.startswith("внутри:")
                         for k in ссылки_вн)
            (результат["полка_только_своим_файлом"] if внутри
             else результат["полка_мертво"]).append(
                (имя, запись["вид"], r, ядро, обос))
    return результат


def дубликаты(определения):
    """Одно имя объявлено в нескольких файлах ЯДРА: кандидаты на две системы.

    Соседство с torch/orjson в списке чужих пакетов — не дубликат закона,
    поэтому считаем только файлы ЯДРА.
    """
    групп = collections.defaultdict(list)
    for имя, запись in определения.items():
        свои = [f for f in запись["файлы"] if f.split(os.sep)[0] in ЯДРО]
        if len(свои) > 1:
            групп[имя] = sorted(свои)
    return dict(групп)


def заглушки():
    """Тела, которые ничего не делают: только строка, только pass, только ...

    Это НЕ замена tests/test_static_hygiene (он ловит молчаливые заглушки
    отдельным замком) — это вторая половина вопроса: функция может быть
    НЕ заглушкой и при этом никем не вызываться. Здесь ищется только то,
    что заведомо не может работать, чтобы не спорить о таких кандидатах
    потом.
    """
    найдено = []
    for path in файлы_пакетов(ЯДРО):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            тело = list(node.body)
            # докстринг не считается действием
            if тело and isinstance(тело[0], ast.Expr) and \
                    isinstance(тело[0].value, ast.Constant) and \
                    isinstance(тело[0].value.value, str):
                тело = тело[1:]
            if not тело:
                continue                      # чистый докстринг = абстракция, не обман
            пусто = True
            for stmt in тело:
                if isinstance(stmt, ast.Pass):
                    continue
                if isinstance(stmt, ast.Expr) and \
                        isinstance(stmt.value, ast.Constant) and \
                        stmt.value.value is Ellipsis:
                    continue
                пусто = False
                break
            if пусто:
                найдено.append((node.name, rel(path), node.lineno))
    return найдено


def слепые_зоны():
    """Места, где имя кода передают ПЕРЕМЕННОЙ: статика тут не знает НИЧЕГО.

    Это признание слепоты меры, а не её сила. В `getattr(self._h, name)` имя
    живёт в переменной, и разбор без исполнения не скажет, читаются ли
    `cortisol`/`dopamine`. По этой дыре я и объявила гормональную систему
    мёртвой: счётчик по идентификаторам видел 3 чтения, живой процесс —
    десятки.

    Правило отсюда: имя из полки, объявленное в файле СЛЕПОЙ ЗОНЫ, мёртвым
    не считается. Оно «не доказано», и решать по нему можно только прогоном.
    """
    зоны = []
    for path in файлы_пакетов(ЯДРО):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for узел in ast.walk(tree):
            if not (isinstance(узел, ast.Call) and
                    isinstance(узел.func, ast.Name) and
                    узел.func.id in ("getattr", "setattr", "hasattr") and
                    len(узел.args) >= 2):
                continue
            арг = узел.args[1]
            if isinstance(арг, ast.Constant):
                continue                    # литерал: его ловит динамические_имена
            имена = sorted({n.id for n in ast.walk(узел)
                            if isinstance(n, ast.Name)})
            зоны.append((rel(path), узел.lineno, узел.func.id,
                         арг.__class__.__name__, имена))
    return зоны


def сироты():
    """Модули ядра, до которых движок и интерфейс НЕ дотягиваются импортом.

    Второй срез того же вопроса, что и «полка»: функция может быть жива, но
    целый модуль не подключён. Учитываем все четыре формы импорта
    (`import a.b`, `import a.b as c`, `from a import b`, `from a.b import c`),
    иначе `from brain import sleep` выглядел бы как отсутствие импорта.
    """
    модули = set()
    for path in файлы_пакетов(ЯДРО):
        r = rel(path)
        if os.path.basename(r) != "__init__.py":
            модули.add(r)

    # какие модули ядра импортируются откуда
    откуда = {m: {"ядро": set(), "интерфейс": set(), "обос": set()}
              for m in модули}
    for path in все_py():
        r = rel(path)
        pkg = r.split(os.sep)[0]
        слой = ("ядро" if pkg in ЯДРО else
                "интерфейс" if (pkg in ИНТЕРФЕЙС or r == "run.py") else "обос")
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        targets = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    targets.add(a.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    targets.add(node.module)
                    for a in node.names:
                        targets.add(f"{node.module}.{a.name}")
        for m in модули:
            полный = m[:-3].replace(os.sep, ".")
            пакет, _, верх = полный.partition(".")
            for t in targets:
                if t == полный or t.startswith(полный + ".") or \
                        (t == пакет and m.count(os.sep) == 0):
                    if r != m:
                        откуда[m][слой].add(r)
                    break
    сироты = []
    for m in sorted(модули):
        if not откуда[m]["ядро"] and not откуда[m]["интерфейс"]:
            сироты.append((m, sorted(откуда[m]["обос"])))
    return сироты


def флаги():
    """Переключатели, которые пишут, но не читают (или читают один раз)."""
    счётчик = collections.Counter()
    for path in файлы_пакетов(ЯДРО):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and \
                    isinstance(node.ctx, ast.Store):
                имя = node.attr
                if имя.startswith("use_") or имя.endswith("_enabled"):
                    счётчик[имя] += 1
    читается = collections.Counter()
    for path in все_py():
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and \
                    isinstance(node.ctx, ast.Load) and \
                    (node.attr.startswith("use_") or
                     node.attr.endswith("_enabled")):
                читается[node.attr] += 1
    return [(и, n, читается.get(и, 0)) for и, n in sorted(счётчик.items())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    арг = ap.parse_args()

    опр, ядро, интер, обос, вн = собрать()
    адреса = адреса_из_данных()
    дин = динамические_имена()
    слеп = слепые_зоны()
    корзины = классифицировать(опр, ядро, интер, обос, вн, адреса, дин, слеп)
    дуб = дубликаты(опр)
    заг = заглушки()
    фл = флаги()
    сир = сироты()

    if арг.json:
        print(json.dumps({
            "полка_мертво": [list(x) for x in корзины["полка_мертво"]],
            "полка_только_тесты": [list(x) for x in корзины["полка_только_тесты"]],
            "полка_только_своим_файлом": [
                list(x) for x in корзины["полка_только_своим_файлом"]],
            "дубликаты": {k: v for k, v in дуб.items()},
            "заглушки": [list(x) for x in заг],
            "флаги": [list(x) for x in фл],
            "сироты": [[m, o] for m, o in сир],
            "полка_под_вопросом": [list(x)
                                   for x in корзины["полка_под_вопросом"]],
            "слепые_зоны": [[z[0], z[1], z[2]] for z in слеп],
        }, ensure_ascii=False, indent=1))
        raise SystemExit(0)

    print("=" * 74)
    print("ИНВЕНТАРИЗАЦИЯ: определения движка против ссылок на них (только AST)")
    print("=" * 74)
    print(f"определений в ядре учтено: {sum(1 for v in опр.values() if v['в_ядру'])}")
    for корзина in ("движок", "полка_только_своим_файлом", "полка_только_тесты",
                    "полка_под_вопросом", "полка_мертво"):
        список = корзины[корзина]
        print(f"\n--- {корзина}: {len(список)} ---")
        if корзина == "движок":
            print("    (живо, показываем только те, куда ядро ходит ОДИН раз:")
            одиночки = [x for x in список if x[3] == 1]
            for имя, вид, r, я, о in sorted(одиночки)[:25]:
                print(f"      {имя:38} {вид:9} {r}:{я}")
            print(f"    всего одиночных входов: {len(одиночки)}")
            continue
        for имя, вид, r, я, о in список:
            if корзина == "полка_под_вопросом":
                print(f"      {имя:38} {вид:9} {r}  (адресом из данных:{я} "
                      f"строкой в ядре:{о})")
                continue
            print(f"      {имя:38} {вид:9} {r}  (ядро:{я} обос:{о})")

    print(f"\n--- имена, объявленные в НЕСКОЛЬКИХ файлах ядра: {len(дуб)} ---")
    for имя, fs in sorted(дуб.items()):
        print(f"      {имя:34} {', '.join(sorted(fs))}")

    print(f"\n--- заглушки в ядре: {len(заг)} ---")
    for имя, r, стр, вид in заг:
        print(f"      {имя:34} {r}:{стр}  ({вид})")

    print(f"\n--- МОДУЛИ ЯДРА БЕЗ ИМПОРТА ИЗ ДВИЖКА И ИНТЕРФЕЙСА: {len(сир)} ---")
    for m, ob in сир:
        print(f"      {m:32} описан в {len(ob)} файлах(е) тестов/инструментов")
        for o in ob[:3]:
            print(f"           {o}")

    print(f"\n--- СЛЕПЫЕ ЗОНЫ МЕРЫ (имя передают переменной): {len(слеп)} ---")
    print("    имена из этих файлов в корзину «мертво» не попадают:")
    for r, стр, фн, вид_арга, имена in слеп:
        print(f"      {r}:{стр}  {фн}(.., {вид_арга})  имён в строке: {len(имена)}")

    print(f"\n--- переключатели: написано, читается ---")
    for имя, n, ч in фл:
        метка = "<<< НЕ ЧИТАЕТСЯ" if ч == 0 else ""
        print(f"      {имя:28} пишут {n}  читают {ч}  {метка}")
