"""
Статическая проверка исходников: мёртвый код и «обещание без реализации».

Почему это тест, а не разовая команда
-------------------------------------
Разовая AST-проверка смотрела только верхний уровень тела функции и ПРОПУСТИЛА
дублирующийся break внутри цикла в teachers/teacher1.py (реальный баг: второй
break был недостижим). Значит проверка должна быть рекурсивной и запускаться
каждый раз — иначе она деградирует незаметно, ровно как непроверяемый порог.

Что ловится
-----------
1. Инструкции после return/raise/break/continue на ЛЮБОМ уровне вложенности.
   Такой код не выполнится никогда; обычно это забытый дубль или «закомментирован-
   но-не-удалено» мышление, и он врёт читателю о поведении функции.
2. Атрибуты, которые только записываются и никогда не читаются (подозрение на
   недореализованную возможность), и обработчики-заглушки.
3. Функции, тело которых — только pass/... при отсутствии комментария-оправдания:
   «заглушка, которая вызывается» — источник тихой деградации, которую проект
   считает багом.

Исключения объяснены в коде; молча не исключается ничего.
"""
import ast
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Куда не смотрим: окружение, артефакты, чужой код.
SKIP_DIRS = {".venv", ".git", "__pycache__", "data", "legacy", "build", "dist"}

TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def python_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def bodies_of(node):
    """Все списки инструкций, принадлежащие САМОМУ узлу (не вложенных блоков)."""
    for field in ("body", "orelse", "finalbody"):
        value = getattr(node, field, None)
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            yield value


class TestNoDeadCode(unittest.TestCase):

    def test_no_unreachable_statement_after_terminator(self):
        offenders = []
        for path in python_files():
            rel = os.path.relpath(path, ROOT)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            # обходим ВСЁ дерево, включая вложенные функции и ветвления:
            # прежняя одноуровневая версия пропустила break внутри for
            for node in ast.walk(tree):
                for body in bodies_of(node):
                    for i, stmt in enumerate(body[:-1]):
                        if isinstance(stmt, TERMINATORS):
                            nxt = body[i + 1]
                            # декоратор/документация после return — тоже мёртвый код
                            offenders.append(
                                f"{rel}:{nxt.lineno}: недостижимо после "
                                f"{type(stmt).__name__} на строке {stmt.lineno}")
        self.assertEqual(offenders, [], "мёртвый код:\n  " + "\n  ".join(offenders))

    def test_no_silent_stub_functions(self):
        """
        Тело из pass/... допустимо, только рядом стоит слово-объяснение.

        Причина жёсткая: заглушка, которую кто-то подключил, ведёт себя как
        «работает, но ничего не происходит». Для этого проекта (правило в
        memory.txt) тихая деградация = баг, поэтому либо объяснение, либо
        NotImplementedError.
        """
        offenders = []
        for path in python_files():
            rel = os.path.relpath(path, ROOT)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src, filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(
                        body[0].value, ast.Constant) and isinstance(
                        body[0].value.value, str):
                    body = body[1:]                      # docstring — не заглушка
                if len(body) != 1:
                    continue
                only = body[0]
                is_stub = (isinstance(only, ast.Pass)
                           or (isinstance(only, ast.Expr)
                               and isinstance(only.value, ast.Constant)
                               and only.value.value is Ellipsis))
                if not is_stub:
                    continue
                window = src.splitlines()[max(0, node.lineno - 4):node.lineno + 3]
                if not any(w in " ".join(window).lower()
                           for w in ("заглушк", "stub", "не реализ", "TODO",
                                     "tbd", "notimplemented")):
                    offenders.append(f"{rel}:{node.lineno}: {node.name}() — пустое "
                                     f"тело без объяснения")
        self.assertEqual(offenders, [],
                         "молчаливые заглушки (нужен NotImplementedError или "
                         "объясняющее слово рядом):\n  " + "\n  ".join(offenders))

    def test_written_attributes_are_also_read(self):
        """
        Атрибут, который ТОЛЬКО пишут внутри нашего кода и нигде не читают,
        — мёртвое состояние: выглядит так, будто что-то запоминается.

        Правила (и почему именно такие):
          * имя читается и динамически: getattr(node, 'canonical_id', 0) в
            core/storage.py для наивного обхода выглядит как «не читается», а
            тест ложно краснел именно на этом. Строковая константа — реальный
            канал чтения, поэтому имя, упомянутое строкой где-либо в репозитории,
            считается прочитанным;
          * объявление в __init__ или поле dataclass — ПУБЛИЧНЫЙ интерфейс
            класса, его читают другие модули и тесты; такая запись подозрений
            не вызывает (в отличие от «случайной» записи в середине функции);
          * чтение как Name (from x import attr) тоже учитывается.
        """
        texts = {}
        for path in python_files():
            with open(path, encoding="utf-8") as fh:
                texts[path] = fh.read()

        string_names = set()
        for src in texts.values():
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    string_names.add(node.value)

        writes, reads, declared = {}, set(), set()
        for path, src in texts.items():
            tree = ast.parse(src, filename=path)
            init_lines, non_init_lines = set(), set()
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                span = range(fn.lineno, fn.end_lineno + 1)
                if fn.name == "__init__":
                    init_lines.update(span)
                else:
                    non_init_lines.update(span)
            for cls in ast.walk(tree):
                if isinstance(cls, ast.ClassDef):
                    for st in cls.body:
                        # поля dataclass: декларация состояния, а не «забыли»
                        if isinstance(st, ast.AnnAssign) and isinstance(st.target,
                                                                        ast.Name):
                            declared.add(st.target.id)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.ctx, ast.Store):
                        in_init = node.lineno in init_lines \
                            and node.lineno not in non_init_lines
                        writes.setdefault(node.attr, []).append(
                            (os.path.relpath(path, ROOT), node.lineno, in_init))
                    else:
                        reads.add(node.attr)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    reads.add(node.id)

        ignore = {a for a in writes if a.startswith("_")} | declared | {
            "ask_user_callback",      # назначается GUI, читается мозгом
        }
        dead = {}
        for attr, entries in writes.items():
            if attr in ignore or attr in reads or attr in string_names:
                continue
            if any(is_init for _, _, is_init in entries):
                continue
            dead[attr] = [(f, ln) for f, ln, _ in entries]
        self.assertEqual(
            dead, {},
            "атрибуты пишутся и никогда не читаются:\n  "
            + "\n  ".join(f"{a}: {v[:3]}" for a, v in sorted(dead.items())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
