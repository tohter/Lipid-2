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
import re
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


# --------------------------------------------------------------------------
# Чужие письма в русском тексте
# --------------------------------------------------------------------------
# За работу эпизод повторялся трижды (experiments/ARCHIVE_SESSIONS.md): в
# русский файл попадали иероглифы и кана. Разовая команда grep не спасла,
# потому что она не входит в прогон тестов. Проверка живёт здесь.

# Дальневосточные письма и полноширинные формы. Кириллица, обычная латиница,
# CJK-пунктуация в диапазоны НЕ входят.
FOREIGN_RANGES = (
    (0x3040, 0x30FF),   # хирагана, катакана
    (0x3400, 0x4DBF),   # иероглифы, расширение A
    (0x4E00, 0x9FFF),   # иероглифы CJK
    (0xAC00, 0xD7AF),   # корейские слоги
    (0xF900, 0xFAFF),   # иероглифы совместимости
    (0xFF01, 0xFF60),   # полноширинные знаки и буквы
    (0xFFE0, 0xFFE6),   # полноширинные символы валют
)

# Сцепление кириллицы и латиницы БЕЗ пробела внутри одного слова. Дефект того
# же класса, что и иероглиф: в русское слово врезался латинский фрагмент
# живые примеры таких врезок ловит test_scanner_is_not_blind. Дефис НЕ
# считается: «UI-поток» и
# «top-вариант» — нормальные термины, а не порча.
GLUED = re.compile(r"[а-яёА-ЯЁ][A-Za-z]{2,}|[A-Za-z]{2,}[а-яёА-ЯЁ]")

# Чем оправдан исключений. КЛЮЧ — файл, ЗНАЧЕНИЕ — список тех и только тех
# чужих фрагментов, которые там имеют право жить. Сам файл с исключениями
# записан escape-последовательностями, иначе проверка краснела бы на собственном
# тексте (в этом и смысл исключений: они цитируют дефект, а не допускают его).
FOREIGN_ALLOWED = {
    "experiments/ARCHIVE_SESSIONS.md": ["\u7c7b\u76ee",
                                        "\u30cd\u30c3\u30c8\u30ef\u30fc\u30af"],
}


def text_files():
    """Все текстовые файлы проекта (не только .py): порча ловится в .md и .txt."""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if name.endswith((".py", ".md", ".txt", ".json", ".toml", ".cfg")):
                yield os.path.join(dirpath, name)


def read(path) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (UnicodeDecodeError, OSError):
        return ""


class TestNoForeignScript(unittest.TestCase):
    """Чужие письма и врезанная латиница в русском тексте."""

    def scan(self):
        """[(файл, строка, тип, фрагмент)] — всё подозрительное, без исключений."""
        found = []
        for path in text_files():
            rel = os.path.relpath(path, ROOT)
            for num, line in enumerate(read(path).splitlines(), 1):
                for ch in line:
                    if any(a <= ord(ch) <= b for a, b in FOREIGN_RANGES):
                        found.append((rel, num, "письмо", line.strip()[:60]))
                        break
                for m in GLUED.finditer(line):
                    found.append((rel, num, "сцепка", m.group()))
        return found

    def allowed_for(self, rel):
        """Символы чужих писем и строки-сцепки, которым тут разрешено жить."""
        chars, glued = set(), set()
        for tok in FOREIGN_ALLOWED.get(rel, []):
            if GLUED.fullmatch(tok):
                glued.add(tok)
            else:
                chars.update(ord(c) for c in tok
                             if any(a <= ord(c) <= b
                                    for a, b in FOREIGN_RANGES))
        return chars, glued

    def test_no_foreign_script(self):
        offenders = []
        for rel, num, kind, frag in self.scan():
            chars, glued = self.allowed_for(rel)
            if kind == "письмо":
                if all(ord(c) in chars for c in frag if any(
                        a <= ord(c) <= b for a, b in FOREIGN_RANGES)):
                    continue
            elif frag in glued:
                continue
            offenders.append(f"{rel}:{num} [{kind}] {frag}")
        self.assertEqual(offenders, [],
                         "чужие письма / врезанная латиница в русском тексте. "
                         "Исправлять текстом, а не списком исключений:\n  "
                         + "\n  ".join(offenders[:40])
                         + ("" if len(offenders) <= 40
                            else f"\n  ... и ещё {len(offenders) - 40}"))

    def test_exceptions_are_still_used(self):
        """
        Исключение обязано быть ЖИВЫМ. Сцитированный дефект убрали из файла, а
        запись оставили — и проверка молча расширяется: следующий иероглиф в том
        же файла уже не заметят. Поэтому каждое разрешение обязано по-прежнему
        находить то, ради чего записано.
        """
        scanned = {}
        for rel, num, kind, frag in self.scan():
            scanned.setdefault(rel, []).append((kind, frag))
        stale = []
        for rel, tokens in FOREIGN_ALLOWED.items():
            path = os.path.join(ROOT, rel)
            self.assertTrue(os.path.exists(path), f"исключение в несуществующем {rel}")
            src = read(path)
            for tok in tokens:
                if tok not in src:
                    stale.append(f"{rel}: «{tok}» больше нет в файле")
        self.assertEqual(stale, [],
                         "устаревшие исключения (снять, иначе проверка слепнет):\n  "
                         + "\n  ".join(stale))

    def test_scanner_is_not_blind(self):
        """
        Собственная самопроверка: проверка обязана видеть и иероглиф, и врезанную
        латиницу, и НЕ реагировать на нормальный русский текст. Без этого теста
        сломанный шаблон дал бы вечный зелёный — ровно так сломанный порог
        деградировал незаметно (см. docstring модуля).
        """
        bad_cjk = "невестка " + chr(0x501F)   # собрано: сканер видит и этот
        bad_glue = "фоновый" + "job"   # собрано: в исходнике дефекта быть не должно
        good = "невестка ждёт ответа: job-очередь UI-поток 5 \u00d7 3"
        self.assertTrue(any(any(a <= ord(c) <= b for a, b in FOREIGN_RANGES)
                            for c in bad_cjk), "шаблон не видит иероглиф")
        self.assertTrue(GLUED.search(bad_glue), "шаблон не видит врезанную латиницу")
        self.assertIsNone(GLUED.search(good),
                          "шаблон ловит нормальный текст (дефис обязан быть цел)")




# ─────────────────────────────────────────────────────────────
# СООБЩЕНИЯ КОММИТОВ. Тот же сканер, что для файлов (FOREIGN_RANGES +
# GLUED), тот же список исключений — ВТОРОГО списка нет по требованию
# владелицы. Проверка файлов порчу не ловит: текст коммита лежит в .git,
# а не в рабочем дереве. За сессию чужие буквы трижды попадали в МОИ
# собственные сообщения коммитов, и поймать их было нечем.
# ─────────────────────────────────────────────────────────────

# Коммит, с которого история обязана быть чистой. ВСЁ ДО НЕГО не проверяется
# (историю не переписываем) и не заносится в белый список: анкор — ОДНА
# точка, а не список прощений, который можно бесконечно пополнять.
#
# ВАЖНО: анкор выбран НЕ по принципу «последний чистый на глаз», а потому
# что с этого дня проверка существует. Если его удалить или переписать
# историю так, что его не станет, тест обязан покраснить, а не просканировать
# молча всё (см. test_anchor_commit_exists).
MSG_ANCHOR = "7e2b269"


def git(*args):
    import subprocess
    return subprocess.run(("git",) + args, cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def commit_messages(since=None):
    """[(хэш, сообщение)] — от newest к oldest. since:exclusive (сам не входит).

    strip() на ЗАПИСИ обязателен: git ставит перевод строки после каждой
    записи, и без него хэш каждой записи, кроме первой, приходит как "\n" +
    хэш. Замерено на живой истории: сравнение анкор-хэша со списком тогда
    НЕ находило анкор никогда, а «диапазон пуст» не срабатывал ни разу —
    проверка выглядела зелёной, пока HEAD случайно не совпал с анкором.
    """
    rev = f"{since}..HEAD" if since else "HEAD"
    raw = git("log", "--format=%H\x01%s\x01%b\x02", rev)
    out = []
    for rec in raw.split("\x02"):
        if not rec.strip():
            continue
        h, subj, body = rec.split("\x01")
        out.append((h.strip(), subj.strip() + "\n" + body.strip()))
    return out


def anchor_hash():
    """ПОЛНЫЙ хэш анкора. Сравнивать надо полным: git log отдаёт 40 символов,
    MSG_ANCHOR короткий, а молчаливое «не совпало» в такой проверке — это
    дыра, а не отказ."""
    return git("rev-parse", MSG_ANCHOR).strip()


def dirty_before_anchor():
    """Коммиты ДО anchor с чужими буквами. Живой материал для самопроверки."""
    ah = anchor_hash()
    after = {h for h, _ in commit_messages(ah)}
    return [h for h, msg in commit_messages()
            if h != ah and h not in after and scan_message(msg)]


def scan_message(text):
    """Те же правила, что для файлов: чужое письмо и сцепка кириллицы с латиницей."""
    hits = []
    for ch in text:
        if any(a <= ord(ch) <= b for a, b in FOREIGN_RANGES):
            hits.append(("письмо", ch))
            break
    for m in GLUED.finditer(text):
        hits.append(("сцепка", m.group()))
    return hits


class TestCommitMessagesClean(unittest.TestCase):

    def test_anchor_commit_exists(self):
        """
        Без этого теста проверка дырявая по построению: если анкор удалить или
        историю переписать так, что `git log ANCHOR..HEAD` станет пустым,
        следующий тест пройдёт молча на НУЛЕ коммитов и отчитается зелёным.
        """
        try:
            git("cat-file", "-e", MSG_ANCHOR + "^{commit}")
        except Exception:
            self.fail(f"анкор {MSG_ANCHOR} не найден: историю переписали или "
                      f"проверка потеряла точку отсчёта — пересмотреть, а не "
                      f"подобрать новый анкор потише")

    def test_anchor_is_clean_and_scope_is_not_empty(self):
        """
        Анкор обязан (1) существовать, (2) быть чистым и (3) оставлять за собой
        НЕПУСТОЙ диапазон. Иначе подбор точки отсчёта превращается в подгонку:
        «последний чистый» всегда найдётся, и проверка ничего не обещает.
        """
        anchor_msg = git("show", "-s", "--format=%B", MSG_ANCHOR)
        self.assertEqual([], scan_message(anchor_msg),
                         "анкор грязный — с него нельзя считать историю чистой")
        # Под катой ОСТАЮТСЯ грязные коммиты: значит мы что-то исключаем, а не
        # выставили анкор в самый хвост пустой истории.
        self.assertGreater(len(dirty_before_anchor()), 0,
                           "до анкора грязных коммитов нет: либо историю "
                           "зачистили, либо анкор подобран в самое начало — "
                           "проверка тогда ничего не обещает")

    def test_new_commit_messages_have_no_foreign_script(self):
        """
        От anchor и включая его — ни чужих писем, ни врезанной латиницы.
        Исключения общие (FOREIGN_ALLOWED) и здесь НЕ применяются: они про
        цитаты в файлах, а сообщение коммита цитатой не является.
        """
        offenders = []
        h_anchor = anchor_hash()
        for h, msg in commit_messages(h_anchor):
            for kind, frag in scan_message(msg):
                offenders.append(f"{h[:8]} [{kind}] {frag}")
        if h_anchor not in [h for h, _ in commit_messages()]:
            self.fail("анкор не является предком HEAD — диапазон проверки пуст")
        for h, msg in [(h_anchor, git("show", "-s", "--format=%B", h_anchor))]:
            for kind, frag in scan_message(msg):
                offenders.append(f"{h[:8]} [{kind}] {frag}")
        self.assertEqual(offenders, [],
                         "чужие письма / врезанная латиница в сообщении "
                         "коммита. Исправлять текстом коммита (amend), а не "
                         "списком исключений:\n  " + "\n  ".join(offenders[:20]))

    def test_parser_is_not_blind(self):
        """
        Парсер списка коммитов обязан проверять СЕБЯ, а не только сканер.
        Замеренный дефект: git добавляет \n после каждой записи, хэши сдвигались
        на одну строку, и единственное место, где это было видно (проверка
        «анкор — предок HEAD»), молча врал. Признаки правильного разбора:
          * хэш — ровно 40 шестнадцатеричных символов, без пробелов и \n;
          * записей ровно столько, сколько в git rev-list (ни потерянных,
            ни склеенных);
          * анкор присутствует в полном списке — иначе диапазон проверки пуст.
        """
        full = git("rev-list", "HEAD").split()
        msgs = commit_messages()
        self.assertEqual(len(msgs), len(full),
                         f"парсер вернул {len(msgs)} записей, а в истории "
                         f"{len(full)} — формат %H\\x01%b\\x02 разобран неверно")
        for h, msg in msgs:
            self.assertEqual(len(h), 40, f"хэш неверной длины: {h!r}")
            self.assertEqual(h, h.strip(), f"в хэше остался пробел: {h!r}")
            int(h, 16)
        self.assertIn(anchor_hash(), [h for h, _ in msgs],
                      "анкор не найден в истории — проверка новых сообщений "
                      "сканирует НУЛЬ коммитов и обязана краснить, а не молчать")

    def test_scope_excludes_anchor_and_is_anchored(self):
        """
        Диапазон «новые сообщения» обязан (а) НЕ включать анкор (он проверяется
        отдельно) и (б) быть непустым, если после анкора есть коммиты. Если
        диапазон пуст, следующий тест зелёный на пустоте.
        """
        ah = anchor_hash()
        scope = [h for h, _ in commit_messages(ah)]
        self.assertNotIn(ah, scope, "анкор попал в диапазон новых сообщений")
        newer = git("rev-list", f"{ah}..HEAD").split()
        self.assertEqual(len(scope), len(newer),
                         "диапазон проверки не совпадает с git rev-list — "
                         "сколько коммитов реально проверяется, неизвестно")

    def test_message_scanner_is_not_blind(self):
        """
        Сканер сообщений обязан видеть дефект. Без самопроверки пустой
        FOREIGN_RANGES/сломанный GLUED дали бы вечный зелёный на всём — тот же
        урок, что и с проверкой файлов (урок 71).
        """
        self.assertTrue(scan_message("невестка " + chr(0x501F)),
                        "сканер сообщений не видит иероглиф")
        self.assertTrue(scan_message("фоновый" + "job"),
                        "сканер сообщений не видит врезанную латиницу")
        self.assertEqual([], scan_message("Дыра F: потолок 163 не тронут, "
                                          "job-очередь UI-поток"),
                         "сканер сообщений ловит нормальный текст")

    def test_scanner_is_not_trivially_green_on_history(self):
        """
        Проверка обязана уметь краснеть НА ЖИВОМ МАТЕРИАЛЕ. В истории ДО anchor
        грязные коммиты ЕСТЬ (замерено: 6 из 121) — значит сканер, запущенный
        на всю историю, обязан найти ненулевое число. Если он найдёт ноль,
        сломан сканер, а не история стала чище.
        """
        dirty = [h for h, msg in commit_messages() if scan_message(msg)]
        self.assertGreater(len(dirty), 0,
                           "на всей истории чистых коммитов ноль — сканер "
                           "сообщений ослеп, проверка зелёная по болезни")


if __name__ == "__main__":
    unittest.main(verbosity=2)
