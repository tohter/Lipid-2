"""
Контракт миграции накопленных данных (tools/clean_world.py).

Правильное чтение чинит БУДУЩЕЕ, файл чинит ПРОШЛОЕ: пока правила filters
применяются только при записи, в lipid_world.json остаётся всё, что нажито
до правки. Поэтому у миграции есть свои тесты — она меняет настоящую память,
и ошибка здесь дороже, чем ошибка в ответе.

Все проверки идут на ВРЕМЕННОМ файле. Реальный lipid_world.json тесты не
трогают ни при каком раскладе (проверяется отдельным тестом).
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "clean_world", os.path.join(ROOT, "tools", "clean_world.py"))
clean_world = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean_world)

from core.worldfile import read_world, write_world                   # noqa: E402


class TestPlan(unittest.TestCase):

    def test_tautology_is_cut_from_hierarchy(self):
        world = {"entities": {"экосистема": {"concept": "экосистема",
                                             "hierarchy": ["экосистемы", "природа"]}},
                 "unknown_words": {}}
        clean, changes = clean_world.plan(world)
        self.assertEqual(clean["entities"]["экосистема"]["hierarchy"], ["природа"])
        self.assertTrue(changes, "миграция молчала там, где есть что чинить")

    def test_broken_definition_is_dropped_but_real_one_kept(self):
        """
        Критерий по ЗАМЕРУ реального файла: «Пчёлы (лат» — обрывок, а
        122-символьное определение «ввод» без точки в конце — знание.
        Правило «нет точки = мусор» сожрало бы оба настоящих определения.
        """
        world = {"entities": {
            "пчела": {"concept": "пчела", "definition": "Пчёлы (лат"},
            "ввод": {"concept": "ввод",
                     "definition": "Термин ввод обозначает либо появление, "
                                   "либо изменение условий, которые влияют "
                                   "на систему"},
        }, "unknown_words": {}}
        clean, _ = clean_world.plan(world)
        # Запись, у которой после среза не осталось ничего, обязана УЕХАТЬ в
        # unknown_words: «Пчёлы (лат» — не знание, а его видимость.
        self.assertNotIn("пчела", clean["entities"],
                         "огрызок остался притворяться знанием")
        self.assertIn("пчела", clean["unknown_words"])
        self.assertIn("definition", clean["entities"]["ввод"],
                      "настоящее определение удалено как обрывок")

    def test_entity_and_unknown_cannot_both_be_true(self):
        world = {"entities": {"пчела": {"concept": "пчела", "properties": ["гудит"]}},
                 "unknown_words": {"пчела": {"count": 2}}}
        clean, changes = clean_world.plan(world)
        self.assertNotIn("пчела", clean["unknown_words"])
        self.assertIn("пчела", clean["entities"])
        self.assertTrue(any("unknown_words" in c for c in changes))

    def test_unknown_without_knowledge_stays(self):
        """Слово, у которого нечего оставить, обязано остаться неизвестным."""
        world = {"entities": {"зюзя": {"concept": "зюзя"}},
                 "unknown_words": {}}
        clean, _ = clean_world.plan(world)
        self.assertNotIn("зюзя", clean["entities"])
        self.assertIn("зюзя", clean["unknown_words"])

    def test_plan_does_not_mutate_input(self):
        world = {"entities": {"флюгер": {"concept": "флюгер",
                                         "hierarchy": ["флюгеры", "приборы"]}},
                 "unknown_words": {}}
        before = json.dumps(world, sort_keys=True)
        clean_world.plan(world)
        self.assertEqual(json.dumps(world, sort_keys=True), before,
                         "plan изменил входной мир — «показ» опасен")

    def test_plan_does_not_mutate_nested_unknown_records(self):
        """То же обещание, но на ВЛОЖЕННЫХ словарях — где оно и нарушалось.

        dict(world["unknown_words"]) копирует словарь ПОПЕРЕК: запись внутри —
        ТОТ ЖЕ объект. Шаг 6 («нет status -> not_asked») писал прямо в неё, и
        ЗАМЕР показывал: после plan() у АРГУМЕНТА появляется status и
        status_note. Прошлый тест этого не видел, потому что unknown_words у
        него был пуст — тест обязан доставать до того места, где больно.
        """
        world = {"entities": {"флюгер": {"concept": "флюгер",
                                         "definition": "Флюгер — это прибор."}},
                 "unknown_words": {"рыбак": {"reason": "знания нет"}},
                 "observations": {}, "relations": []}
        before = json.dumps(world, sort_keys=True)
        out, changes = clean_world.plan(world)
        self.assertEqual(json.dumps(world, sort_keys=True), before,
                         "plan мутирует вложенную запись аргумента: «просто "
                         "посмотреть» меняет данные в памяти вызывающего")
        self.assertEqual(out["unknown_words"]["рыбак"]["status"], "not_asked",
                         "при переделке записи под new-словарь миграция обязана остаться")
        self.assertTrue(any("status" in c for c in changes))

    def test_idempotent(self):
        world = {"entities": {"луга": {"concept": "луга",
                                       "hierarchy": ["луга", "деревни"]}},
                 "unknown_words": {}}
        once, ch1 = clean_world.plan(world)
        twice, ch2 = clean_world.plan(once)
        self.assertEqual(json.dumps(once, sort_keys=True),
                         json.dumps(twice, sort_keys=True))
        self.assertEqual(ch2, [], "повторный прогон нашёл то, что уже исправил")

    def test_mixed_provenance_normalized_to_objects(self):
        """
        П.14: в одном списке лежали И строки «wiki:Флюгер», И объекты
        {source, evidence, at}. Миграция обязана привести обе формы к объекту.
        """
        world = {"entities": {"экосистема": {
            "concept": "экосистема",
            "properties": ["биологическая система"],
            "provenance": ["wiki:Экосистема", {"source": "user",
                                                  "evidence": "рассказ"}]}},
            "unknown_words": {}}
        out, changes = clean_world.plan(world)
        prov = out["entities"]["экосистема"]["provenance"]
        self.assertTrue(all(isinstance(p, dict) for p in prov),
                        f"в provenance остались строки: {prov}")
        sources = {p["source"] for p in prov}
        self.assertEqual(sources, {"wiki", "user"})
        self.assertTrue(any("provenance" in c for c in changes),
                        "миграция provenance не отчиталась об изменении")

    def test_provenance_migration_is_idempotent(self):
        world = {"entities": {"экосистема": {
            "concept": "экосистема",
            "properties": ["биологическая система"],
            "provenance": ["wiki:Экосистема"]}},
            "unknown_words": {}}
        once, _ = clean_world.plan(world)
        twice, ch2 = clean_world.plan(once)
        self.assertEqual(ch2, [],
                         "миграция provenance не идемпотентна")
        self.assertEqual(once["entities"]["экосистема"]["provenance"],
                         twice["entities"]["экосистема"]["provenance"])

    def test_sections_unknown_to_the_migrator_are_carried_through(self):
        """
        Миграция пересобирает мир, а не правит на месте, и список переносимых
        ключей был ЖЁСТКИМ: всё, чего чистильщик не знает, исчезало из файла
        без единого слова. ЗАМЕРЕНО на секции «transitions» (сила переходов
        предиктивного цикла): план не нашёл ни одного изменения, а запись после
        него уменьшила Вселенную на весь накопленный опыт. Худший вид потери
        данных — та, о которой не сказали.
        """
        sections = {
            "transitions": {"version": 1, "pack": 2,
                            "rows": [{"head": "кошка", "tails": ["спать"],
                                      "strengths": [3]}]},
            "совсем_неизвестный_ключ": {"важное": [1, 2, 3]},
        }
        world = {"entities": {"пчела": {"concept": "пчела",
                                        "definition": "Пчёлы (лат",
                                        "hierarchy": ["пчёлы"]}},
                 "relations": [], "observations": {}, "unknown_words": {}}
        world.update(sections)
        clean, changes = clean_world.plan(world)
        for name, payload in sections.items():
            self.assertIn(name, clean, f"секция «{name}» выброшена из мира")
            self.assertEqual(clean[name], payload,
                             f"секция «{name}» искажена чистильщиком, который "
                             "про неё ничего не знает")
        # и что миграция РАБОТАЕТ, а не просто копирует всё подряд
        self.assertTrue(changes)
        self.assertIn("пчела", clean["unknown_words"])


class TestWriteMode(unittest.TestCase):

    def test_backup_created_and_file_stays_readable(self):
        d = tempfile.mkdtemp(prefix="l2clean-")
        path = os.path.join(d, "lipid_world.json")
        broken = {"entities": {"флюгер": {"concept": "флюгер",
                                          "hierarchy": ["флюгеры"]}},
                  "unknown_words": {}}
        write_world(path, broken)
        rc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "clean_world.py"),
             "--write"],
            env=dict(os.environ, LIPID2_MEMORY_DIR=d, LIPID2_STATUS="0"),
            capture_output=True, text=True, timeout=120)
        self.assertEqual(rc.returncode, 0, rc.stderr[-500:])
        self.assertTrue(os.path.exists(path + ".pre-clean"),
                        "исходник не сохранён: ошибочная миграция невычислима")
        world = read_world(path)
        self.assertTrue(world.get("unknown_words", {}).get("флюгер")
                        or world.get("entities", {}).get("флюгер"),
                        "запись исчезла совсем вместо переноса")

    def test_no_write_flag_leaves_file_untouched(self):
        d = tempfile.mkdtemp(prefix="l2clean2-")
        path = os.path.join(d, "lipid_world.json")
        raw = json.dumps({"entities": {"ф": {"concept": "ф",
                                             "hierarchy": ["фы"]}},
                          "unknown_words": {}}, ensure_ascii=False)
        with io.open(path, "w", encoding="utf-8") as fh:
            fh.write(raw)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                     "clean_world.py"), "--show"],
                       env=dict(os.environ, LIPID2_MEMORY_DIR=d, LIPID2_STATUS="0"),
                       capture_output=True, text=True, timeout=120)
        with io.open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), raw, "--show переписал файл")


class TestVerbStatusMigration(unittest.TestCase):
    """
    asked_empty на ГЛАГОЛЕ — диагноз, выставленный неверным вопросом.

    Прежний код спрашивал «гасить» так же, как «пчелу»: найди онтологию
    сущности. Википедия на глагол такой статьи не даёт, слово получало
    статус «спрашили — пусто» и NEVER спрашивалось снова. Мусор в файле,
    который выглядит как результат поиска, — опаснее пустоты: его никто не
    идёт проверять.
    """

    def test_asked_empty_on_verb_is_lifted(self):
        world = {"entities": {}, "observations": {},
                 "unknown_words": {"гасить": {"status": "asked_empty",
                                              "count": 3,
                                              "evidence": "Вода гасит пламя"}}}
        clean, changes = clean_world.plan(world)
        rec = clean["unknown_words"]["гасить"]
        self.assertEqual(rec["status"], "not_asked",
                         f"неверный диагноз пережил миграцию: {rec}")
        self.assertTrue(any("гасить" in c and "глагол" in c for c in changes),
                        f"снятие статуса промолчано: {changes}")

    def test_noun_asked_empty_survives(self):
        """Для существительного asked_empty ЧЕСТНЫЙ: трогать его нельзя."""
        world = {"entities": {}, "observations": {},
                 "unknown_words": {"пчела": {"status": "asked_empty", "count": 2}}}
        clean, changes = clean_world.plan(world)
        self.assertEqual(clean["unknown_words"]["пчела"]["status"], "asked_empty",
                         f"честный диагноз существительного сломан: {changes}")

    def test_infinitive_recognised_as_verb(self):
        """
        Леммы глаголов в памяти — ИНФИНИТИВЫ, а у них часть речи INFN, не VERB.
        Без этого живая память (гасить/чинить/опылять) осталась бы отравленной.
        """
        from core.actions import looks_like_verb
        for v in ("гасить", "чинить", "опылять", "катиться"):
            self.assertTrue(looks_like_verb(v), f"«{v}» не распознан как действие")
        for n in ("пчела", "цветок", "вода", "берег", "сеть", "ввод"):
            self.assertFalse(looks_like_verb(n), f"«{n}» ложно назван действием")

    def test_legacy_entries_without_status_become_not_asked(self):
        """
        Записи СТАРЫХ уроков поля status не застали. Поведенчески они и так
        трактуются как «не спрашивали», но файл обязан быть самоописанным:
        пустое поле — это не утверждение, а его отсутствие.
        ЗАМЕРЕНО в живой памяти: 6 из 7 unknown-записей без status.
        """
        world = {"entities": {}, "observations": {},
                 "unknown_words": {"берег": {"count": 1, "evidence": "на берегу"},
                                   "рыбак": {"count": 2}}}
        clean, changes = clean_world.plan(world)
        for key in ("берег", "рыбак"):
            self.assertEqual(clean["unknown_words"][key]["status"], "not_asked",
                             f"«{key}» осталось без выверенного статуса")
        self.assertTrue(any("нет status" in c for c in changes),
                        f"дописание статуса промолчано: {changes}")
        # идемпотентность: повторный проход не плодит изменений
        again, ch2 = clean_world.plan(clean)
        self.assertEqual(again["unknown_words"], clean["unknown_words"])
        self.assertEqual([c for c in ch2 if "нет status" in c], [])

    def test_migration_is_idempotent_on_verb_status(self):
        world = {"entities": {}, "observations": {},
                 "unknown_words": {"чинить": {"status": "asked_empty"}}}
        once, ch1 = clean_world.plan(world)
        twice, ch2 = clean_world.plan(once)
        self.assertEqual(twice["unknown_words"]["чинить"],
                         once["unknown_words"]["чинить"])
        self.assertTrue(ch1 and not ch2,
                        f"идемпотентность нарушена: {ch2}")


class TestPathArgument(unittest.TestCase):
    """
    `clean_world.py /путь/к/файлу` обязан чистить ЭТОТ файл.

    ЗАМЕРЕНО: позиционный путь игнорировался, и «покажи мусор вот в этой
    памяти» печатал цифры рабочей памяти Вселенной — то есть оператор получал
    «мусора не найдено» про совершенно другой файл.
    """

    def _run(self, path, extra_env=None):
        env = dict(os.environ, LIPID2_STATUS="0")
        env.pop("LIPID2_MEMORY_DIR", None)
        env.pop("LIPID2_WORLD_FILE", None)
        env.update(extra_env or {})
        return subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "clean_world.py"),
             path, "--show"], env=env, capture_output=True, text=True, timeout=180)

    def test_positional_path_is_used(self):
        d = tempfile.mkdtemp(prefix="l2path-")
        path = os.path.join(d, "lipid_world.json")
        write_world(path, {"entities": {}, "observations": {},
                           "unknown_words": {"гасить": {"status": "asked_empty"}}})
        res = self._run(path)
        self.assertIn(path, res.stdout, f"путь не показан: {res.stdout}")
        self.assertIn("гасить", res.stdout,
                      "переданный файл не был разобран (показан чужой)")

    def test_positional_path_beats_env(self):
        d = tempfile.mkdtemp(prefix="l2path2-")
        path = os.path.join(d, "lipid_world.json")
        write_world(path, {"entities": {}, "observations": {},
                           "unknown_words": {"гасить": {"status": "asked_empty"}}})
        res = self._run(path, {"LIPID2_MEMORY_DIR": "/nonexistent-dir-l2"})
        self.assertIn("гасить", res.stdout,
                      "env победил явный путь — оператор чистит не тот файл")


class TestCheckMode(unittest.TestCase):
    """
    --check обязан ОТВЕЧАТЬ ЧИСЛОМ, а не только печатать текст.

    Причина отдельная и злая: до появления флага показ при мусоре возвращал 0.
    Значит любая «проверка состояния памяти» в скрипте или CI была бы вечно
    зелёной — а именно тихую деградацию этот проект и считает багом (memory.txt).
    Каждый тест здесь проверяет КОД ВОЗВРАТА, а не подстроку в выводе: вывод
    может соврать, rc — нет.

    Три кода, а не два: 0 чисто, 1 грязно, 2 проверка НЕ СОСТОЯЛАСЬ. Разделение
    повторяет выверенный в проекте принцип not_asked / asked_empty — «не
    проверено» и «проверено, грязно» лечатся по-разному, и смешав их в единицу
    мы заставили бы оператора запускать --write на файле, который не открылся.
    """

    DIRTY = {"entities": {"пчела": {"concept": "пчела"},
                          "флюгер": {"concept": "флюгер",
                                     "hierarchy": ["флюгеры"],
                                     "definition": "Флюгер — это прибор."}},
             "observations": {}, "unknown_words": {}, "relations": []}

    CLEAN = {"entities": {"флюгер": {"concept": "флюгер",
                                     "hierarchy": ["приборы"],
                                     "definition": "Флюгер — это прибор."}},
             "observations": {}, "unknown_words": {}, "relations": []}

    def _run(self, path, *flags):
        env = dict(os.environ, LIPID2_STATUS="0")
        env.pop("LIPID2_MEMORY_DIR", None)
        env.pop("LIPID2_WORLD_FILE", None)
        return subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "clean_world.py"),
             path, *flags], env=env, capture_output=True, text=True, timeout=180)

    def _file(self, d, name, payload):
        path = os.path.join(d, name)
        with io.open(path, "w", encoding="utf-8") as fh:
            fh.write(payload if isinstance(payload, str)
                     else json.dumps(payload, ensure_ascii=False))
        return path

    def test_dirty_file_returns_one(self):
        d = tempfile.mkdtemp(prefix="l2chk1-")
        path = self._file(d, "lipid_world.json", self.DIRTY)
        res = self._run(path, "--check")
        self.assertEqual(res.returncode, 1,
                         f"грязный файл дал rc={res.returncode} — проверка "
                         f"зелена на мусоре:\n{res.stdout}\n{res.stderr}")

    def test_clean_file_returns_zero(self):
        d = tempfile.mkdtemp(prefix="l2chk0-")
        path = self._file(d, "lipid_world.json", self.CLEAN)
        res = self._run(path, "--check")
        self.assertEqual(res.returncode, 0,
                         f"чистый файл Красный — проверка плачет на порядке:\n"
                         f"{res.stdout}\n{res.stderr}")

    def test_check_writes_absolutely_nothing(self):
        """Грязный файл + --check: ни файла, ни бэкапа, ни копии битых байтов.

        Проверка, которая пишет, — это уже не проверка. Бэкап .pre-clean и
        побэкап .broken.json появляются только в режимах записи.
        """
        d = tempfile.mkdtemp(prefix="l2chk3-")
        path = self._file(d, "lipid_world.json", self.DIRTY)
        with io.open(path, "rb") as fh:
            before = fh.read()
        res = self._run(path, "--check")
        self.assertEqual(res.returncode, 1)
        with io.open(path, "rb") as fh:
            self.assertEqual(fh.read(), before, "--check переписал файл")
        extra = sorted(n for n in os.listdir(d) if n != "lipid_world.json")
        self.assertEqual(extra, [], f"--check оставил в каталоге: {extra}")

    def test_broken_file_is_not_clean(self):
        """Битый JSON — НЕ «чисто». read_world вернул бы пустой мир, и наивная
        проверка на plan() сказала бы «мусора нет» про файл, в котором знания
        есть, но не читаются.
        """
        d = tempfile.mkdtemp(prefix="l2chk4-")
        path = self._file(d, "lipid_world.json", '{"entities": {"а":')
        res = self._run(path, "--check")
        self.assertEqual(res.returncode, 2,
                         f"нечитаемый файл получил rc={res.returncode}\n{res.stdout}")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("НЕ ЧИТАЕТСЯ", res.stdout)
        self.assertFalse(os.path.exists(os.path.join(d, "lipid_world.broken.json")),
                         "сама проверка скопировала битые байты — она пишет")

    def test_missing_file_is_not_clean(self):
        d = tempfile.mkdtemp(prefix="l2chk5-")
        res = self._run(os.path.join(d, "нет-такого.json"), "--check")
        self.assertEqual(res.returncode, 2,
                         "неверный путь дан как «чисто» — это тот же класс "
                         "неверный путь дан как «чисто» — это тот же класс "
                         "лжи, что и вечнозелёная проверка\n" + res.stdout)

    def test_check_and_write_cannot_both_be_requested(self):
        d = tempfile.mkdtemp(prefix="l2chk6-")
        path = self._file(d, "lipid_world.json", self.DIRTY)
        with io.open(path, "rb") as fh:
            before = fh.read()
        res = self._run(path, "--check", "--write")
        self.assertEqual(res.returncode, 2)
        with io.open(path, "rb") as fh:
            self.assertEqual(fh.read(), before,
                             "противоречивые флаги всё-таки записали")

    def test_check_agrees_with_write(self):
        """Сходимость: после --write проверка ОБЯЗАНА стать зелёной.

        Иначе --check красен всегда, и его перестают запускать — флаг, который
        все игнорируют, хуже отсутствующего: он создаёт видимость контроля.
        """
        d = tempfile.mkdtemp(prefix="l2chk7-")
        path = self._file(d, "lipid_world.json", self.DIRTY)
        self.assertEqual(self._run(path, "--check").returncode, 1)
        wr = self._run(path, "--write")
        self.assertEqual(wr.returncode, 0, wr.stdout + wr.stderr)
        again = self._run(path, "--check")
        self.assertEqual(again.returncode, 0,
                         f"--write отработал, а проверка всё ещё красная:\n"
                         f"{again.stdout}")

    def test_check_does_not_drop_transitions_or_unknown_sections(self):
        """--check обязан видеть те же секции, что и план.

        План уже переносит незнакомые ключи (иначе стирался бы опыт цикла),
        но проверка обязана согласовываться с планом и на них: если --check
        говорит «чисто», а план что-то пересобирает — один из двух врёт.
        """
        d = tempfile.mkdtemp(prefix="l2chk8-")
        world = json.loads(json.dumps(self.CLEAN))
        world["transitions"] = {"version": 1, "pack": 2, "rows": [
            {"head": "кошка", "tails": ["спать"], "strengths": [3]}]}
        world["секция_которой_я_не_знаю"] = {"важное": 1}
        path = self._file(d, "lipid_world.json", world)
        res = self._run(path, "--check")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        with io.open(path, encoding="utf-8") as fh:
            after = json.load(fh)
        self.assertEqual(len(after["transitions"]["rows"]), 1)
        self.assertIn("секция_которой_я_не_знаю", after)


class TestRealMemoryNotTouchedByTests(unittest.TestCase):
    """
    Тесты не имеют права писать в рабочую память. Проверка — контрольная сумма
    реального файла ДО и ПОСЛЕ запуска всего набора (запускается в подпроцессе,
    чтобы другие тесты уже успели отработать).
    """
    PATH = os.path.join(ROOT, "data", "memory", "lipid_world.json")

    def test_checksum_stable_across_a_dummy_import(self):
        if not os.path.exists(self.PATH):
            self.skipTest("рабочей памяти ещё нет")
        with io.open(self.PATH, "rb") as fh:
            before = fh.read()
        env = dict(os.environ, LIPID2_STATUS="0")
        env.pop("LIPID2_MEMORY_DIR", None)
        subprocess.run([sys.executable, "-c", "import brain.lipid2_brain"],
                       cwd=ROOT, env=env, capture_output=True, timeout=180)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                     "clean_world.py"), "--show"],
                       env=env, capture_output=True, timeout=120)
        with io.open(self.PATH, "rb") as fh:
            self.assertEqual(fh.read(), before,
                             "показ/импорт мозга изменил рабочую память")


if __name__ == "__main__":
    unittest.main(verbosity=2)
