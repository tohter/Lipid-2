"""
Контракт файла Вселенной (core/worldfile.py) и правила записи учителей.

Что тут на самом деле проверяется
--------------------------------
lipid_world.json пишут четверо (мозг, learner, Учитель-агент, Википедия). Из
этого вытекают два свойства, которые ДО этого файла нигде не были обеспечены:

  1. Читатель никогда не должен увидеть ПОЛОВИНУ файла. Обычная запись
     open(path, "w") уссекает файл мгновенно: процесс, убитый во время
     json.dump (закрытие окна во время обучения — самый реальный сценарий),
     оставляет оборванный JSON, а следующий старт теряет ВСЮ память.

  2. «Файл не читается» НЕ равно «памяти нет». Прежний код трёх мест возвращал
     пустой мир; merge («перечитать и слить») поверх пустого мира = ЗАМЕНИТЬ
     Вселенную своим снимком. То есть защита от гонки writer'ов при битом файле
     сама становилась орудием стирания.

Плюс регресс на реальную ошибку: teacher_wiki писал world[concept] = clean_base
и стирал наблюдения/provenance уже известной сущности, хотя core/records.py
объявлен единственным местом, решающим, что имеет право перезаписать известное.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.worldfile import atomic_write_json, read_world, write_world


class TestAtomicWrite(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2world-")
        self.path = os.path.join(self.dir, "lipid_world.json")

    def test_roundtrip(self):
        write_world(self.path, {"entities": {"флюгер": {"properties": ["вертится"]}}})
        world = read_world(self.path)
        self.assertEqual(world["entities"]["флюгер"]["properties"], ["вертится"])

    def test_reads_do_not_see_partial_file(self):
        """
        Имитация обрыва СРЕДИ записи. Читатель обязан получить старую версию,
        а не обрывок: для этого временный файл лежит в том же каталоге и
        подменяется одним os.replace.
        """
        write_world(self.path, {"entities": {"кошка": {}}, "observations": {}})
        with open(self.path, "rb") as fh:
            good = fh.read()
        # роняем процесс-писателя на середине json.dump
        code = (
            "import json,sys,os,time\n"
            "sys.path.insert(0, %r)\n"
            "path = sys.argv[1]\n"
            "import core.worldfile as w\n"
            "class Boom(Exception): pass\n"
            "real = json.dump\n"
            "def fake(*a, **k):\n"
            "    real(*a, **k)\n"
            "    os._exit(9)   # смерть процесса в момент записи\n"
            "w.json.dump = fake\n"
            "w.atomic_write_json(path, {'entities': {'НОВОЕ': {}}})\n"
            % ROOT)
        script = os.path.join(self.dir, "killer.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(code)
        proc = subprocess.run([sys.executable, script, self.path],
                              capture_output=True, timeout=60)
        self.assertEqual(proc.returncode, 9, "писатель должен был умереть в записи")
        with open(self.path, "rb") as fh:
            self.assertEqual(fh.read(), good, "файл изменён посреди неудачной записи")
        world = read_world(self.path)
        self.assertIn("кошка", world["entities"], "старая версия не читается")

    def test_no_temp_junk_left_behind(self):
        write_world(self.path, {"entities": {}})
        atomic_write_json(self.path, {"entities": {"a": {}}})
        left = [f for f in os.listdir(self.dir) if f.startswith(".l2world-")]
        self.assertEqual(left, [], "неудачные/успешные записи не должны оставлять "
                                   "временные файлы в каталоге данных")

    def test_write_failure_leaves_previous_content(self):
        write_world(self.path, {"entities": {"старое": {}}})
        with self.assertRaises(TypeError):        # сериализовать нельзя
            atomic_write_json(self.path, {"x": {1, 2, 3}})   # set не сериализуется
        self.assertIn("старое", read_world(self.path)["entities"])

    def test_creates_missing_directory(self):
        deep = os.path.join(self.dir, "a", "b", "world.json")
        atomic_write_json(deep, {"entities": {}})
        self.assertTrue(os.path.exists(deep))

    def test_write_preserves_existing_file_mode(self):
        """
        П.15.2: атомарная запись НЕ имеет права менять права файла.

        ЗАМЕРЕНО на живой Вселенной: mkstemp создаёт временный файл 0600,
        os.replace сохраняет ЕГО режим — и мир после каждой записи терял
        права 0664 -> 0600. Пока пишет и читает один пользователь, это
        невидимо; как только пишет демон, а читает человек — непонятный
        отказ по правам. Тест обязан КРАСНЕТЬ без переноса режима.
        """
        write_world(self.path, {"entities": {}})
        os.chmod(self.path, 0o664)
        atomic_write_json(self.path, {"entities": {"a": {}}})
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o664,
                         "атомарная запись молча сменила права файла (0664 -> 0600)")

    def test_write_preserves_restrictive_mode(self):
        """Обратная сторона: если мир был закрыт (0600) — он обязан им остаться.

        Иначе лечение одной ошибки создало бы другую: запись РАСШИРЯЛА бы
        права секретного файла до умолчания.
        """
        write_world(self.path, {"entities": {}})
        os.chmod(self.path, 0o600)
        atomic_write_json(self.path, {"entities": {"a": {}}})
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600,
                         "запись расширила права закрытого файла")

    def test_new_file_gets_umask_default_not_0600(self):
        """Новый файл: обычные права (0666 & ~umask), а не 0600 от mkstemp."""
        import stat as _st
        # Прочитать текущую umask, не меняя её (два вызова возвращают исходную).
        old = os.umask(0)
        os.umask(old)
        try:
            # Ожидание СЧИТАЕТСЯ из umask машины, а не забито числом: на этой
            # машине umask = 002 (не 022), и жёсткое 0644 сделало бы тест
            # красным на здоровом коде — ложная тревога вместо проверки.
            expected = 0o666 & ~old
            atomic_write_json(self.path, {"entities": {}})
            mode = os.stat(self.path).st_mode & 0o777
            self.assertEqual(mode, expected,
                             f"новый мир получил {oct(mode)}, а не {oct(expected)} "
                             f"(umask {oct(old)})")
            self.assertTrue(_st.S_IMODE(mode))
        finally:
            os.umask(old)


class TestReadWorldSemantics(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2world-")
        self.path = os.path.join(self.dir, "lipid_world.json")

    def test_missing_file_is_normal_and_silent(self):
        world = read_world(self.path)
        self.assertEqual(world["entities"], {})
        self.assertIn("observations", world)
        self.assertIn("unknown_words", world)

    def test_broken_file_is_loud_and_preserved(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write('{"entities": {"флюгер": {"prop')      # оборванный JSON
        code = ("import sys\n"
                "sys.path.insert(0, ROOTPLACEHOLDER)\n"
                "from core.worldfile import read_world\n"
                "world = read_world(sys.argv[1])\n"
                "print('ENTITIES=' + str(len(world['entities'])))\n").replace(
                    "ROOTPLACEHOLDER", repr(ROOT))
        proc = subprocess.run([sys.executable, "-c", code, self.path],
                              capture_output=True, timeout=60, text=True)
        self.assertIn("ENTITIES=0", proc.stdout)
        self.assertIn("НЕ ЧИТАЕТСЯ", proc.stderr,
                      "битый файл обязан кричать: пустой мир без сообщения "
                      "выглядит как «памяти и не было»")
        broken = [f for f in os.listdir(self.dir) if "broken" in f]
        self.assertTrue(broken, "битые байты обязаны быть сохранены до перезаписи")
        with open(os.path.join(self.dir, broken[0]), encoding="utf-8") as fh:
            self.assertIn("флюгер", fh.read())

    def test_non_dict_json_is_not_silently_used(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("[1, 2, 3]")
        world = read_world(self.path)
        self.assertEqual(world["entities"], {})

    def test_missing_sections_are_added(self):
        """Старый файл без unknown_words не должен уронить читающий код."""
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entities": {"а": {}}}, fh)
        world = read_world(self.path)
        for key in ("observations", "unknown_words", "relations"):
            self.assertIn(key, world)


class TestWikiWriteMerges(unittest.TestCase):
    """
    Регресс: Википедия писала запись ЦЕЛИКОМ, затирая накопленное.

    Проверка строится на реальном write_to_world(), а не на пересказе логики:
    иначе тест осталось бы зелёным после возврата к присваиванию.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2world-")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir
        self.path = os.path.join(self.dir, "lipid_world.json")

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)

    def _wiki(self):
        from teachers.teacher_wiki import WikiTeacher
        return WikiTeacher(memory_dir=self.dir)

    def _read(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)

    def test_existing_observations_and_provenance_survive(self):
        write_world(self.path, {
            "entities": {"джойстик": {
                "concept": "джойстик",
                "properties": ["есть стик"],
                "actions": [],
                "hierarchy": ["устройство"],
                "provenance": ["user:рассказ"]}},
            "observations": {"джойтик|has|x": {"support": 3}},
            "unknown_words": {"юла": {}}})

        self._wiki().write_to_world({"base_entity": {
            "concept": "джойстик", "title": "Джойстик",
            "definitions": ["это устройство"],
            "hierarchy": ["устройство ввода"],
            "properties": ["есть кнопки"], "actions": [{"name": "двигают"}]}})

        ent = self._read()["entities"]["джойстик"]
        self.assertIn("есть стик", ent["properties"],
                      "свойство из раннего наблюдения пропало")
        self.assertIn("есть кнопки", ent["properties"], "новое свойство не добавилось")
        # П.14: происхождение — ЕДИНАЯ схема (объект), а не смесь строк и объектов.
        # Строка «user:рассказ» после слияния обязана стать объектом.
        prov = ent.get("provenance", [])
        sources = [p.get("source") for p in prov if isinstance(p, dict)]
        evidences = [p.get("evidence") for p in prov if isinstance(p, dict)]
        self.assertIn("user", sources,
                      "происхождение знания стёрто — теперь не отличить, "
                      "откуда оно взялось")
        self.assertIn("рассказ", evidences,
                      "доказательство пользователя стёрто при нормализации")
        self.assertTrue(all(isinstance(p, dict) for p in prov),
                        f"в provenance остались строки: {prov}")

        world = self._read()
        self.assertIn("джойтик|has|x", world["observations"],
                      "наблюдения не должны гибнуть при записи Вики")
        self.assertIn("юла", world["unknown_words"])

    def test_degenerate_derived_does_not_overwrite_base(self):
        write_world(self.path, {"entities": {"сеть": {
            "concept": "сеть", "properties": ["передаёт данные"], "actions": [],
            "hierarchy": ["техника"]}}})
        self._wiki().write_to_world({
            "base_entity": {"concept": "сеть", "title": "Сеть",
                            "hierarchy": [], "properties": [], "actions": []},
            "derived_entity": {"concept": "сеть", "hierarchy": ["топология"],
                               "properties": [], "actions": []}})
        ent = self._read()["entities"]["сеть"]
        self.assertIn("передаёт данные", ent["properties"])

    def test_write_is_atomic(self):
        """После записи не должно оставаться временных файлов рядом."""
        self._wiki().write_to_world({"base_entity": {
            "concept": "руль", "title": "Руль", "hierarchy": [],
            "properties": ["поворачивает"], "actions": []}})
        left = [f for f in os.listdir(self.dir) if f.startswith(".l2world-")]
        self.assertEqual(left, [])


class TestTeacherWriteIsAtomic(unittest.TestCase):
    """Учитель-агент пишет в тот же файл — его запись обязана быть такой же."""

    def test_write_world_used_by_teacher(self):
        from teachers import teacher1
        self.assertIsNotNone(teacher1.write_world,
                             "учитель пишет НЕатомарно: обрыв процесса = "
                             "битый lipid_world.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
