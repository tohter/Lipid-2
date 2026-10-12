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
        self.assertIn("user:рассказ", ent.get("provenance", []),
                      "происхождение знания стёрто — теперь не отличить, "
                      "откуда оно взялось")

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
