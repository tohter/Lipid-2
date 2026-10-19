"""
Где живёт память и кто имеет право менять её состояние.

Три контракта, каждый найден на живом прогоне (а не в теории):

1. ИЗОЛЯЦИЯ. Мозг, Учитель и Вики обязаны читать LIPID2_MEMORY_DIR. Пока
   переменную знал только тестовый мир, сквозной прогон GUI с
   LIPID2_MEMORY_DIR=/tmp писал В РАБОЧУЮ Вселенную: «учебные» пчела и цветок
   осели в настоящем lipid_world.json. Тест, который изолирует себя, но не
   изолирует проверяемый код, — это тест, который портит данные.

2. ПРОТИВОРЕЧИЕ. Выученное слово обязано исчезнуть из unknown_words. Пока оно
   лежит в обоих списках, файл одновременно утверждает «не знаю» и «вот что
   это такое», а панель «сущностей N · неизвестных M» выглядит сломанной.
   ЗАМЕРЕНО: после обучения «пчела» и «ввод» числились и известными, и
   неизвестными.

3. ОТВЕТ ПОЛЬЗОВАТЕЛЯ — тоже запись. Он беднее статьи Википедии, поэтому
   прямое присваивание стирало ранее выученные свойства того же слова.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestMemoryDirIsolation(unittest.TestCase):
    """Каталог памяти задаётся ОДНИМ способом для всех писателей файла."""

    CODE = (
        "import sys, json\n"
        "sys.path.insert(0, ROOTPLACEHOLDER)\n"
        "from brain.lipid2_brain import Lipid2Brain\n"
        "from teachers.teacher_wiki import WikiTeacher\n"
        "from teachers.teacher1 import AgentTeacherV2\n"
        "print(json.dumps([Lipid2Brain().world_filepath,\n"
        "                  WikiTeacher().world_filepath,\n"
        "                  AgentTeacherV2().world_filepath]))\n").replace(
            "ROOTPLACEHOLDER", repr(ROOT))

    def _paths(self, env_dir=None, explicit=None):
        env = dict(os.environ, LIPID2_STATUS="0")
        if env_dir:
            env["LIPID2_MEMORY_DIR"] = env_dir
        else:
            env.pop("LIPID2_MEMORY_DIR", None)
        code = self.CODE if explicit is None else self.CODE.replace(
            "Lipid2Brain()", "Lipid2Brain(memory_dir=%r)" % explicit)
        proc = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_env_var_moves_every_writer(self):
        d = tempfile.mkdtemp(prefix="l2mem-")
        paths = self._paths(env_dir=d)
        for p in paths:
            self.assertTrue(p.startswith(d),
                            f"{p} игнорирует LIPID2_MEMORY_DIR — кто-то пишет "
                            f"в рабочую память во время теста/тренировки")

    def test_explicit_argument_beats_env(self):
        env_d, arg_d = tempfile.mkdtemp(prefix="env-"), tempfile.mkdtemp(prefix="arg-")
        paths = self._paths(env_dir=env_d, explicit=arg_d)
        self.assertTrue(paths[0].startswith(arg_d))

    def test_default_is_project_data(self):
        paths = self._paths()
        for p in paths:
            self.assertEqual(p, os.path.join(ROOT, "data", "memory",
                                             "lipid_world.json"))


class TestUnknownContradiction(unittest.TestCase):
    """Выучил — забудь, что не знал."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2unk-")
        self.path = os.path.join(self.dir, "lipid_world.json")

    def _brain(self):
        os.environ["LIPID2_MEMORY_DIR"] = self.dir
        from brain.lipid2_brain import Lipid2Brain
        return Lipid2Brain(memory_dir=self.dir)

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)

    def test_learned_word_leaves_unknown(self):
        from core.worldfile import write_world
        write_world(self.path, {
            "entities": {"пчела": {"concept": "пчела", "properties": ["гудит"]}},
            "unknown_words": {"пчела": {"count": 2}, "опылять": {"count": 1}}})
        brain = self._brain()
        brain._forget_unknown("пчела", {"base_entity": {"concept": "пчела"}})
        world = json.load(open(self.path, encoding="utf-8"))
        self.assertNotIn("пчела", world["unknown_words"],
                         "слово выучено, но файл всё ещё говорит «не знаю»")
        self.assertIn("опылять", world["unknown_words"],
                      "чистка обязана точечной: чужие неизвестные слова "
                      "удалять нельзя")

    def test_forms_are_removed_too(self):
        from core.worldfile import write_world
        write_world(self.path, {
            "entities": {"джойстик": {"concept": "джойстик", "properties": ["x"]}},
            "unknown_words": {"джойстики": {"count": 1}}})
        brain = self._brain()
        brain._forget_unknown("джойстик", {"base_entity": {"concept": "джойстик"}})
        self.assertEqual(json.load(open(self.path, encoding="utf-8"))["unknown_words"],
                         {}, "форма слова осталась неизвестной, хотя ядро её "
                             "находит и отвечает по ней")

    def test_learner_removes_unknown_on_success(self):
        """Тот же порядок на пути обучения из текста (brain/learner.py)."""
        os.environ["LIPID2_MEMORY_DIR"] = self.dir
        from brain.learner import Learner
        from core.worldfile import write_world, read_world
        write_world(self.path, {"entities": {}, "observations": {},
                                "unknown_words": {"флюгер": {"count": 3}}})

        class Core:
            # Learner зовёт sync_lemmas, чтобы словарь форм узнавания совпал с
            # только что записанными сущностями. Здесь ядра нет — и словари
            # не нужны, поэтому тело пустое ОСОЗНАННО (контракт ядра проверяется
            # в test_learning_contract).
            def sync_lemmas(self):
                return None

        learner = Learner(load_world=lambda: read_world(self.path),
                          save_world=lambda w: write_world(self.path, w),
                          teacher=lambda c: {"hierarchy": ["приборы"],
                                             "properties": ["крутится"]},
                          core=Core())
        rep = learner.learn_from_text("флюгер", source="test", allow_teacher=True)
        self.assertIn("флюгер", rep.learned_concepts)
        world = read_world(self.path)
        self.assertIn("флюгер", world["entities"])
        self.assertNotIn("флюгер", world["unknown_words"],
                         "обучение прошло, а слово осталось в списке незнания")


class TestUserAnswerMerges(unittest.TestCase):
    """Ответ пользователя дополняет запись, а не заменяет её."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2user-")
        self.path = os.path.join(self.dir, "lipid_world.json")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)

    def test_existing_properties_survive_user_answer(self):
        from core.worldfile import write_world
        write_world(self.path, {
            "entities": {"джойстик": {
                "concept": "джойстик", "properties": ["есть стик"],
                "hierarchy": ["устройства"],
                "definition": "полное определение из Википедии"}},
            "unknown_words": {}})
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=self.dir)
        brain._learn_from_user("джойстик", "это устройство; с кнопками")
        ent = json.load(open(self.path, encoding="utf-8"))["entities"]["джойстик"]
        self.assertIn("есть стик", ent["properties"],
                      "ответ пользователя стёр то, что принёс Учитель")
        self.assertEqual(ent["definition"], "полное определение из Википедии")
        self.assertIn("с кнопками", ent["properties"])
        prov = json.dumps(ent.get("provenance"), ensure_ascii=False)
        self.assertIn("user", prov, "происхождение ответа пользователя не записано")


class TestChannelSwitchesReachBrain(unittest.TestCase):
    """
    Галочка «спрашивать Учителя» обязана гасить КАНАЛ, а не только обучение.

    ЗАМЕРЕНО на GUI-прогоне: переключатель читался лишь в do_learn, поэтому
    вопрос про незнакомое слово всё равно уходил в Gemma и висел ~70 с, после
    чего приходил «ответ без вызова инструмента». Пользователь видит выключенный
    канал ждёт минуту — это ровно тот класс «молчаливой лжи», который проект
    считает багом.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2chan-")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir
        from core.worldfile import write_world
        write_world(os.path.join(self.dir, "lipid_world.json"),
                    {"entities": {}, "unknown_words": {}})

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)

    def _brain(self):
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=self.dir)
        # Подменные учителя: считаем вызовы, знание НЕ даём.
        class Boom:
            def __init__(self):
                self.calls = 0

            def run_agent(self, **kw):
                self.calls += 1
                raise AssertionError("LLM вызван при выключенном канале")

            def fetch_concept(self, word):
                self.calls += 1
                raise AssertionError("интернет вызван при выключенном канале")

        brain.teacher_llm = Boom()
        brain.teacher_wiki = Boom()
        return brain

    def test_disabled_channels_are_not_called(self):
        brain = self._brain()
        brain.use_llm_teacher = False
        brain.use_wiki_teacher = False
        brain.ask_user_enabled = False
        brain.ask_user_callback = lambda c: None
        out = brain._trigger_curiosity_and_learn("зюзя")
        self.assertIsNone(out)
        self.assertEqual(brain.teacher_llm.calls, 0)
        self.assertEqual(brain.teacher_wiki.calls, 0)

    def test_says_why_channel_was_skipped(self):
        """Выключенный канал обязан СКАЗАТЬ об этом, а не молча промолчать."""
        brain = self._brain()
        brain.use_llm_teacher = False
        brain.use_wiki_teacher = False
        brain.ask_user_enabled = False
        brain.ask_user_callback = lambda c: None
        seen = []
        brain._say = lambda m, **kw: seen.append(m)
        brain._trigger_curiosity_and_learn("зюзя")
        text = " ".join(seen)
        self.assertIn("ВЫКЛЮЧЕН", text, f"молчание вместо отказа: {seen}")

    def test_enabled_channel_is_used(self):
        """Обратная сторона: включённый канал обязан работать, иначе проверка
        выше проходит на сломанном коде."""
        brain = self._brain()
        brain.use_llm_teacher = True
        brain.use_wiki_teacher = False
        brain.ask_user_enabled = True
        answers = []
        brain.ask_user_callback = lambda c: (answers.append(c) or
                                             "это тестовая сущность; свойство")
        out = brain._trigger_curiosity_and_learn("зюзя")
        self.assertIsNotNone(out, "включённый канал 3 не сработал")
        self.assertEqual(answers, ["зюзя"])


class TestGuiSwitchesReachBrainFlags(unittest.TestCase):
    """
    Мутация M6 (очередь, п.16): ГАЛОЧКА ОКНА — ПЕРЕКЛЮЧАТЕЛЬ, А НЕ УКРАШЕНИЕ.

    Дырка, из-за которой этот класс появился. Тесты TestChannelSwitchesReachBrain
    выставляют brain.use_llm_teacher / brain.use_wiki_teacher НАПРЯМУЮ. Значит
    они проверяют МОЗГ, но не ПРОВОД от окна к мозгу. Стоит сломать сам провод —
    убрать строку в _push_channel_flags (или, как в мутации M6, заменить тело на
    pass) — и ни один тест не краснеет:
      * brain-тесты проходят, потому что сами выставляют флаги;
      * test_gui_live/headless проверяет ТОЛЬКО 3-й канал (ask_user), а флаги
        Учителя (LLM/Вики) не трогает вовсе.
    ВОСПРОИЗВЕДЕНО 14.10.26: с телом `pass` в _push_channel_flags набор
    оставался зелёным (10 + 1 тестов OK).

    Здесь проверяется ИМЕННО ПРОВОД: вызывается НАСТОЯЩИЙ метод окна (unbound),
    и утверждается, что значение галочки ДОШЛО до мозга. Любая правка, после
    которой флаг не выставляется, обязана покраснить этот тест.
    """

    def _app(self):
        """Объект-переключатель: только те поля, что читает метод."""
        from ui.app_gui import Lipid2GUI
        class _Journal:
            def __init__(self):
                self.rows = []

            def add(self, text, *a, **k):
                self.rows.append(text)

        App = type("App", (), {})
        app = App()
        app.brain = type("Brain", (), {})()
        app.journal = _Journal()
        app.llm_enabled = True
        app.wiki_enabled = True
        app.ask_enabled = True
        app.experience_enabled = True
        app._push = Lipid2GUI._push_channel_flags.__get__(app, type(app))
        return app

    def test_llm_switch_reaches_llm_flag_only(self):
        """П.19: галочка LLM обязана доходить до мозга и НЕ трогать Вики."""
        app = self._app()
        app.llm_enabled = False
        app._push()
        self.assertFalse(app.brain.use_llm_teacher,
                         "галочка LLM НЕ дошла до LLM-канала: переключатель "
                         "стал украшением")
        self.assertTrue(app.brain.use_wiki_teacher,
                        "снятие LLM погасило Википедию — каналы склеены")

    def test_wiki_switch_reaches_wiki_flag_only(self):
        """П.19: вторая половина разделения — Вики гасится одна."""
        app = self._app()
        app.wiki_enabled = False
        app._push()
        self.assertFalse(app.brain.use_wiki_teacher,
                         "галочка Вики НЕ дошла до Вики-канала")
        self.assertTrue(app.brain.use_llm_teacher,
                        "снятие Вики погасило локальную модель — каналы склеены")

    def test_both_teacher_switches_off_reach_both_flags(self):
        app = self._app()
        app.llm_enabled = False
        app.wiki_enabled = False
        app._push()
        self.assertFalse(app.brain.use_llm_teacher)
        self.assertFalse(app.brain.use_wiki_teacher)

    def test_third_channel_alone_keeps_search_alive(self):
        """
        П.19: снятые УЧИТЕЛЯ (1+2) не имеют права отменять поиск знания, если
        включён 3-й канал. ЗАМЕРЕНО до правки: окно передавало
        allow_teacher=teacher_enabled, и при llm=off, wiki=off, ask=on вопрос
        пользователю не был задан НИ РАЗУ (rep.not_asked = 4 слова).
        """
        from ui.app_gui import Lipid2GUI
        app = type("App", (), {})()
        app.llm_enabled, app.wiki_enabled, app.ask_enabled = False, False, True
        self.assertTrue(
            Lipid2GUI._any_knowledge_channel.__get__(app, type(app))(),
            "3-й канал включён, но поиск знания отключён: переключатель "
            "Учителя глушит канал, который к нему не относится")
        app.ask_enabled = False
        self.assertFalse(
            Lipid2GUI._any_knowledge_channel.__get__(app, type(app))(),
            "все три канала сняты, а поиск всё равно обещан")

    def test_ask_switch_reaches_brain(self):
        app = self._app()
        app.ask_enabled = False
        app._push()
        self.assertFalse(app.brain.ask_user_enabled,
                         "галочка «спрашивать меня» НЕ дошла до мозга")

    def test_experience_switch_reaches_brain(self):
        app = self._app()
        app.experience_enabled = False
        app._push()
        self.assertFalse(app.brain.learn_experience,
                         "галочка опыта НЕ дошла до мозга")

    def test_each_switch_is_independent(self):
        """Четыре канала — ЧЕТЫРЕ разных флага; один выключенный не гасит остальные."""
        app = self._app()
        app.llm_enabled = False
        app.wiki_enabled = True
        app.ask_enabled = True
        app.experience_enabled = True
        app._push()
        self.assertFalse(app.brain.use_llm_teacher)
        self.assertTrue(app.brain.use_wiki_teacher,
                        "галочка LLM погасила Википедию — флаги перепутаны")
        self.assertTrue(app.brain.ask_user_enabled,
                        "галочка Учителя погасила 3-й канал — флаги перепутаны")
        self.assertTrue(app.brain.learn_experience)

    def test_error_in_push_is_reported_not_silent(self):
        """Если провод порвался исключением, окно обязано сказать, а не молчать."""
        app = self._app()
        class Anger:
            @property
            def use_llm_teacher(self):
                raise RuntimeError("провод оборван")
        app.brain = Anger()
        app._push()
        self.assertTrue(any("не дошли до мозга" in r for r in app.journal.rows),
                        "исключение в _push_channel_flags проглочено молча")


if __name__ == "__main__":
    unittest.main(verbosity=2)
