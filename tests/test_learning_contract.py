"""
Контракт обучения: текст -> знание, и НИКАКОЙ тихой порчи известного.

Это НЕ тест «красиво ли разобрано предложение». Каждый тест защищает
конкретную ошибку, которую я реально получил на этом коде:

  1. инверсия агент/объект («он купил автомобиль» -> «автомобиль умеет
     покупать») — опаснейший класс: неверное знание воспроизводится во всех
     ответах и отличить его от настоящего потом нельзя;
  2. потеря связи при перезапуске (граф собирался из JSON, а связи жили только
     в памяти процесса — обучение длилось «до закрытия окна»);
  3. повторный текст гонял LLM про уже известное слово;
  4. слияние записывало новую сущность ВМЕСТО старой, теряя свойства;
  5. молчаливая деградация без pymorphy3.

Тесты НЕ трогают llama.cpp: Учитель — заглушка со счётчиком, и число вызовов
измеряется напрямую. Эмбеддинги не загружаются: Learner получает core=None
там, где достаточно проверить записанные наблюдения.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import SimpleExtractor
from core.morph import Morph

BASE_WORLD = {
    "entities": {
        "кошка": {"concept": "кошка", "hierarchy": ["животные"],
                  "properties": ["пушистая"], "actions": ["спать"]},
        "флюгер": {"concept": "флюгер", "hierarchy": ["приборы"],
                   "properties": ["вращается"], "actions": []},
    },
    "observations": {},
    "unknown_words": {},
}


def facts_of(text, known=()):
    ex = SimpleExtractor(known_lemmas=set(known))
    out = []
    for sf in ex.extract_text(text):
        for f in sf.facts:
            out.append((f.head, f.relation, f.tail))
    return out


class TestExtractionContract(unittest.TestCase):
    """Золотой набор предложений: что ОБЯЗАНО и что НЕЛЬЗЯ извлечь."""

    def assertFact(self, text, triple):
        self.assertIn(triple, facts_of(text),
                      f"не извлечено {triple} из «{text}», получено {facts_of(text)}")

    def forbid(self, text, triple):
        self.assertNotIn(triple, facts_of(text),
                         f"извлечено ЗАПРЕЩЁННОЕ {triple} из «{text}»")

    # ── обязательное ───────────────────────────────────────
    def test_adjective_becomes_property(self):
        self.assertFact("Стол деревянный.", ("стол", "HAS_PROPERTY", "деревянный"))
        self.assertFact("Кошки пушистые.", ("кошка", "HAS_PROPERTY", "пушистый"))

    def test_verb_becomes_action(self):
        self.assertFact("Собака сторожит дом.", ("собака", "CAN_DO", "сторожить"))

    def test_object_attaches_to_verb_not_subject(self):
        """Объект относится к действию: иначе «кто» и «что» сливаются в кучу."""
        self.assertFact("Собака сторожит дом.", ("сторожить", "OBJECT_OF", "дом"))

    def test_word_order_does_not_change_roles(self):
        """Падеж важнее порядка: «рыбу съела кошка» — кошка eater, а не рыба."""
        self.assertFact("Рыбу съела кошка.", ("кошка", "CAN_DO", "съесть"))
        self.assertFact("Рыбу съела кошка.", ("съесть", "OBJECT_OF", "рыба"))
        self.forbid("Рыбу съела кошка.", ("рыба", "CAN_DO", "съесть"))

    def test_genitive_object_after_verb(self):
        """«боится грозы»: родительный — объект, а не «часть грозы»."""
        self.assertFact("Кошка боится грозы.", ("бояться", "OBJECT_OF", "гроза"))
        self.forbid("Кошка боится грозы.", ("кошка", "PART_OF", "гроза"))

    def test_possessive_genitive_is_part(self):
        self.assertFact("Хвост собаки мокрый.", ("хвост", "PART_OF", "собака"))

    def test_adjective_agreement_picks_right_noun(self):
        """Свойство едет к согласованному существительному, а не к первому."""
        self.assertFact("На широком подоконнике сидит кошка.",
                        ("подоконник", "HAS_PROPERTY", "широкий"))

    def test_location_not_identity(self):
        """Предлог места: «кошка на подоконнике» — НЕ «кошка есть подоконник»."""
        facts = facts_of("Кошка сидит на подоконнике.")
        self.assertIn(("сидеть", "LOCATED_IN", "подоконник"), facts)
        self.forbid("Кошка сидит на подоконнике.", ("кошка", "IS_ROLE", "подоконник"))

    def test_pronoun_actor_does_not_invert(self):
        """ГЛАВНЫЙ ТЕСТ. Местоимённый актёр: субъектом обязан стать никто,
        а не винительный объект."""
        self.forbid("Он купил красный автомобиль.",
                    ("автомобиль", "CAN_DO", "купить"))

    def test_instrumental_is_instrument(self):
        """Тв.п. = орудие: «открыла мышь» мышь не является объектом в
        смысле «пациент» и тем более не субъект."""
        self.forbid("Кошка открыла мышь.", ("мышь", "CAN_DO", "открыть"))

    def test_copula_is_role(self):
        self.assertFact("Вася — это инженер.", ("вася", "IS_A", "инженер")
                        if ("вася", "IS_A", "инженер") in facts_of("Вася — это инженер.")
                        else ("вася", "IS_ROLE", "инженер"))

    def test_possession_construct(self):
        self.assertFact("У кошки есть хвост.", ("кошка", "HAS_PART", "хвост"))
        self.forbid("У кошки есть хвост.", ("хвост", "IS_ROLE", "кошка"))

    def test_price_is_not_ability(self):
        """«Шкаф стоит 5000 рублей» — цена, а не умение шкафа «стоить»."""
        self.forbid("Шкаф стоит пять тысяч рублей.", ("шкаф", "CAN_DO", "стоить"))

    def test_temporal_instrumental_is_not_instrument(self):
        """Солнце светит НЕ «с помощью дня»."""
        self.forbid("Солнце светит днём.", ("светить", "INSTRUMENT", "день"))

    def test_negation_is_marked(self):
        from core.extract import SimpleExtractor as SE
        sf = SE().extract("Кошка не боится воды.")
        self.assertTrue(any(f.negated for f in sf.facts),
                        "отрицание потеряно: «не боится» станет «боится»")

    def test_extraction_is_deterministic(self):
        t = "Пушистая кошка боится грозы."
        self.assertEqual(facts_of(t), facts_of(t))


class StubTeacher:
    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or {}

    def __call__(self, lemma):
        self.calls.append(lemma)
        return self.answers.get(lemma)


class TestLearnerContract(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        self._write(json.loads(json.dumps(BASE_WORLD)))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, w):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(w, f, ensure_ascii=False)

    def _learner(self, teacher=None):
        from brain.learner import Learner
        return Learner(load_world=self._read, save_world=self._write,
                       core=None, teacher=teacher)

    def test_sentence_is_persisted(self):
        lr = self._learner()
        rep = lr.learn_from_text("Кошка сторожит мышь.", source="test")
        self.assertGreater(rep.observations_new, 0)
        obs = self._read()["observations"]
        self.assertTrue(any(k.startswith("кошка|") for k in obs),
                        f"связь не попала в файл: {list(obs)}")

    def test_relearn_is_idempotent_and_strengthens(self):
        """Повтор НЕ создаёт дублей, но повышает support — два независимых
        свидетельства это уже не случайность."""
        lr = self._learner()
        lr.learn_from_text("Кошка сторожит мышь.", source="t")
        first = json.dumps(self._read()["observations"], sort_keys=True)
        rep2 = lr.learn_from_text("Кошка сторожит мышь.", source="t")
        obs = self._read()["observations"]
        self.assertEqual(len(obs), len(json.loads(first)), "появились дубли связей")
        self.assertTrue(all(v["support"] == 2 for v in obs.values()),
                        f"поддержка не выросла: {[v['support'] for v in obs.values()]}")
        self.assertGreater(rep2.observations_seen, 0)

    def test_unknown_word_asked_once_not_every_time(self):
        """Неизвестное слово помнится: иначе каждое упоминание гонит LLM."""
        t = StubTeacher()
        lr = self._learner(teacher=t)
        lr.learn_from_text("Шмаздрит барсук.", source="t")
        first = len(t.calls)
        lr.learn_from_text("Шмаздрит барсук.", source="t")
        self.assertEqual(len(t.calls), first,
                         f"учитель вызван повторно: {t.calls}")

    def test_teacher_answer_creates_entity_with_provenance(self):
        t = StubTeacher({"флакон": {"hierarchy": ["посуда"],
                                    "properties": ["стеклянный"]}})
        lr = self._learner(teacher=t)
        lr.learn_from_text("Флакон стоит на полке.", source="урок")
        ent = self._read()["entities"].get("флакон")
        self.assertIsNotNone(ent, "выученная сущность не записана")
        self.assertEqual(ent["hierarchy"], ["посуда"])
        self.assertTrue(ent.get("provenance"), "нет происхождения знания")

    def test_forms_do_not_split_one_entity_in_two(self):
        """
        Во Вселенной сущность записана во МНОЖЕСТВЕННОМ числе («кошки»), а
        текст говорит «кошка». Если резолвер не увидит, что это одно имя,
        появится ВТОРАЯ запись о том же предмете, и знания разъедутся по двум
        ячейкам: у одной свойства, у другой — наблюдения, и ни одна не полна.
        Проверка идёт без ядра (known_forms=None) — только морфология.
        """
        w = self._read()
        w["entities"]["кошки"] = w["entities"].pop("кошка")
        w["entities"]["кошки"]["concept"] = "кошки"
        self._write(w)
        self._learner().learn_from_text("Кошка сидит и сторожит мышь.", source="t")
        w = self._read()
        dupes = [k for k in w["entities"] if k.startswith("кошк")]
        self.assertEqual(dupes, ["кошки"], f"появилась вторая запись о предмете: {dupes}")
        keys = list(w["observations"])
        self.assertTrue(keys and all(k.startswith("кошки|") for k in keys
                                     if k.split("|")[0].startswith("кошк")),
                        f"наблюдения уехали под другое имя: {keys}")

    def test_failed_teacher_does_not_create_fake_entity(self):
        """Сущность без иерархии = ложное «мы про неё знаем»."""
        lr = self._learner(teacher=StubTeacher({}))
        lr.learn_from_text("Зюзя.", source="t")
        self.assertNotIn("зюзя", self._read()["entities"])
        self.assertIn("зюзя", self._read()["unknown_words"])

    def test_low_confidence_facts_are_skipped(self):
        lr = self._learner()
        lr.min_confidence = 0.99
        lr.learn_from_text("Кошка сторожит мышь.", source="t")
        self.assertEqual(self._read()["observations"], {},
                         "низкоуверенный факт записан в память")

    def test_report_counts_are_honest(self):
        lr = self._learner()
        rep = lr.learn_from_text("Кошка сидит. Кошка пушистая.", source="t")
        self.assertEqual(rep.sentences, 2)
        self.assertGreaterEqual(rep.content_words, 3)
        self.assertGreaterEqual(rep.known_words, 1)
        self.assertIn("morph", rep.quality)

    def test_quality_reports_missing_morphology(self):
        """Без pymorphy3 разбор деградирует — отчёт обязан это сказать."""
        lr = self._learner()
        rep = lr.learn_from_text("Кошка сидит.", source="t")
        self.assertEqual(rep.quality["morph"], Morph.available)
        if not Morph.available:
            self.assertTrue(any("pymorphy" in n for n in rep.notes))


class TestMergeRule(unittest.TestCase):
    """
    Правило слияние ОДНО (core/records.merge_entity) и используется всеми
    каналами: извлечением из текста и учителем. Дважды в истории проекта
    разные места имели СВОЮ версию — каждая теряла данные по-своему, и
    выглядело это как «система забыла то, что знала».
    """

    def test_additive_never_destructive(self):
        from core.records import merge_entity
        old = {"concept": "кошка", "hierarchy": ["животные"],
               "properties": ["пушистая"], "actions": ["спать"],
               "observations": ["хвост"]}
        new = {"hierarchy": ["хищники"], "properties": ["хитрая"],
               "summary": "кошачьи"}
        m = merge_entity(old, new)
        self.assertEqual(m["hierarchy"], ["животные", "хищники"])
        self.assertEqual(m["properties"], ["пушистая", "хитрая"])
        self.assertEqual(m["actions"], ["спать"], "новое не должно стирать старое")
        self.assertEqual(m["observations"], ["хвост"],
                         "неизвестное поле удалено — так теряются наблюдения")
        self.assertEqual(m["summary"], "кошачьи", "пустое поле не заполнено")

    def test_scalar_not_overwritten_by_weaker(self):
        from core.records import merge_entity
        self.assertEqual(merge_entity({"hierarchy": ["птицы"]},
                                      {})["hierarchy"], ["птицы"])

    def test_provenance_accumulates_but_bounded(self):
        from core.records import merge_entity
        m = {}
        for i in range(40):
            m = merge_entity(m, {"provenance": [{"source": f"s{i}"}]})
        self.assertLessEqual(len(m["provenance"]), 12,
                             "доказательства растут бесконечно — файл станет нечитабельным")
        self.assertGreater(len(m["provenance"]), 1, "доказательства потеряны")

    def test_teacher_does_not_clobber_existing_entity(self):
        """
        Прямая проверка исправленной ошибки: base_entity записывался
        присваиванием и стирал всё, что у сущности было (включая выученные
        из рассказов наблюдения). Учитель пишет сам файл, поэтому тест
        проверяет его файл-письмо напрямую, без LLM и без llama.cpp.
        """
        from teachers.teacher1 import AgentTeacherV2
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "lipid_world.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"entities": {"кошка": {
                    "concept": "кошка", "hierarchy": ["животные"],
                    "properties": ["пушистая"], "actions": ["спать"]}},
                    "observations": {"кошка|HAS_PART|хвост": {"head": "кошка"}}},
                    f, ensure_ascii=False)
            t = AgentTeacherV2(host="http://127.0.0.1:1", memory_dir=d)
            t.write_json_file("кошка", {"base_entity": {
                "concept": "кошка", "hierarchy": ["млекопитающие"],
                "properties": ["домашняя"], "actions": ["мяукать"]}})
            w = json.load(open(path, encoding="utf-8"))
            ent = w["entities"]["кошка"]
            self.assertIn("животные", ent["hierarchy"], "иерархия стёрта учителем")
            self.assertIn("млекопитающие", ent["hierarchy"])
            self.assertIn("пушистая", ent["properties"], "свойства стёрты учителем")
            self.assertIn("кошка|HAS_PART|хвост", w["observations"],
                          "наблюдения потеряны при обновлении сущности")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestConcurrentWriter(unittest.TestCase):
    """
    AgentTeacherV2 пишет в lipid_world.json СВОИМ tool-вызовом из обработки
    ответа модели — т.е. в любой момент, пока learner «думает» над предложением
    (тот же интерпретатор, но вызов приходит изнутри teacher-цикла). Сохранение
    собственного снимка стёрло бы ответ учителя, полученный за реальные деньги
    и секунды ожидания. Здесь это моделируется правкой файла извне (учитель
    живёт в другом процессе и трогать его HTTP незачем).
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        self._write(json.loads(json.dumps(BASE_WORLD)))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, w):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(w, f, ensure_ascii=False)

    def test_foreign_writer_is_not_overwritten(self):
        from brain.learner import Learner
        lr = Learner(load_world=self._read, save_world=self._write, core=None)
        # учитель успел записать сущность ПОСЛЕ того, как мы прочитали мир
        real_read = self._read
        state = {"done": False}

        def read_then_teacher():
            w = real_read()
            if not state["done"]:
                state["done"] = True
                w2 = json.loads(json.dumps(w))
                w2["entities"]["подоконник"] = {"concept": "подоконник",
                                                "hierarchy": ["мебель"],
                                                "properties": ["деревянный"]}
                self._write(w2)
            return w
        lr.load_world = read_then_teacher
        lr.learn_from_text("У кошки есть хвост.", source="t")
        ent = self._read()["entities"]
        self.assertIn("подоконник", ent, "сущность учителя стёрта сохранением learner")
        self.assertEqual(ent["подоконник"]["hierarchy"], ["мебель"])
        self.assertIn("животные", ent["кошка"]["hierarchy"], "слиться перестали")
        self.assertTrue(any(k.startswith("кошка|") for k in self._read()["observations"]))


class TestReplayAcrossRestart(unittest.TestCase):
    """
    Перезапуск процесса — самый дешёвый способ поймать «обучение до закрытия
    окна». Граф собирается из JSON; если связи живут только в памяти, он их
    теряет, и выглядит это как полное незнание.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        self._write(json.loads(json.dumps(BASE_WORLD)))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, w):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(w, f, ensure_ascii=False)

    def test_observations_rebuild_edges(self):
        from brain.learner import Learner
        from core.graph import GraphMemory
        core1 = GraphMemory()
        lr = Learner(load_world=self._read, save_world=self._write, core=core1)
        lr.learn_from_text("Кошка сторожит мышь.", source="t")
        self.assertGreater(sum(len(n.out_synapses_typed) for n in core1.nodes.values()), 0)

        core2 = GraphMemory()
        lr2 = Learner(load_world=self._read, save_world=self._write, core=core2)
        n = lr2.replay_observations(self._read())
        self.assertGreater(n, 0, "связи не восстановлены из наблюдений")
        edges = {(h, t) for h, node in core2.nodes.items()
                 for t in node.out_synapses_typed}
        # Объект привязан к ДЕЙСТВИЮ, а не к субъекту — см. OBJECT_OF:
        # «кошка сторожит мышь» = кошка->сторожить->мышь.
        self.assertIn(("кошка", "сторожить"), edges)
        self.assertIn(("сторожить", "мышь"), edges)

    def test_replay_never_lowers_weight(self):
        """Повторное воспроизведение не должно делать память хуже."""
        from brain.learner import Learner
        from core.graph import GraphMemory
        core = GraphMemory()
        lr = Learner(load_world=self._read, save_world=self._write, core=core)
        lr.learn_from_text("Кошка сторожит мышь.", source="t")
        before = {(h, t): i["weight"] for h, n in core.nodes.items()
                  for t, i in n.out_synapses_typed.items()}
        lr.replay_observations(self._read())
        after = {(h, t): i["weight"] for h, n in core.nodes.items()
                 for t, i in n.out_synapses_typed.items()}
        for k, w in before.items():
            self.assertGreaterEqual(after.get(k, 0), w, f"вес {k} упал: {w} -> {after.get(k)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
