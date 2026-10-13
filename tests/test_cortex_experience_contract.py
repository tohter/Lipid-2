"""
Контракт предиктивного цикла (core/cortex.py + core/transitions.py) и его места
в мозге.

Что тут на самом деле проверяется
--------------------------------
Цикл «предсказание -> ошибка -> пластичность» был написан корректно как стенд и
жил в legacy/. Подключение его к живому мозгу «как есть» дало пять дефектов, и
каждый измерен ДО правки (числа в docstring core/cortex.py). Тесты ниже — не
украшение, а замки на этих замерах: без них следующий, кто «поправит одну
строчку», вернёт ровно то поведение, которое стоило дней отладки.

  A. Предсказание брало самый тяжёлый синапс графа. У узла из Вселенной самый
     тяжёлый — IS_A (180), поэтому на «кошка» цикл отвечал «кошачьи»: он
     предсказывал категорию, а не следующее слово опыта.
  B. Попаданием считалось overlap >= 13, то есть «кошка» вместо «кошки» и
     «огонь» вместо «пламя» шли в точность. Замер: мягкая мера 4/9, строгая 1/9.
  C. Пропущенное слово создавало узел графа: после 20 предложений 14 узлов
     стало 43, включая «а», «и», «на», «что», «когда».
  D. Наработанный STDP не переживал перезапуск (жил в lipid_db.json, а мозг
     читает lipid_world.json).
  E. И главное: цикл писал в ТОТ ЖЕ weight, который learner хранит как
     УВЕРЕННОСТЬ факта. Замер: один промах — IS_A 180 -> 116, одно попадание
     — 116 -> 231. После этого по весу нельзя отличить «факт сомнительный» от
     «следующее слово редко угадывали».

Отдельно проверяется, что цикл НЕ добывает знания: у Learner'а для этого есть
бюджеты, лимиты и статусы, и обход их делал бы счётчики каналов враньём.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("LIPID2_STATUS", "0")

from core.cortex import Hormones, PredictiveCortex
from core.graph import GraphMemory
from core.storage import RelationType
from core.transitions import (MAX_STRENGTH, TransitionTable, merge_json,
                              table_from_world)
from core.worldfile import EMPTY_WORLD, read_world, write_world
from core.sdr import SDR1024

TEXT = ("Кошка сидит на подоконнике. Кошка спит на диване. "
        "Собака бегает во дворе. Ребёнок читает книгу.")


def make_graph(words=("кошка", "собака", "ребёнок", "вода", "пламя")):
    g = GraphMemory()
    for w in words:
        g.get_or_create_node(w)
    g.sync_lemmas()
    return g


# ─────────────────────────────────────────────────────────────
# A + E. ПРАВА ЗАПИСИ: цикл не трогает онтологию
# ─────────────────────────────────────────────────────────────

class TestCortexHasNoWriteRightsToOntology(unittest.TestCase):

    def setUp(self):
        self.graph = GraphMemory()
        self.graph.get_or_create_node("кошка")
        self.graph.get_or_create_node("кошачьи")
        self.graph.get_or_create_node("молоко")
        self.graph.sync_lemmas()
        node = self.graph.nodes["кошка"]
        node.connect_to(self.graph.nodes["кошачьи"],
                        rel_type=RelationType.IS_A, weight=180)
        node.connect_to(self.graph.nodes["молоко"],
                        rel_type=RelationType.CAN_DO, weight=103)
        self.before = self._weights()

    def _weights(self):
        return {n.name: {tgt: syn["weight"]
                         for tgt, syn in n.out_synapses_typed.items()}
                for n in self.graph.nodes.values()}

    def test_ontology_weights_survive_many_ticks(self):
        """
        ГЛАВНЫЙ ЗАМОК. Раньше цикл был единственным писателем в weight, и
        180 -> 116 за один промах ломало смысл уверенности факта. Цикл обязан
        иметь СВОЁ число и не иметь доступа к чужому.
        """
        cortex = PredictiveCortex(self.graph)
        for _ in range(12):
            cortex.process_stream(TEXT, verbose=False)
            cortex.observe("кошка", "кошачьи")     # «попадание» в онтологию
            cortex.observe("кошка", "молоко")       # и «промах» по ней
        self.assertEqual(self._weights(), self.before,
                         "цикл изменил веса онтологии: уверенность факта и сила "
                         "перехода снова одно число")

    def test_prediction_ignores_synapse_weights(self):
        """
        Предсказание не должно зависеть от весов: 180-килограммовый IS_A больше
        не перетягивает ответ на «что идёт после кошки» (дефект A).
        """
        cortex = PredictiveCortex(self.graph)
        self.assertIsNone(cortex.predict("кошка"),
                          "без опыта предсказания быть не должно — вместо него "
                          "возвращается самый тяжёлый синапс")
        cortex.transitions.confirm("кошка", "молоко", 5)
        self.assertEqual(cortex.predict("кошка"), "молоко")

    def test_graph_structure_is_not_mutated(self):
        nodes_before = set(self.graph.nodes)
        edges_before = sum(len(n.out_synapses_typed) for n in self.graph.nodes.values())
        cortex = PredictiveCortex(self.graph)
        cortex.process_stream(TEXT, verbose=False)
        self.assertEqual(set(self.graph.nodes), nodes_before)
        self.assertEqual(sum(len(n.out_synapses_typed) for n in self.graph.nodes.values()),
                         edges_before)


# ─────────────────────────────────────────────────────────────
# C. НИКАКИХ НОВЫХ СУЩНОСТЕЙ ИЗ СЛУЖЕБНЫХ СЛОВ
# ─────────────────────────────────────────────────────────────

class TestStopWordsNeverBecomeConcepts(unittest.TestCase):

    def test_no_stopword_nodes_and_no_new_nodes_at_all(self):
        graph = make_graph()
        before = set(graph.nodes)
        cortex = PredictiveCortex(graph)
        cortex.process_stream(
            "Кошка, которая сидит на подоконнике, спит, а собака бегает и "
            "играет в мяч, когда ребёнок читает книгу про динозавра.",
            verbose=False)
        self.assertEqual(set(graph.nodes), before,
                         "цикл создал узлы: у него нет права на это (дефект C)")
        for junk in ("а", "и", "на", "что", "когда", "который", "в", "про"):
            self.assertNotIn(junk, graph.nodes)

    def test_word_forms_collapse_to_one_lemma(self):
        """
        «кошка» и «кошки» — ОДИН концепт. Пока цикл учил сырые формы, переход
        «кошка -> кошки» и «кошки -> кошка» были разными строками памяти.
        """
        cortex = PredictiveCortex(make_graph())
        chain = cortex.experience_sentences("Кошка видит кошку. Кошки видят кошку.")
        flat = [w for sent in chain for w in sent]
        self.assertEqual(flat.count("кошка"), 4, f"лемматизации нет: {flat}")

    def test_function_words_do_not_produce_ticks(self):
        cortex = PredictiveCortex(make_graph())
        m = cortex.process_stream("и на в а у о", verbose=False)
        self.assertEqual(m["ticks"], 0)


# ─────────────────────────────────────────────────────────────
# B. СТРОГАЯ И МЯГКАЯ МЕРЫ РАЗДЕЛЕНЫ
# ─────────────────────────────────────────────────────────────

class TestTwoMetricsStaySeparate(unittest.TestCase):

    def test_similar_word_is_not_a_strict_hit(self):
        """
        Близкое по битам слово — НЕ угаданное продолжение. Прежний критерий
        (overlap >= 13 ИЛИ имя == факт) засчитывал такие случаи в точность:
        ЗАМЕРЕНО на 9 controlных парах — мягкая мера даёт 4 «попадания» там,
        где строгая насчитывает 1.

        Пара выбрана по УСЛОВИЮ, которое проверяет сам тест: нужны две РАЗНЫЕ
        ЛЕММЫ с перекрытием не ниже мягкой меры. «На слух» такие пары не
        подобрать — цикл обучается на леммах, и формы одного слова ('кошка'
        /'кошки' = 18) в реальный путь не попадают. Замер: overlap('огонь',
        'пламя') = 17 из 20, обе леммы остаются леммами после разбора.
        """
        overlap = SDR1024("огонь").overlap(SDR1024("пламя"))
        self.assertGreaterEqual(overlap, PredictiveCortex.SOFT_OVERLAP,
                                f"пара перестала быть близкой: overlap={overlap}")
        from core.morph import analyze
        lemmas = {t["lemma"] or t["word"] for t in analyze("Огонь пламя")}
        self.assertEqual(lemmas, {"огонь", "пламя"},
                         "пара перестала быть двумя разными леммами: "
                         f"{lemmas} — тест проверяет уже не то")
        cortex = PredictiveCortex(make_graph())
        # Ожидание после «жар» — «огонь»; приходит «пламя».
        cortex.transitions.confirm("жар", "огонь", 4)
        tick = cortex.observe("жар", "пламя")
        self.assertEqual(tick["predicted"], "огонь")
        self.assertTrue(tick["soft_hit"], "мягкая мера перестала ловить близкое")
        self.assertFalse(tick["strict_hit"])
        final = cortex.metrics()
        self.assertEqual(final["hits"], 0)
        self.assertEqual(final["soft_overlap_hits"], 1)
        self.assertEqual(final["misses"], 1)
        self.assertEqual(final["accuracy_strict"], 0.0,
                         "близкое по битам слово попало в точность")

    def test_soft_measure_threshold_matches_recognition_threshold(self):
        """
        СОГЛАСОВАННОСТЬ МЕР. Мягкая мера сверки предсказания взята тем же
        числом (13), что и порог распознавания неизвестного. Смысл у них
        РАЗНЫЕ (и смешивать их нельзя — см. docstring модуля), но если пол
        распознавания уедет выше 13, «близко по битам» станет слабее порога
        «слово неизвестно», и мягкая мера начнёт засчитывать пары, которые
        цикл сам объявил бы нераспознанными. Тест обязан кричать об этом.
        """
        self.assertGreaterEqual(PredictiveCortex.UNKNOWN_CEILING,
                                PredictiveCortex.SOFT_OVERLAP)
        self.assertLessEqual(PredictiveCortex.UNKNOWN_CEILING,
                             PredictiveCortex.KNOWN_FLOOR)
        cortex = PredictiveCortex(make_graph())
        cortex.recalibrate_floor()
        thr = cortex.unknown_threshold()
        self.assertLessEqual(thr, PredictiveCortex.KNOWN_FLOOR,
                             "адаптивный порог вылез за замеры формы слова")
        self.assertGreaterEqual(thr, PredictiveCortex.SOFT_OVERLAP)

    def test_repeated_word_is_not_a_tick_at_all(self):
        """
        Прежний hit-критерий включал «имя == факт», и на неизвестном языке
        (оба имени — сырой токен одного слова) точность была 100%. Теперь
        такое наблюдение вообще не тик: это НЕ «успех» и НЕ «ошибка», а
        отсутствие наблюдения.
        """
        cortex = PredictiveCortex(make_graph())
        cortex.transitions.confirm("кошка", "спать", 3)
        result = cortex.observe("кошка", "кошка")
        self.assertTrue(result.get("skipped"))
        m = cortex.metrics()
        self.assertEqual(m["ticks"], 0)
        self.assertEqual(m["hits"], 0)

    def test_no_expectation_is_not_an_error_and_not_a_hit(self):
        """
        Пустой опыт обязан выглядеть как пустой опыт. Прежний hit-критерий
        «имя == факт» давал 100% точности на неизвестном языке, потому что оба
        имени были сырыми токенами одного слова.
        """
        cortex = PredictiveCortex(make_graph())
        # Каждая голова встречается ОДИН раз -> ожиданий возникнуть не может.
        fresh = "Собака бегает во дворе. Ребёнок читает книгу."
        m = cortex.process_stream(fresh, verbose=False)
        self.assertEqual(m["hits"], 0)
        self.assertEqual(m["misses"], 0)
        self.assertGreater(m["no_expectation"], 0)
        self.assertEqual(m["accuracy_strict"], 0.0)
        self.assertEqual(m["coverage"], 0.0,
                         "при нулевом опыте покрытие обязано быть нулевым, иначе "
                         "«0 ошибок» читается как «ни разу не ошиблись»")

    def test_strict_hit_eventually_happens(self):
        """
        Обратная сторона: строгая мера НЕ должна быть недостижимой. Без этого
        теста «0% навсегда» выглядел бы как корректная работа.
        """
        cortex = PredictiveCortex(make_graph())
        for _ in range(4):
            cortex.process_stream("Кошка пьёт молоко. Кошка пьёт молоко.",
                                  verbose=False)
        # Следующее ЗА КОШКОЙ значимое слово — «пить»; «молоко» идёт вторым
        # шагом и предсказывается уже из «пить».
        self.assertEqual(cortex.predict("кошка"), "пить")
        self.assertEqual(cortex.predict("пить"), "молоко")
        self.assertGreater(cortex.metrics()["hits"], 0)


# ─────────────────────────────────────────────────────────────
# D + персистентность: опыт переживает перезапуск
# ─────────────────────────────────────────────────────────────

class TestTransitionsPersistence(unittest.TestCase):

    def test_world_keeps_transitions_section(self):
        """
        read_world дописывает отсутствующие ключи из EMPTY_WORLD. Без ключа
        здесь писатель снимка молча убрал бы весь накопленный опыт.
        """
        self.assertIn("transitions", EMPTY_WORLD)
        d = tempfile.mkdtemp(prefix="l2tx-")
        try:
            path = os.path.join(d, "lipid_world.json")
            write_world(path, {"entities": {"кошка": {}}, "relations": [],
                               "observations": {}, "unknown_words": {}})
            self.assertIn("transitions", read_world(path))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_json_roundtrip_is_deterministic(self):
        t = TransitionTable()
        t.confirm("кошка", "молоко", 7)
        t.confirm("кошка", "спать", 3)
        t.confirm("вода", "течёт", 5)
        blob = t.to_json()
        self.assertEqual(blob, TransitionTable().from_json(blob).to_json())
        # порядок фиксирован: сила DESC, затем имя (иначе git показывает диффы
        # там, где данных не меняли)
        row = blob["rows"][0]
        self.assertEqual(row["head"], "вода")
        t2 = TransitionTable()
        t2.confirm("кошка", "спать", 3)
        t2.confirm("кошка", "молоко", 7)
        one = TransitionTable()
        one.confirm("кошка", "молоко", 7)
        one.confirm("кошка", "спать", 3)
        self.assertEqual(one.to_json()["rows"], t2.to_json()["rows"])

    def test_binary_roundtrip_and_magic(self):
        t = TransitionTable()
        t.confirm("кошка", "молоко", MAX_STRENGTH)
        t.confirm("кошка", "спать", 1)
        blob = t.encode()
        self.assertTrue(blob.startswith(b"LPD2TX"))
        back = TransitionTable().decode(blob)
        self.assertEqual(back.get("кошка", "молоко"), MAX_STRENGTH)
        self.assertEqual(back.get("кошка", "спать"), 1)
        self.assertEqual(back.degraded_reason, "")
        # Чужой образ обязан быть ОТВЕРГНУТ громко и с флагом: молчаливый
        # частичный образ или пустота означают «опыта нет» вместо «файл не от нас».
        foreign = TransitionTable().decode(b"XXXX00" + blob[6:])
        self.assertIn("LPD2TX", foreign.degraded_reason)
        self.assertEqual(foreign.size(), 0)
        truncated = TransitionTable().decode(blob[:len(blob) - 3])
        self.assertTrue(truncated.degraded_reason,
                        "оборот образ принят без единого слова")

    def test_saturates_instead_of_wrapping(self):
        t = TransitionTable()
        for _ in range(400):
            t.confirm("кошка", "молоко")
        self.assertEqual(t.get("кошка", "молоко"), MAX_STRENGTH)
        for _ in range(400):
            t.surprise("кошка", "молоко")
        self.assertEqual(t.get("кошка", "молоко"), 0)
        self.assertNotIn(("кошка", "молоко"), t)

    def test_corrupt_section_is_loud_not_empty(self):
        """
        «Не смог прочитать» и «переходов ещё нет» — РАЗНЫЕ статусы. Тот же
        урок, что asked_empty != not_asked: тихая пустота выглядит как успех.
        При этом приложение падать не должно: читающий путь обязан работать и
        с битой секцией, но ОБЯЗАН оставить след в degraded_reason.
        """
        t = TransitionTable()
        t.confirm("кошка", "молоко", 4)
        good = t.to_json()
        t2 = TransitionTable().from_json(dict(good, version=99))
        self.assertIn("версия", t2.degraded_reason)
        self.assertEqual(t2.size(), 0, "секцию чужой версии частично приняли")
        t3 = TransitionTable()
        t3.from_json({"version": 1, "pack": 2,
                      "rows": [{"head": "кошка", "tails": ["a", "б"],
                                "strengths": [3]}]})
        self.assertTrue(t3.degraded_reason, "рассинхрон массивов принят молча")
        t4 = TransitionTable()
        t4.from_json({"version": 1, "pack": 2,
                      "rows": [{"head": "кошка", "tails": ["a"],
                                "strengths": [99]}]})
        self.assertIn("потолка", t4.degraded_reason,
                      "сила вне шкалы молча срезана: это чужие данные, а не наши")

    def test_empty_section_is_not_degradation(self):
        t = TransitionTable().from_json({})
        self.assertEqual(t.degraded_reason, "")
        self.assertEqual(t.size(), 0)


class TestMergeIsMonotonic(unittest.TestCase):

    def test_merge_takes_max_and_is_commutative(self):
        a = TransitionTable()
        a.confirm("кошка", "пить", 3)
        a.confirm("кошка", "спать")
        b = TransitionTable()
        b.confirm("кошка", "пить", 1)
        b.confirm("вода", "течёт", 5)
        ab, ba = merge_json(a.to_json(), b.to_json()), merge_json(b.to_json(), a.to_json())
        self.assertEqual(ab, ba, "слияние зависит от порядка: кто раньше "
                                 "записался, тот стёр чужой опыт")
        m = TransitionTable().from_json(ab)
        self.assertEqual(m.get("кошка", "пить"), 3)
        self.assertEqual(m.get("кошка", "спать"), 1)
        self.assertEqual(m.get("вода", "течёт"), 5)

    def test_merge_never_shrinks_experience(self):
        a = TransitionTable()
        a.confirm("кошка", "пить", 9)
        empty = TransitionTable()
        m = TransitionTable().from_json(merge_json(a.to_json(), empty.to_json()))
        self.assertEqual(m.get("кошка", "пить"), 9)

    def test_bad_records_are_counted_aloud(self):
        bad = {"version": 1, "pack": 2,
               "rows": [{"head": "x", "tails": ["a", "б"], "strengths": [-1, 99]}]}
        good = TransitionTable()
        good.confirm("кошка", "пить", 2)
        merged = TransitionTable().from_json(merge_json(bad, good.to_json()))
        self.assertEqual(merged.get("кошка", "пить"), 2,
                         "кривая строка утащила за собой всю секцию")


# ─────────────────────────────────────────────────────────────
# ГРАНИЦА ПРЕДЛОЖЕНИЯ — РАЗРЫВ ОПЫТА
# ─────────────────────────────────────────────────────────────

class TestSentenceBoundary(unittest.TestCase):

    def test_no_transition_across_sentences(self):
        """
        Первый вариант склеивал текст в одну цепочку: «…на подоконнике. Кошка
        спит.» рождало переход «подоконник -> кошка». Иначе цикл выучает порядок
        предложений документа вместо структуры мира.
        """
        cortex = PredictiveCortex(make_graph())
        m = cortex.process_stream("Кошка сидит на подоконнике. Кошка спит.",
                                  verbose=False)
        self.assertEqual(m["ticks"], 3, f"тик через границу предложения: {m}")
        self.assertEqual(cortex.transitions.get("подоконник", "кошка"), 0)


class TestSessionCounters(unittest.TestCase):
    """
    Метрики должны описывать ТЕКУЩИЙ текст, а не всю жизнь процесса.

    Тот же дефект, что уже чинили с brain.channel_calls: счётчик был один и
    пожизненный. Отчёт второго урока печатал сумму первого и второго, и
    «угадано 18» при восьми тиках на уроке означало «неизвестно что». Проверка
    обязана быть обратимой: если после второго текста числа РАВНЫ первому —
    обнуления нет и проверка зелёна по ошибке.
    """

    FIRST = "Кошка пьёт молоко. Собака ест мясо."
    SECOND = "Ветер гонит облака. Дождь смывает пыль."

    def test_second_stream_does_not_sum_the_first(self):
        """Метрики второго текста = метрики того же текста на ЧИСТОМ цикле.

        Сравнивать с числом первого текста бессмысленно (разные предложения ->
        разное число тиков); сравнивать надо с ЭТАЛОНОМ: тот же текст, тот же
        граф, но цикл ни чего не видел раньше. Разница = протёкший прошлый опыт.
        """
        cortex = PredictiveCortex(make_graph())
        first = cortex.process_stream(self.FIRST, verbose=False)
        second = cortex.process_stream(self.SECOND, verbose=False)
        fresh = PredictiveCortex(make_graph())
        alone = fresh.process_stream(self.SECOND, verbose=False)
        for key in ("ticks", "hits", "misses", "no_expectation", "skipped",
                    "unknown_input", "new_transitions", "soft_overlap_hits"):
            self.assertEqual(second[key], alone[key],
                             f"{key}: на втором тексте {second[key]}, а на том же "
                             f"тексте с чистого листа {alone[key]} — числа первого "
                             f"текста протекли во второй (первый дал {first[key]})")
        self.assertEqual(second["accuracy_strict"], alone["accuracy_strict"])
        self.assertEqual(second["coverage"], alone["coverage"])

    def test_lifetime_survives_session_reset(self):
        """lifetime — тоже публичное число и обязано РАСТИ, а не обнуляться."""
        cortex = PredictiveCortex(make_graph())
        first = cortex.process_stream(self.FIRST, verbose=False)
        second = cortex.process_stream(self.SECOND, verbose=False)
        self.assertEqual(first["lifetime"]["ticks"], first["ticks"])
        self.assertEqual(second["lifetime"]["ticks"],
                         first["ticks"] + second["ticks"],
                         "пожизненный счётчик обнулили вместе с сеансом — "
                         "тогда его вообще нечего показывать")

    def test_report_numbers_cannot_exceed_ticks(self):
        """Арифметика отчёта: hits+misses+no_expectation == ticks.

        Счётчики правятся в двух ветках; если одна ветка забывает обнуление или
        удваивает, сумма расходится и ВСЕ проценты в отчёте врут.
        """
        cortex = PredictiveCortex(make_graph())
        for text in (self.FIRST, self.SECOND, self.FIRST, self.SECOND):
            m = cortex.process_stream(text, verbose=False)
            self.assertEqual(m["hits"] + m["misses"] + m["no_expectation"],
                             m["ticks"],
                             f"расхождение по тикам на «{text[:20]}…»: {m}")
            self.assertLessEqual(m["hits"], m["ticks"])
            self.assertLessEqual(m["accuracy_strict"], 1.0)

    def test_unknown_counter_fails_loud(self):
        """Опечатка в имени счётчика обязана упасть, а не завести новую корзину."""
        cortex = PredictiveCortex(make_graph())
        with self.assertRaises(KeyError):
            cortex._bump("hits_with_a_typo")

    def test_both_counter_sets_have_identical_keys(self):
        cortex = PredictiveCortex(make_graph())
        self.assertEqual(set(cortex.stats), set(cortex.lifetime),
                         "наборы счётчиков разошлись по полям — сравнивать их "
                         "станет нельзя, а расхождение не увидит никто")


# ─────────────────────────────────────────────────────────────
# ЦИКЛ НЕ ДОБЫВАЕТ ЗНАНИЯ
# ─────────────────────────────────────────────────────────────

class TestCortexDoesNotTeachItself(unittest.TestCase):

    def test_teacher_is_refused_not_silently_used(self):
        """
        Прежний _ask_teacher был duck-typing'ом поверх учителей: писал
        integrate_knowledge(weight=180) в обход merge_entity и не трогал
        channel_calls, то есть его работу было НЕЧЕМ проверить. Теперь refusal,
        и он громкий.
        """
        class FakeTeacher:
            def ask(self, *a, **kw):
                raise AssertionError("цикл сам вызвал Учителя")

        graph = make_graph()
        stderr_save, sys.stderr = sys.stderr, open(os.devnull, "w")
        try:
            cortex = PredictiveCortex(graph, teacher=FakeTeacher())
            cortex.process_stream(TEXT, verbose=False)
        finally:
            sys.stderr.close()
            sys.stderr = stderr_save
        self.assertEqual(set(graph.nodes), {"кошка", "собака", "ребёнок",
                                            "вода", "пламя"})

    def test_shared_hormones_are_not_decayed_by_the_loop(self):
        """
        Мозг показывает ОДНУ шкалу. Если цикл гасит чужие значения на каждом
        тике, «дофамин 255» в GUI перестаёт что-либо значить.
        """
        hormones = Hormones()
        hormones.dopamine = 200
        cortex = PredictiveCortex(make_graph(), hormones=hormones)
        self.assertTrue(cortex.metrics()["hormones_shared_with_brain"])
        cortex.process_stream(TEXT, verbose=False)
        self.assertGreaterEqual(hormones.dopamine, 200,
                                "цикл погасил гормоны владельца шкалы")
        owner = PredictiveCortex(make_graph())
        owner.hormones.dopamine = 200
        owner.process_stream("кошка молоко. кошка спать.", verbose=False)
        self.assertLess(owner.hormones.dopamine, 200,
                        "собственная шкала не затухает — цикл не «живёт»")


# ─────────────────────────────────────────────────────────────
# МОЗГ: ОТВЕТ, ОПЫТ И ВЫКЛЮЧАТЕЛЬ
# ─────────────────────────────────────────────────────────────

class TestBrainExperience(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2brain-")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brain(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
        return b

    def _world(self):
        path = os.path.join(self.dir, "lipid_world.json")
        if not os.path.exists(path):
            # мозг создаёт файл на первом сохранении; «файла нет» означает
            # «не записывали», а не «секция пустая»
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def test_experience_survives_brain_restart(self):
        """ЗАМОК на дефект D: опыт жил в lipid_db.json и пропадал при закрытии."""
        first = self._brain()
        first.learn_experience = True
        first.experience_from_text(TEXT, save=True)
        self.assertGreater(first.transitions.size(), 0)
        saved = self._world().get("transitions") or {}
        self.assertTrue(saved.get("rows"), "секция transitions пуста после обучения")

        second = self._brain()
        self.assertEqual(second.transitions.size(), first.transitions.size())
        heads = {r["head"] for r in saved["rows"]}
        self.assertIn("кошка", heads)
        for head in heads:
            self.assertIsNotNone(second.transitions.top_name(head),
                                 f"голова «{head}» не восстановилась из мира")

    def _world_transitions(self):
        path = os.path.join(self.dir, "lipid_world.json")
        if not os.path.exists(path):
            return 0
        rows = (self._world().get("transitions") or {}).get("rows") or []
        return sum(len(r.get("tails") or []) for r in rows)

    def test_disabled_loop_touches_nothing(self):
        """
        Снятый переключатель обязан означать «не трогает НИЧЕГО» — то же
        правило, что для каналов знания (там его нарушение стоило 70 с молчания
        на выключенном LLM).
        """
        brain = self._brain()
        brain.learn_experience = False
        rep = brain.learn_from_text(TEXT, allow_teacher=False)
        self.assertEqual(rep.experience, {"disabled": True})
        self.assertEqual(self._world_transitions(), 0)

    def test_report_shows_experience_numbers(self):
        brain = self._brain()
        rep = brain.learn_from_text(TEXT, allow_teacher=False)
        text = rep.as_text()
        self.assertIn("предсказание", text)
        self.assertIn("покрытие", text,
                      "точность без покрытия врёт: «0 ошибок» при нулевом опыте")

    def test_disabled_loop_is_visible_in_report(self):
        """
        Выключенный цикл обязан быть ВИДЕН в отчёте. Молчание в строке отчёта
        неотличимо от «работал и не нашёл» — за это проект уже платил.
        """
        brain = self._brain()
        brain.learn_experience = False
        self.assertIn("ВЫКЛЮЧЕН", brain.learn_from_text(TEXT,
                                                        allow_teacher=False).as_text())

    def test_experience_does_not_create_entities(self):
        brain = self._brain()
        before = set(self._world().get("entities", {}))
        brain.experience_from_text(
            "Динозавр ест папоротник, а птеродактиль летит над вулканом.",
            save=True)
        self.assertEqual(set(self._world().get("entities", {})), before,
                         "цикл породил сущности: у него нет на это права")

    def test_transitions_section_survives_other_writers(self):
        """
        Мир пишут четверо. Сохранение наблюдений не обязано стирать секцию
        опыта (и наоборот) — иначе «пережил перезапуск» верно только пока никто
        больше не писал файл.
        """
        brain = self._brain()
        brain.experience_from_text(TEXT, save=True)
        rows = self._world()["transitions"]["rows"]
        brain.learn_from_text("Вулкан извергает лаву. Лава остывает.",
                              allow_teacher=False)
        after = {(r["head"], t): s for r in self._world()["transitions"]["rows"]
                 for t, s in zip(r["tails"], r["strengths"])}
        for row in rows:
            for tail, strength in zip(row["tails"], row["strengths"]):
                self.assertGreaterEqual(after.get((row["head"], tail), 0), strength,
                                        f"переход «{row['head']}->{tail}» "
                                        "уменьшился после чужой записи")
        # и новый опыт добавился, а не заменил старый
        self.assertIn("вулкан", {r["head"] for r in
                                 self._world()["transitions"]["rows"]})


class TestExperienceParticipatesInAnswer(unittest.TestCase):
    """
    Очередь п.24: Experience участвует в ОТВЕТЕ, а не только считает точность.

    Замер, который замок обязан удержать: цикл — измерительный прибор,
    но его предсказание обязано менять порядок ответа и быть видимым.
    И одновременно: у него НЕТ прав записи и он молчит, когда выключен.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2ans-")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brain_with_experience(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
        # опыт: после «кошка» всегда «пить» (4 подтверждения -> сила 4)
        for _ in range(4):
            b.experience_from_text("Кошка пьёт молоко.", save=False)
        # и знание: кошка умеет пить И спать (два кандидата — есть что упорядочить)
        world = b._load_world()
        world.setdefault("observations", {})["кошка|CAN_DO|пить"] = {
            "head": "кошка", "relation": "CAN_DO", "tail": "пить",
            "confidence": 0.9, "negated": False, "evidence": "t",
            "source": "text", "support": 1}
        world["observations"]["кошка|CAN_DO|спать"] = {
            "head": "кошка", "relation": "CAN_DO", "tail": "спать",
            "confidence": 0.9, "negated": False, "evidence": "t",
            "source": "text", "support": 1}
        world.setdefault("entities", {})["кошка"] = {
            "concept": "кошка", "definition": "домашнее животное",
            "hierarchy": ["кошачьи"], "properties": [], "actions": ["пить"]}
        b._save_world(world)
        return b

    def test_prediction_is_public_and_reads_experience(self):
        b = self._brain_with_experience()
        pred = b.predict_next("кошка")
        self.assertTrue(pred, "опыт накоплен, а предсказание пусто")
        self.assertEqual(pred[0]["tail"], "пить")
        self.assertEqual(pred[0]["strength"], 4)
        self.assertAlmostEqual(pred[0]["norm"], 4 / float(MAX_STRENGTH), places=4)

    def test_prediction_resolves_word_forms(self):
        """«кошки» и «кошку» — та же лемма, что хранит цикл."""
        b = self._brain_with_experience()
        for form in ("кошка", "кошки", "кошку", "кошкой"):
            pred = b.predict_next(form)
            self.assertTrue(pred, f"форма {form!r} не нашла опыт леммы")
            self.assertEqual(pred[0]["tail"], "пить")

    def test_disabled_loop_predicts_nothing(self):
        """Выключенный цикл молчит — предсказание не выдумывается."""
        b = self._brain_with_experience()
        b.learn_experience = False
        self.assertEqual(b.predict_next("кошка"), [])

    def test_answer_orders_expected_action_first(self):
        """Опыт меняет ПОРЯДОК ответа и говорит, почему это первое."""
        b = self._brain_with_experience()
        ans = b.process_input("Что делает кошка?")
        self.assertIn("опыт", ans.lower(),
                      "ответ не показал, что опыт участвовал")
        self.assertIn("пить", ans)
        self.assertIn("4/15", ans, "сила предсказания обязана быть видимой числом")

    def test_prediction_creates_no_entities(self):
        """Чтение опыта — не запись. Сущностей не появляется."""
        b = self._brain_with_experience()
        before = set(b._load_world().get("entities", {}))
        b.predict_next("динозавр")
        b.process_input("Что делает динозавр?")
        self.assertEqual(set(b._load_world().get("entities", {})), before)

    def test_interaction_reports_facts_not_placeholder(self):
        """Взаимодействие читает связи, а не печатает заглушку про симулятор."""
        b = self._brain_with_experience()
        ans = b.process_input("Что происходит, когда кошка видит молоко")
        self.assertNotIn("Векторный просчёт", ans)


class TestForceIsConsumedVisibly(unittest.TestCase):
    """
    Очередь п.25: сила (переход) не копится молча.

    Вторая половина п.25: цикл должен уметь ТОРМОЗИТЬ/ОКЛИКАТЬ запись, которой
    опыт противоречит. Полномочий писать в онтологию у него нет (это замок из
    TestCortexHasNoWriteRightsToOntology), поэтому расход силы выражен как
    ВИДИМЫЙ диагноз расхождения.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2force-")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brain(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
        return b

    def _add_obs(self, b, head, tail, relation="CAN_DO", weight=180, negated=False):
        world = b._load_world()
        world.setdefault("observations", {})[f"{head}|{relation}|{tail}"] = {
            "head": head, "relation": relation, "tail": tail,
            "confidence": 0.9, "negated": negated, "weight": weight,
            "evidence": "t", "source": "text", "support": 1}
        b._save_world(world)

    def test_strong_ontology_without_experience_is_flagged(self):
        """Вес есть, опыт молчит — это расхождение, а не «всё хорошо»."""
        b = self._brain()
        self._add_obs(b, "вода", "гасить", weight=180)
        conflicts = b.experience_conflicts()
        kinds = {c["kind"] for c in conflicts}
        self.assertIn("ontology_without_experience", kinds)

    def test_conflict_is_reported_loudly_after_experience(self):
        b = self._brain()
        self._add_obs(b, "вода", "гасить", weight=180)
        metrics = b.experience_from_text("Вода гасит пламя.", save=False)
        self.assertIn("conflicts", metrics)
        self.assertGreaterEqual(metrics["conflicts"], 1)

    def test_strong_experience_without_ontology_is_flagged(self):
        """Опыт ждёт сильно, а связи в онтологии нет — тоже расхождение."""
        b = self._brain()
        for _ in range(8):
            b.experience_from_text("Кошка пьёт молоко.", save=False)
        kinds = {c["kind"] for c in b.experience_conflicts()}
        self.assertIn("experience_without_ontology", kinds)

    def test_negated_observation_is_not_a_conflict(self):
        """Отрицание — не «онтология утверждает»: это отдельный флаг."""
        b = self._brain()
        self._add_obs(b, "вода", "гасить", weight=180, negated=True)
        kinds = {c["kind"] for c in b.experience_conflicts()}
        self.assertNotIn("ontology_without_experience", kinds)

    def test_conflict_inspector_writes_nothing(self):
        """Инспектор — только чтение: мир до и после совпадает побайтово."""
        b = self._brain()
        self._add_obs(b, "вода", "гасить", weight=180)
        path = os.path.join(self.dir, "lipid_world.json")
        with open(path, "rb") as fh:
            before = fh.read()
        b.experience_conflicts()
        with open(path, "rb") as fh:
            after = fh.read()
        self.assertEqual(before, after, "инспектор изменил файл Вселенной")


if __name__ == "__main__":
    unittest.main(verbosity=2)
