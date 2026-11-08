# -*- coding: utf-8 -*-
"""Пересказ НЕ подтверждает. Достоверность растёт только от ОТДЕЛЬНОГО
свидетельства.

ЗАМЕР 08.11 (experiments, живой прогон ДО всякой правки). Одна и та же фраза
«Дождь гасит огонь.», трижды прогнанная через обучающий:

    прогон 1: дождь|CAN_DO|гасить   support=1  weight=200
    прогон 2: дождь|CAN_DO|гасить   support=2  weight=210
    прогон 3: дождь|CAN_DO|гасить   support=3  weight=220

Ни одного НОВОГО наблюдения не было: мир не двигался, учитель ничего не
приносил. Движение было только в процессе — и оно превратилось в убеждённость.
Это ровно то, что Хозяин запретил словом 08.11: «одна история не должна
становиться пожизненным ребром». Ребёнок, ОДИН раз бросивший мяч и испугавшийся
его, имеет право на гипотезу; три повтора ЕГО ЖЕ рассказа об этом эпизоде не
дают права на знание.

ЧТО ЗАКРЕПЛЕНО ЭТИМ ФАЙЛОМ.

  Свидетельство считается ОТДЕЛЬНЫМ, если отличается ИСТОЧНИК или ТЕКСТ-
  ОСНОВАНИЕ. Совпали оба — это пересказ: support не растёт, weight не
  пересчитывается, в provenance не добавляется строка.

  Честная цена правила (и она часть контракта, а не случайный побочный
  эффект). Омонимичные формулировки разных настоящих событий тоже совпадут:
  «мяч отскочил», сказанное сегодня и вчера одним каналом, — мы их НЕ согреем.
  Мы теряем часть подтверждений и НИКОГДА не наживаем ложную убеждённость.
  Выбор направлен сознательно: ложная связь хуже пропущенной — закон проекта;
  ложная уверенность в связи — тот же грех в более тихой форме, потому что
  её нечем перепроверить (число выглядит заработанным).

  Поэтому же правило стоит в ОДНОМ месте (core/records.py::add_witness) и
  вызывается обоими писателями наблюдений. Два писателя с двумя правилами —
  это дыра, которую мы уже чинили кирпичом ранее: brain писал в свою схему,
  learner — в свою, и читатель умел читать только одного.

ЧТО УМРЁТ ПРИ МУТАЦИЯХ: прибавка support без проверки отдельности; сравнение
только источника (пересказ одним каналом пройдёт); сравнение только текста
(тот же текст из другого канала не пройдёт, а это законное свидетельство);
порча weight при пересказе; дубль в provenance.

ЗАЧЕМ ЗДЕСЬ ЖЕ ПЕРЕПИСАН СТАРЫЙ ТЕСТ. tests/test_learning_contract.py
содержал test_relearn_is_idempotent_and_strengthens с docstring'ом «два
независимых свидетельства это уже не случайность» — при этом он подавал
ОДНУ И ТУ ЖЕ фразу с ОДНИМ И ТЕМ ЖЕ source. независимость в нём заявлялась
словом, а не создавалась опытом, и арифметика (support == 2) охраняла дыру.
Он переписан так же: дублей нет, а подтверждения нет.
"""

import json
import os
import shutil
import tempfile
import unittest

from core.records import add_witness, independent_bases, record_weight, observation_weight
from core.worldfile import read_world, write_world
from lang.ru.lemma import install as _install_lemma_service

_install_lemma_service()

from brain.lipid2_brain import Lipid2Brain       # noqa: E402
from brain.learner import Learner                # noqa: E402


class TestRetellingIsNotProof(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entities": {}, "observations": {}, "unknown_words": {}},
                      fh, ensure_ascii=False)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _learner(self):
        return Learner(load_world=lambda: read_world(self.path),
                       save_world=lambda w: write_world(self.path, w))

    def _obs(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)["observations"]

    # ── 1. одна и та же фраза не убеждает ───────────────────────
    def test_1_replay_of_same_sentence_does_not_warm(self):
        """Три прогона ОДНОЙ фразы = одно свидетельство, а не три."""
        lr = self._learner()
        for _ in range(3):
            lr.learn_from_text("Дождь гасит огонь.", source="text",
                               allow_teacher=False)
        obs = self._obs()
        self.assertTrue(obs, "связь вообще не записана")
        for key, rec in obs.items():
            self.assertEqual(1, int(rec.get("support", 0)),
                             f"пересказ одной фразы поднял support по {key}: "
                             f"замер до правки давал 1 -> 2 -> 3")
            self.assertIsNotNone(rec.get("weight"), "нет числа достоверности")

    def test_2_doubles_still_absent(self):
        """Не греть — не значит множить: дублей связи не было и нет."""
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        first = dict(self._obs())
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        self.assertEqual(sorted(first), sorted(self._obs()),
                         "перестали сворачивать: появились дубли связей")

    def test_3_weight_untouched_by_replay(self):
        """Пересказ не имеет права двигать ЧИСЛО достоверности."""
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        before = {k: observation_weight(v) for k, v in self._obs().items()}
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        after = {k: observation_weight(v) for k, v in self._obs().items()}
        self.assertEqual(before, after,
                         "убеждённость выросла без нового наблюдения")

    def test_4_no_duplicate_provenance(self):
        """Пересказ не добавляет в журнал вторую строку о том же."""
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        key = next(iter(self._obs()))
        marks = [(p.get("source"), p.get("evidence"))
                 for p in (self._obs()[key].get("provenance") or [])]
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        marks_after = [(p.get("source"), p.get("evidence"))
                       for p in (self._obs()[key].get("provenance") or [])]
        self.assertEqual(marks, marks_after,
                         "журнал пухнет от пересказа: число строк начинают "
                         "читать как число наблюдений")

    # ── 2. отдельное свидетельство греет ────────────────────────
    def test_5_other_document_is_a_witness(self):
        """Подтверждением стала ДРУГАЯ НАЗВАННАЯ книга, а не другая фраза.

        ПЕРЕПИСАНО 08.11 вместе с законом lineage
        (tests/test_lineage_contract.py). Прежняя формулировка — «другая
        фраза про ту же связь есть подтверждение» — измерена и оказалась
        дырой: Жозефина предсказала, что четыре текста с одним
        первоначальным основанием дадут четыре подтверждения, и цепочка
        «статья -> пересказ модели -> перефраз» подняла support 1 -> 5 при
        ОДНОМ основании. Пересказ отличается от оригинала ВСЕГДА, значит
        различие текстов не спорный аргумент, а самый лёгкий способ надуть
        достоверность.

        Цена принята: второй прогон того же рассказа без имени файла больше
        НЕ греет. Греют два разных названных документа.
        """
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False,
                           document="рассказ-про-костёр.txt")
        before = observation_weight(next(iter(self._obs().values())))
        lr.learn_from_text("Дождь способен гасить пламя и огонь.",
                           source="text", allow_teacher=False,
                           document="учебник-физики-5")
        warmed = [k for k, v in self._obs().items()
                  if v.get("head") == "дождь" and v.get("tail") == "гасить"]
        self.assertTrue(warmed, "связь потерялась при втором факте")
        rec = self._obs()[warmed[0]]
        self.assertEqual(2, int(rec["support"]),
                         "два названных документа не стали двумя основаниями")
        self.assertGreater(observation_weight(rec), before,
                           "подтверждение не добавило достоверности")

    def test_5b_anonymous_second_sentence_no_longer_warms(self):
        """Две безымянные фразы — НЕ два основания: цена нового закона.

        Это ОБРАТНАЯ сторона теста 5 и она дорогая: раньше здесь support
        рос. Отличить независимый рассказ от пересказа той же статьи нам
        нечем, пока вызывающий не назвал документ, — поэтому молчание
        выбрано вместо уверенности, которую нечем перепроверить.
        """
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        lr.learn_from_text("Дождь способен гасить пламя и огонь.",
                           source="text", allow_teacher=False)
        rec = [v for v in self._obs().values()
               if v.get("head") == "дождь" and v.get("tail") == "гасить"]
        self.assertTrue(rec, "связь потерялась")
        self.assertEqual(1, int(rec[0]["support"]),
                         "безымянные тексты по-прежнему греют support — "
                         "дыра, названная Жозефиной, вернулась")

    def test_6_channel_names_alone_are_not_two_bases(self):
        """Два канала без документа — НЕ два основания. ПРАВИЛО ПЕРЕМЕНЕНО.

        Прежняя версия этого теста запирала обратное («совпадение двух
        каналов — самый сильный из доступных нам аргументов») и была права
        ровно в одной части: независимые каналы действительно независимы.
        Ошибка была в слове «канал»: НЕЗАВИСИМОСТЬ — свойство ДОКУМЕНТА, а
        не названия трубы. Замер: «wiki» и «text» с одним и тем же
        предложением могли быть переписаны с одной страницы, и отличить это
        нам было нечем; support при этом рос до 7 там, где оснований 3.

        Слово Хозяина остаётся отдельным основанием — он своё начало, его
        нельзя «переписать с той же статьи», и это закреплено ниже.
        """
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        lr.learn_from_text("Дождь гасит огонь.", source="wiki",
                           allow_teacher=False)
        rec = self._obs()["дождь|CAN_DO|гасить"]
        self.assertEqual(1, int(rec["support"]),
                         "два безымянных канала согрели связь: название "
                         "канала снова стало основанием")
        # ЖУРНАЛ у обучающего ведётся с ВТОРОГО свидетельства: первое лежит
        # верхним уровнем записи (source/evidence), и это его законное место
        # — иначе одна и та же строка хранилась бы дважды.
        self.assertEqual("text", rec.get("source"),
                         "первое свидетельство не осталось верхним уровнем")
        self.assertEqual(1, len(rec.get("provenance") or []),
                         "свидетельство выброшено из журнала: греть не "
                         "греет, но данные терять нельзя")
        self.assertEqual("wiki", (rec["provenance"][0] or {}).get("source"),
                         "отказавшее свидетельство не сохранено")

    def test_6b_two_named_documents_are_two_bases(self):
        """Независимость по-прежнему учится — когда её можно назвать."""
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False, document="рассказ-1")
        lr.learn_from_text("Дождь гасит огонь.", source="wiki",
                           allow_teacher=False, document="статья-дождь")
        rec = self._obs()["дождь|CAN_DO|гасить"]
        self.assertEqual(2, int(rec["support"]),
                         "два разных названных документа перестали быть "
                         "двумя основаниями — закон перемудрил")

    def test_6c_owner_word_is_its_own_basis(self):
        """Слово Хозяина — отдельное основание при том же тексте."""
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        lr.learn_from_text("Дождь гасит огонь.", source="user",
                           allow_teacher=False)
        rec = self._obs()["дождь|CAN_DO|гасить"]
        # ПОДДЕРЖКА = НАЗВАННЫЕ ОСНОВАНИЯ (смена закона 08.11, шаг Г).
        # Прежняя редакция ждала support=2. Слово Хозяина ЗАСЧИТАНО и
        # осталось засчитанным — но видно это по ВЕСУ, а не по числу
        # «видела N раз»: основание здесь ОДНО (owner), безымянная строка
        # source="text" сослаться на себя не может. Разойтись числам
        # означало бы то измеренное враньё, ради которого правка и делалась:
        # «видела 2 раза» при одном основании (experiments/
        # measure_support_vs_bases.py, случай A).
        self.assertEqual(1, int(rec["support"]))
        self.assertEqual(1, independent_bases(rec),
                         "слово Хозяина слито с чужим текстом: он своё "
                         "начало, а не чей-то пересказ")
        self.assertGreater(record_weight(rec), 100,
                           "слово Хозяина не дало веса: засчёт не виден "
                           "ни в одном числе")


class TestRuleLivesInOnePlace(unittest.TestCase):
    """Правило должно быть ОДНИМ: иначе второй писатель его обойдёт."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"entities": {}, "observations": {}, "unknown_words": {}},
                      fh, ensure_ascii=False)
        self.brain = Lipid2Brain(memory_dir=self.dir)
        self.brain.ask_user_enabled = False
        self.brain.use_llm_teacher = False
        self.brain.use_wiki_teacher = False

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _obs(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)["observations"]

    def test_7_brain_respects_the_same_rule(self):
        """Любопытство обязано греться тем же законом, что и обучающий.

        Ветка «греет» переписана 08.11 под закон lineage: раньше «другой
        раз» от того же канала поднимал support, теперь это пересказ без
        названия. Мозг обязан греться от НАЗВАННОГО документа (заголовок
        статьи доезжает в data["title"]), а не от смены формулировки.
        """
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"]},
                                       confidence=0.5, evidence="один рассказ",
                                       source="llm")
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"]},
                                       confidence=0.5, evidence="один рассказ",
                                       source="llm")
        rec = self._obs()["мяч|IS_A|игрушка"]
        self.assertEqual(1, int(rec["support"]),
                         "мозг греет связь пересказом — правило существует "
                         "только для обучающего")
        # ЖУРНАЛ СВИДЕТЕЛЬСТВ ведёт именно мозг (обучающий держит пару
        # source/evidence верхним уровнем), поэтому проверка «пересказ не
        # добавляет строку» обязана быть здесь, а не только в классе выше.
        # Без неё мутация «при пересказе дописать в журнал строку» выживала:
        # support целый, а число строк журнала кто-нибудь однажды прочитает
        # как число наблюдений.
        self.assertEqual(1, len(rec.get("provenance") or []),
                         "пересказ добавил строку в журнал: число строк "
                         "начинают читать как число наблюдений")
        # ПЕРВОЕ НАЗВАННОЕ основание. Первая запись выше пришла от модели
        # без документа — она установила связь (support 1), но сослаться на
        # неё нечем. Статья «Мяч» — первое основание, которое можно
        # назвать, и зачёт ей законен: неизвестное происхождение греет только как повтор, а
        # впервые названное происхождение — это новое основание.
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"],
                                              "title": "Мяч"},
                                       confidence=0.5, evidence="другая фраза",
                                       source="wiki")
        rec = self._obs()["мяч|IS_A|игрушка"]
        # ЗАКОН ИЗМЕНЁН ОТКРЫТО (08.11, шаг Г): зачёт остался (вес растёт),
        # support больше не «сколько раз засчитали», он равен названным
        # основаниям. Здесь оно первое — и молчание числа честно: второй
        # опоры у связи нет.
        self.assertEqual(1, int(rec["support"]),
                         "первое названное основание не стало основанием")
        self.assertEqual(1, independent_bases(rec))
        # ВОТ ГЛАВНАЯ ДЫРА: модель пересказывает СТАТЬЮ, которую мы сами
        # читали. Другой канал, другие слова, ТОТ ЖЕ документ.
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"],
                                              "title": "Мяч"},
                                       confidence=0.9,
                                       evidence="мяч — предмет для игр",
                                       source="llm")
        rec = self._obs()["мяч|IS_A|игрушка"]
        self.assertEqual(1, int(rec["support"]),
                         "та же статья, пересказанная другим каналом, согрела "
                         "связь: независимость снова измеряется словами")
        self.assertEqual(3, len(rec.get("provenance") or []),
                         "пересказ выброшен из журнала: отказывать в доверии "
                         "— не значит терять данные")
        # ДРУГОЙ названный документ — вот это третье основание.
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"],
                                              "title": "Список игрушек"},
                                       confidence=0.5, evidence="ещё один",
                                       source="wiki")
        rec = self._obs()["мяч|IS_A|игрушка"]
        # Число = НАЗВАННЫЕ ОСНОВАНИЯ (закон 08.11, шаг Г): «Мяч» + «Список
        # игрушек». Прежняя редакция ждала 3 — там считалось и безымянное
        # первое свидетельство, то есть «видела 3 раза» при двух опорах.
        self.assertEqual(2, int(rec["support"]),
                         "мозг перестал замечать ОТДЕЛЬНОЕ свидетельство — "
                         "опыт перестал чему-то учить")
        self.assertEqual(2, independent_bases(rec))

    def test_8_helper_refuses_same_source_and_text(self):
        """Само правило: пересказ и НЕИЗВЕСТНО ОТКУДА — не подтверждение.

        Вторая ветка переписана 08.11: раньше «новое основание» от канала
        без документа грело. Теперь такое свидетельство сохраняет журнал и
        молчит в числе — независимо́вость надо НАЗВАТЬ, а не только сменить
        формулировку.
        """
        rec = {"head": "а", "relation": "IS_A", "tail": "б", "confidence": 0.5,
               "support": 1, "weight": 100,
               "provenance": [{"source": "llm", "evidence": "тот же текст"}]}
        warmed = add_witness(rec, source="llm", evidence="тот же текст",
                             confidence=0.5, negated=False, relation="IS_A")
        self.assertFalse(warmed, "пересказ признан подтверждением")
        self.assertEqual(1, int(rec["support"]))
        warmed = add_witness(rec, source="llm", evidence="новое основание",
                             confidence=0.5, negated=False, relation="IS_A")
        self.assertFalse(warmed,
                         "свидетельство без происхождения подогрело support")
        self.assertEqual(1, int(rec["support"]))
        warmed = add_witness(rec, source="llm", evidence="новое основание",
                             confidence=0.5, negated=False, relation="IS_A",
                             document="глоссарий игрушек")
        self.assertTrue(warmed, "названный документ не принят как основание")
        # Основание стало ОДНИМ названным — поддержка обязана показать ровно
        # одно (смена закона 08.11; floor 1 при этом сохранён).
        self.assertEqual(1, int(rec["support"]))


if __name__ == "__main__":
    unittest.main()
