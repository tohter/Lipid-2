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

from core.records import add_witness, observation_weight
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
    def test_5_other_sentence_is_a_witness(self):
        """ДРУГАЯ фраза про ту же связь — подтверждение, и оно видно числом."""
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        before = observation_weight(next(iter(self._obs().values())))
        lr.learn_from_text("Дождь способен гасить пламя и огонь.", source="text",
                           allow_teacher=False)
        warmed = [k for k, v in self._obs().items()
                  if v.get("head") == "дождь" and v.get("tail") == "гасить"]
        self.assertTrue(warmed, "связь потерялась при втором факте")
        rec = self._obs()[warmed[0]]
        self.assertGreater(int(rec["support"]), 1,
                           "второе независимое свидетельство не стало "
                           "подтверждением")
        self.assertGreater(observation_weight(rec), before,
                           "подтверждение не добавило достоверности")

    def test_6_same_words_other_channel_is_a_witness(self):
        """Те же слова ИЗ ДРУГОГО КАНАЛА — отдельное свидетельство.

        Гасит ли это спор? Нет, и вот почему: Википедия и рассказ Хозяина
        говорят одними словами о разных основаниях, и совпадение двух
        независимых каналов — это и есть то, что повышает уверенность.
        """
        lr = self._learner()
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        lr.learn_from_text("Дождь гасит огонь.", source="wiki",
                           allow_teacher=False)
        rec = self._obs()["дождь|CAN_DO|гасить"]
        self.assertEqual(2, int(rec["support"]),
                         "совпадение двух каналов не засчитано — а это "
                         "самый сильный из доступных нам аргументов")


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
        """Любопытство обязано греться тем же законом, что и обучающий."""
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
        self.brain._record_observation("мяч", {"hierarchy": ["игрушка"]},
                                       confidence=0.5, evidence="другой раз",
                                       source="llm")
        rec = self._obs()["мяч|IS_A|игрушка"]
        self.assertEqual(2, int(rec["support"]),
                         "мозг перестал замечать ОТДЕЛЬНОЕ свидетельство — "
                         "опыт перестал чему-то учить")
        self.assertEqual(2, len(rec.get("provenance") or []),
                         "подтверждение не оставило следа в журнале")

    def test_8_helper_refuses_same_source_and_text(self):
        """Само правило: совпал и канал, и основание — подтверждения нет."""
        rec = {"head": "а", "relation": "IS_A", "tail": "б", "confidence": 0.5,
               "support": 1, "weight": 100,
               "provenance": [{"source": "llm", "evidence": "тот же текст"}]}
        warmed = add_witness(rec, source="llm", evidence="тот же текст",
                             confidence=0.5, negated=False, relation="IS_A")
        self.assertFalse(warmed, "пересказ признан подтверждением")
        self.assertEqual(1, int(rec["support"]))
        warmed = add_witness(rec, source="llm", evidence="новое основание",
                             confidence=0.5, negated=False, relation="IS_A")
        self.assertTrue(warmed, "отдельное свидетельство не принято")
        self.assertEqual(2, int(rec["support"]))


if __name__ == "__main__":
    unittest.main()
