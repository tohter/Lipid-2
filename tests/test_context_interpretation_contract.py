"""
ДЫРА 7 шаг 2 (КОНТЕКСТ И ИНТЕРПРЕТАЦИЯ, план 0.4): запись несёт, ГДЕ и В
КАКОМ СМЫСЛЕ существует утверждение.

Зачем. Хозяин решил (0.4, Денди-философия): «метла — инвентарь для уборки»
и «метла — транспорт ведьмы» ОБЕ имеют право жить — одна научная, другая
мифологическая. Чтобы движок воспринимал обе в разной степени, запись
должна помнить:
  context        — мир существования: реальный мир / сказка / гипотеза / игра;
  interpretation — смысл употребления: буквально / метафора / фантазия.
Жозефина (ADV_Q8): это РАЗНЫЕ оси — нельзя сливать в одну бирку. «Транспорт
ведьмы» истинно в сказке, но не в реальном мире; «глаз не оторвать» —
метафора, но не буквальное действие.

Что здесь заперено:
  1) нормализация context/interpretation к закрытому множеству значений;
  2) запись хранит их, когда источник дал (data.get), иначе дефолт;
  3) у «метлы» среда различит научную и мифологическую записи по контексту;
  4) отчётливое сравнение: context и interpretation независимы (одно значение
     в одном поле не вытаскивает чужое из другого).
"""
import unittest
import tempfile
import json

from core.records import norm_context, norm_interpretation
from brain.lipid2_brain import Lipid2Brain


class TestNormFunctions(unittest.TestCase):
    """Нормализация к закрытому множеству значений."""

    def test_context_known_values(self):
        self.assertEqual("real", norm_context("real"))
        self.assertEqual("real", norm_context("real"))
        self.assertEqual("story", norm_context("story"))
        self.assertEqual("story", norm_context("story"))
        self.assertEqual("hypothesis", norm_context("hypothesis"))
        self.assertEqual("hypothesis", norm_context("hypothesis"))
        self.assertEqual("game", norm_context("game"))
        self.assertEqual("game", norm_context("game"))

    def test_context_default_and_unknown(self):
        # Дефолт для обычного текста — реальный мир; неизвестное тоже туда.
        self.assertEqual("real", norm_context(None))
        self.assertEqual("real", norm_context(""))
        self.assertEqual("real", norm_context("unknown_context"))

    def test_interpretation_known_values(self):
        self.assertEqual("literal", norm_interpretation("literal"))
        self.assertEqual("literal", norm_interpretation("literal"))
        self.assertEqual("metaphor", norm_interpretation("metaphor"))
        self.assertEqual("metaphor", norm_interpretation("metaphor"))
        self.assertEqual("fantasy", norm_interpretation("fantasy"))
        self.assertEqual("fantasy", norm_interpretation("fantasy"))

    def test_interpretation_default_and_unknown(self):
        self.assertEqual("literal", norm_interpretation(None))
        self.assertEqual("literal", norm_interpretation("unknown_interpretation"))


class TestWrittenObservationHasContext(unittest.TestCase):
    """Запись хранит context/interpretation, когда источник дал."""

    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="l2_ctx_")
        self.b = Lipid2Brain(memory_dir=tmp)
        self.b.use_llm_teacher = False
        self.b.use_wiki_teacher = False
        self.b.ask_user_enabled = False

    def _obs(self, head, data):
        self.b._integrate_entity_into_core(
            head, data,
            confidence=0.9, evidence="ev", source="user")
        with open(self.b.world_filepath, encoding="utf-8") as fh:
            world = json.load(fh)
        return list(world["observations"].values())

    def test_metadata_written_when_given(self):
        # Источник дал контекст и интерпретацию — они доехали до записи.
        obs = self._obs("метла", {
            "hierarchy": ["инвентарь"],
            "context": "реальный мир",
            "interpretation": "буквально",
        })
        self.assertTrue(obs)
        self.assertEqual("real", obs[0]["context"])
        self.assertEqual("literal", obs[0]["interpretation"])

    def test_defaults_when_absent(self):
        # Источник не дал — дефолт: реальный мир, буквально.
        obs = self._obs("ложка", {"hierarchy": ["столовый прибор"]})
        self.assertTrue(obs)
        self.assertEqual("real", obs[0]["context"])
        self.assertEqual("literal", obs[0]["interpretation"])

    def test_story_context_distinguishes_myth(self):
        # Две записи про метлу различимы по контексту: научная и миф.
        b = self.b
        b._integrate_entity_into_core(
            "метла", {"hierarchy": ["инвентарь"]},
            confidence=0.9, evidence="подметать", source="user",
            context="real", interpretation="literal")
        b._integrate_entity_into_core(
            "метла", {"hierarchy": ["транспорт"]},
            confidence=0.6, evidence="сказка про ведьму", source="text",
            context="story", interpretation="fantasy")
        with open(b.world_filepath, encoding="utf-8") as fh:
            world = json.load(fh)
        obs = list(world["observations"].values())
        sci = [o for o in obs if o["context"] == "real"]
        myth = [o for o in obs if o["context"] == "story"]
        self.assertTrue(sci, "научная запись обязана остаться")
        self.assertTrue(myth, "мифологическая запись обязана остаться")
        self.assertEqual("literal", sci[0]["interpretation"])
        self.assertEqual("fantasy", myth[0]["interpretation"])
        # Обе живы — ничего не выброшено (вариант Б хозяина).
        self.assertEqual(2, len(obs))

    def test_context_and_interpretation_are_independent(self):
        # Значение одного поля не вытаскивает чужое из другого.
        b = self.b
        b._integrate_entity_into_core(
            "глаз", {"hierarchy": ["орган"]},
            confidence=0.9, evidence="не могла оторвать глаз", source="text",
            context="real", interpretation="metaphor")
        with open(b.world_filepath, encoding="utf-8") as fh:
            world = json.load(fh)
        obs = list(world["observations"].values())
        self.assertTrue(obs)
        self.assertEqual("real", obs[0]["context"])
        self.assertEqual("metaphor", obs[0]["interpretation"],
                         "метафора в реальном мире: реальность мира и смысл "
                         "употребления — разные оси")


if __name__ == "__main__":
    unittest.main()