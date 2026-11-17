"""
ДЫРА 7 (БАЙТ ДОВЕРИЯ, план 0.4 шаг 1): «уверенность» обязана быть ЦЕЛЫМ
байтом 0..255, а не float-долей 0..1.

Зачем. Хозяин строит Вселенную на восьмибитной философии (как старая игровая
консоль). Float 0..1 тянет за собой дробную арифметику и мысленный
«сопроцессор»; целый байт 0..255 эргономичнее и согласован с тем, что вес
связи (weight) уже хранится байтом. Значит и «уверенность» (confidence) в
хранимых наблюдениях обязана быть тем же байтом 0..255, одним и тем же.

Правило-мост (core/records.py::byte_confidence):
  число <= 1.0  -> legacy-доля, умножить на 255 (0.9 -> 230, 0.5 -> 128);
  число >  1.0  -> уже байт 0..255, оставить (230 -> 230).
Так старое (float 0..1) и новое (байт) живут на одной шкале без сопроцессора.

Что здесь заперено:
  1) byte_confidence переводит долю и байт в ОДИН и тот же байт;
  2) запись через _record_observation хранит confidence ЦЕЛЫМ байтом;
  3) канал-источник даёт байт (user=255, wiki=216, llm=191, text=153);
  4) веса НЕ поехали: _integrate_entity_into_core и fact_weight дают те же
     числа, что давали на долях (конверсия в долю внутри, без смены масштаба).
"""
import unittest

from core.records import byte_confidence, fact_weight

# Канонические байты каналов (бывшие доли 1.0/0.85/0.75/0.6; round!):
# 1.0->255, 0.85->round(216.75)=217, 0.75->round(191.25)=191, 0.6->round(153.0)=153.
CANON = {"user": 255, "wiki": 217, "llm": 191, "text": 153}


class TestByteConfidenceConversion(unittest.TestCase):
    """Мост доля<->байт даёт ОДНУ шкалу."""

    def test_byte_confidence_keeps_byte(self):
        # Уже байт -> остаётся байтом (не режется до 1.0).
        self.assertEqual(255, byte_confidence(255))
        self.assertEqual(230, byte_confidence(230))
        self.assertEqual(0, byte_confidence(0))

    def test_byte_confidence_raises_legacy_fraction(self):
        # Доля <= 1.0 -> байт 0..255.
        self.assertEqual(255, byte_confidence(1.0))
        self.assertEqual(230, byte_confidence(0.9))
        self.assertEqual(128, byte_confidence(0.5))
        self.assertEqual(153, byte_confidence(0.6))
        self.assertEqual(0, byte_confidence(0.0))

    def test_byte_confidence_clamps_out_of_range(self):
        # Вне 0..255 -> к границе, без вранья в отрицательную сторону.
        self.assertEqual(255, byte_confidence(999))
        self.assertEqual(0, byte_confidence(-5))

    def test_byte_confidence_handles_strings_and_none(self):
        self.assertEqual(230, byte_confidence("0.9"))
        self.assertEqual(128, byte_confidence("128"))
        self.assertEqual(0, byte_confidence(None))
        self.assertEqual(0, byte_confidence("не число"))


class TestChannelBytes(unittest.TestCase):
    """Каналы-источники дают байт, а не долю."""

    def test_canonical_channel_bytes(self):
        # Канальные доли 1.0/0.85/0.75/0.6 -> байты 255/217/191/153 (round).
        for name, byte in CANON.items():
            frac = {"user": 1.0, "wiki": 0.85, "llm": 0.75, "text": 0.6}[name]
            self.assertEqual(byte, byte_confidence(frac),
                             f"{name}: {frac} -> {byte}")

    def test_channel_table_is_bytes(self):
        # В CHANNEL_CONFIDENCE сами каналы уже байты (не доли).
        from brain.lipid2_brain import Lipid2Brain
        for name, val in Lipid2Brain.CHANNEL_CONFIDENCE.items():
            self.assertEqual(val, byte_confidence(val),
                             f"{name}={val} обязан быть готовым байтом")
            self.assertIsInstance(val, int)


class TestWrittenObservationIsByte(unittest.TestCase):
    """Записанное наблюдение хранит confidence ЦЕЛЫМ байтом."""

    def setUp(self):
        import tempfile
        from brain.lipid2_brain import Lipid2Brain
        tmp = tempfile.mkdtemp(prefix="l2_byte_")
        self.b = Lipid2Brain(memory_dir=tmp)
        # Каналы отключены как АТРИБУТЫ экземпляра (сигнатура __init__ их не
        # принимает), чтобы тест не дёргал сеть/учителя.
        self.b.use_llm_teacher = False
        self.b.use_wiki_teacher = False
        self.b.ask_user_enabled = False

    def test_record_observation_stores_int_byte(self):
        # Пишем через _integrate_entity_into_core с каналом wiki.
        self.b._integrate_entity_into_core(
            "кошка", {"hierarchy": ["животное"], "properties": ["усатая"]},
            confidence=0.85, evidence="Кошки — домашние животные.",
            source="teacher:wiki")
        # Прочитать наблюдение из файла и убедиться, что confidence — целый.
        import json, os
        wf = self.b.world_filepath
        with open(wf, encoding="utf-8") as fh:
            world = json.load(fh)
        obs = world["observations"]
        found = [o for o in obs.values()
                 if o.get("head") == "кошка" and o.get("relation") == "IS_A"]
        self.assertTrue(found, "должно появиться наблюдение кошка IS_A животное")
        conf = found[0].get("confidence")
        self.assertEqual(217, conf, "0.85 (wiki) -> байт 217 (round 216.75)")
        self.assertIsInstance(conf, int)

    def test_weights_did_not_drift(self):
        # Вес не изменился от перевода confidence в байт: wiki IS_A по-прежнему
        # 100 + 100*0.85 = 185 (формула _integrate_entity_into_core).
        from core.graph import RelationType
        self.b._integrate_entity_into_core(
            "собака", {"hierarchy": ["животное"]},
            confidence=0.85, evidence="Собака — животное.", source="teacher:wiki")
        node = self.b.core.get_or_create_node("собака")
        edges = node.out_synapses_typed.raw_edges()
        isa = [e for e in edges
               if e["rel_type"].name == "IS_A" and e["target"] == "животное"]
        self.assertTrue(isa)
        # 100 + 100*0.85 = 185. Конверсия байта 216 -> доля 216/255 не должна
        # изменить вес: расчёт ведётся от исходной доли (0.85), а не от байта.
        self.assertEqual(185, isa[0]["weight"])


class TestFactWeightStillFractionBased(unittest.TestCase):
    """fact_weight остаётся на долях (внутренняя формула), байт конвертится."""

    def test_fact_weight_matches_legacy(self):
        # Те же веса, что давала доля: формула НЕ пересчитана в байтах.
        # CAN_DO: 90 + 130*0.9 = 207; прочие: 60 + 100*0.9 = 150.
        self.assertEqual(207, fact_weight(0.9, relation="CAN_DO", support=1))
        self.assertEqual(150, fact_weight(0.9, relation="IS_A", support=1))


if __name__ == "__main__":
    unittest.main()