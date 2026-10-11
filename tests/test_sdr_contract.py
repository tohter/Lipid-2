"""
Контрактные тесты на слой SDR. НЕ зависят от конкретного кодировщика.

Смысл: перед заменой энкодера (буквенный-хэш -> эмбеддинговый) фиксируем
ИНВАРИАНТЫ, которые обязан соблюдать любой кодировщик, и заодно ловим
текущую регрессию (буквенный хэш кодирует написание, а не смысл).

Запуск:  python3 -m unittest discover -s tests -v      # без внешних зависимостей
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sdr import SDR1024

# Синонимическая пара (тот же смысл) и орфографическая пара (тот же корень букв,
# другой смысл). Хороший кодировщик обязан различать эти отношения.
SEMANTIC_PAIRS = [
    ("кошка", "кот"),
    ("грусть", "печаль"),
    ("бег", "беготня"),
    ("дом", "жилище"),
]
ORTHOGRAPHIC_DISTRACTORS = [
    ("кот", "код"),
    ("дом", "том"),
    ("лес", "леск"),
]
UNRELATED_PAIRS = [
    ("кошка", "табуретка"),
    ("грусть", "табуретка"),
    ("любовь", "гвоздь"),
]


class TestStructuralInvariants(unittest.TestCase):
    """Инварианты представления — должны выполняться для ЛЮБОГО кодировщика."""

    def test_exactly_target_density(self):
        for w in ("кот", "кошка", "а", "сельскохозяйственно-машиностроительный", "   "):
            s = SDR1024(w)
            if w.strip():
                self.assertEqual(s.active_bit_count(), SDR1024.TARGET_ON_BITS,
                                 f"«{w}» дал {s.active_bit_count()} активных бит")

    def test_deterministic(self):
        a, b = SDR1024("кошка"), SDR1024("кошка")
        self.assertEqual(a.bits, b.bits, "Кодировщик недетерминирован — кэш графа сломан")

    def test_case_and_space_insensitive(self):
        self.assertEqual(SDR1024("  КоШка ").bits, SDR1024("кошка").bits)

    def test_self_similarity_is_one(self):
        s = SDR1024("кошка")
        self.assertEqual(s.overlap(s), SDR1024.TARGET_ON_BITS)
        self.assertAlmostEqual(s.similarity(s), 1.0)

    def test_empty_text_is_all_zero(self):
        self.assertEqual(SDR1024("   ").active_bit_count(), 0)
        self.assertEqual(SDR1024("кот").overlap(SDR1024("")), 0)

    def test_bits_fit_in_1024(self):
        for w in ("кот", "кошка", "абракадабра"):
            s = SDR1024(w)
            self.assertEqual(len(s.bits), SDR1024.NUM_U64)
            for blk in s.bits:
                self.assertGreaterEqual(blk, 0)
                self.assertLess(blk, 1 << 64)

    def test_symmetry(self):
        a, b = SDR1024("кошка"), SDR1024("кот")
        self.assertEqual(a.overlap(b), b.overlap(a))


class TestSemanticGrounding(unittest.TestCase):
    """Смысловые требования. Это и есть то, что ломает буквенный хэш."""

    def test_unrelated_pairs_are_near_zero(self):
        for a, b in UNRELATED_PAIRS:
            ov = SDR1024(a).overlap(SDR1024(b))
            self.assertLessEqual(ov, 2, f"«{a}»~«{b}» дало шум {ov}/20")

    def test_synonyms_beat_orthography(self):
        """Главный тест: синоним обязан быть ближе, чем орфографический двойник."""
        syn = [SDR1024(a).overlap(SDR1024(b)) for a, b in SEMANTIC_PAIRS]
        orth = [SDR1024(a).overlap(SDR1024(b)) for a, b in ORTHOGRAPHIC_DISTRACTORS]
        mean_syn, mean_orth = sum(syn) / len(syn), sum(orth) / len(orth)
        self.assertGreater(
            mean_syn, mean_orth,
            f"Смысл не кодируется: синонимы {mean_syn:.2f}/20 <= "
            f"орфографические двойники {mean_orth:.2f}/20 (буквенный хэш)",
        )

    def test_synonym_margin_over_threshold(self):
        """Синоним должен уверенно перешагивать UNKNOWN_THRESHOLD=2 ядра."""
        for a, b in SEMANTIC_PAIRS:
            ov = SDR1024(a).overlap(SDR1024(b))
            self.assertGreaterEqual(ov, 6, f"«{a}»~«{b}» лишь {ov}/20 — порог 2 не различим")


class TestScaleSafety(unittest.TestCase):
    """
    Здоровье фиксированного порога при росте словаря.
    Даже на ИДЕАЛЬНО случайных SDR max-of-N растёт с N => абсолютный порог
    UNKNOWN_THRESHOLD обязан стать относительным. Тест документирует потолок.
    """

    def test_false_positive_budget(self):
        import random
        random.seed(11)
        def rand_sdr():
            return frozenset(random.sample(range(SDR1024.TOTAL_BITS), SDR1024.TARGET_ON_BITS))
        N = 20000
        vocab = [rand_sdr() for _ in range(N // 4)]
        vocab += [rand_sdr() for _ in range(N - len(vocab))]
        queries = [rand_sdr() for _ in range(40)]
        fp = [sum(1 for v in vocab if len(q & v) >= 3) for q in queries]
        median_fp = sorted(fp)[len(fp) // 2]
        self.assertLess(median_fp, 20,
                        f"При N={N} медиана ложных соседей с overlap>=3 = {median_fp}. "
                        "Фиксированный порог не масштабируется — нужен относительный.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
