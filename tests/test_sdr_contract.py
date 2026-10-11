"""
Контрактные тесты на слой SDR.

Два кодировщика, два набора ожиданий:
  LetterHashEncoder — преждняя схема (SHA-256 по буквенным n-граммам). Тесты
      ДОКУМЕНТИРУЮТ, что она не годится: она красная на смысле. Это не баг,
      это памятник и точка отсчёта.
  EmbeddingEncoder  — рабочий кодировщик. Красным быть не имеет права.

Запуск:
    python3 -m unittest discover -s tests -v
Внимание: эмбеддинговые тесты тянут SentenceTransformer, первые секунды долгие.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.sdr import SDR1024, prewarm, clear_cache
from core.encoder import (LetterHashEncoder, EmbeddingEncoder, get_encoder,
                          reset_encoder, TOTAL_BITS, TARGET_ON_BITS)

# Курируемые пары. Смысловые — только ВЫСОКОСХОДНЫЕ (косинус >= 0.95 у модели):
# иначе тест требует от MiniLM-L12 того, что не может сама модель. См.
# experiments/scheme_eval.py: у пары «холостой/неженатый» косинус 0.47, и никакой
# бинаризатор этого не вытащит — это ограничение модели, а не слоя SDR.
CLOSE_SYNONYMS = [("кошка", "кот"), ("грусть", "печаль"), ("храбрый", "смелый"),
                  ("огонь", "пламя"), ("врач", "доктор")]
MORPH_VARIANTS = [("кошка", "кошки"), ("кошка", "кошке"), ("грусть", "грустью"),
                  ("книга", "книги"), ("лес", "лесу")]
ORTHO_DISTRACTORS = [("кот", "код"), ("дом", "том")]      # буквы близки, смысл разный
UNRELATED = [("кошка", "табуретка"), ("стол", "река"), ("огонь", "книга"),
             ("дом", "алгебра"), ("любовь", "гвоздь")]


def _ov_pair(enc, a, b):
    m = enc.encode_many([a, b])
    return int((m[0] & m[1]).sum())


class TestStructuralInvariants(unittest.TestCase):
    """Инварианты представления. Обязаны держаться на ЛЮБОМ кодировщике."""

    @classmethod
    def setUpClass(cls):
        cls.enc = get_encoder()

    def _mask(self, w):
        return self.enc.encode(w)

    def test_exactly_target_density(self):
        for w in ("кот", "кошка", "а", "сельскохозяйственно-машиностроительный"):
            self.assertEqual(int(self._mask(w).sum()), TARGET_ON_BITS,
                             f"«{w}» — разреженность поехала")

    def test_deterministic(self):
        self.assertEqual(list(self._mask("кошка")), list(self._mask("кошка")),
                         "кодировщик недетерминирован — кэш графа и .lipid рассинхронятся")

    def test_deterministic_across_processes(self):
        """
        Внутрипроцессный детерминизм тривиален: массивы лежат в одной памяти.
        Опасность в ДРУГОМ процессе — тот же код, но новый import, новая
        калибровка, новый RandomState. Если бы отпечаток разъехался, граф,
        сохранённый сегодня, завтра перестал бы узнавать СОБСТВЕННЫЕ узлы — и
        выглядело бы это как «память деградировала», а не как смена кодировщика.
        """
        words = ["флюгер", "кошка", "шизокарпий", "табуретка"]
        want = "|".join(",".join(map(str, self._mask(w).nonzero()[0]))
                        for w in words)

        probe = ("import warnings; warnings.filterwarnings('ignore'); "
                 f"import sys; sys.path.insert(0, {ROOT!r}); "
                 "from core.sdr import SDR1024; "
                 "print('|'.join(','.join(map(str, SDR1024(w).active_indices()))"
                 f" for w in {words!r}))")
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, timeout=900, cwd=ROOT)
        self.assertEqual(out.returncode, 0, f"дочерний процесс упал: {out.stderr[-400:]}")
        got = out.stdout.strip().splitlines()[-1]
        self.assertEqual(got, want,
                         "SDR зависит от процесса: сохранённый граф перестанет "
                         "узнавать свои узлы после перезапуска")

    def test_case_and_space_insensitive(self):
        self.assertEqual(list(self._mask("  КоШка ")), list(self._mask("кошка")))

    def test_bits_inside_1024(self):
        m = self._mask("абракадабра")
        self.assertEqual(m.shape[0], TOTAL_BITS)

    def test_empty_text_is_all_zero(self):
        self.assertEqual(int(self._mask("   ").sum()), 0)


class TestSdrApi(unittest.TestCase):
    """Публичный API SDR1024 не поехал относительно исторического."""

    def test_self_overlap_full(self):
        s = SDR1024("кошка")
        self.assertEqual(s.overlap(s), TARGET_ON_BITS)
        self.assertAlmostEqual(s.similarity(s), 1.0)

    def test_density_and_symmetry(self):
        a, b = SDR1024("кошка"), SDR1024("кот")
        self.assertEqual(a.active_bit_count(), TARGET_ON_BITS)
        self.assertEqual(a.overlap(b), b.overlap(a))

    def test_empty_vs_word(self):
        self.assertEqual(SDR1024("кот").overlap(SDR1024("")), 0)

    def test_caching_is_consistent(self):
        clear_cache()
        a, b = SDR1024("табуретка"), SDR1024("табуретка")
        self.assertEqual(a.bits, b.bits)

    def test_u64_roundtrip(self):
        """active_indices <-> bits: 16 слов uint64 должны разворачиваться обратно."""
        s = SDR1024("кошка")
        self.assertEqual(len(s.active_indices()), TARGET_ON_BITS)
        self.assertTrue(all(0 <= i < TOTAL_BITS for i in s.active_indices()))

    def test_prewarm_matches_single(self):
        clear_cache()
        prewarm(["шизокарпий", "гипотенуза"])
        single = SDR1024("шизокарпий").bits
        clear_cache()
        batch = SDR1024("шизокарпий").bits
        self.assertEqual(single, batch, "пачечный прогрев даёт другие биты, чем одиночное кодирование")


class TestSemanticGrounding(unittest.TestCase):
    """Главное: SDR кодирует СМЫСЛ, а не написание."""

    @classmethod
    def setUpClass(cls):
        cls.enc = get_encoder()

    def test_skips_on_letterhash(self):
        if not self.enc.semantic:
            self.skipTest("буквенный хэш осознанно не семантичен")

    def test_unrelated_near_zero(self):
        for a, b in UNRELATED:
            ov = _ov_pair(self.enc, a, b)
            self.assertLessEqual(ov, 4, f"«{a}»~«{b}» дало {ov}/20 — это шум")

    def test_close_synonyms_far_above_noise(self):
        for a, b in CLOSE_SYNONYMS:
            ov = _ov_pair(self.enc, a, b)
            self.assertGreaterEqual(ov, 12, f"«{a}»~«{b}» лишь {ov}/20")
        for a, b in UNRELATED:
            self.assertLess(_ov_pair(self.enc, a, b), 6)

    def test_morphological_variants_recognizable(self):
        """Форма того же слова — главный случай узнавания в предиктивном цикле."""
        for a, b in MORPH_VARIANTS:
            ov = _ov_pair(self.enc, a, b)
            self.assertGreaterEqual(ov, 10, f"форма «{a}»/«{b}» лишь {ov}/20")

    def test_meaning_beats_spelling(self):
        """Тот самый тест, на котором буквенный хэш умирал."""
        syn = sum(_ov_pair(self.enc, a, b) for a, b in CLOSE_SYNONYMS) / len(CLOSE_SYNONYMS)
        orth = sum(_ov_pair(self.enc, a, b) for a, b in ORTHO_DISTRACTORS) / len(ORTHO_DISTRACTORS)
        self.assertGreater(syn, orth + 5,
                           f"смысл не кодируется: синонимы {syn:.1f} против "
                           f"орфо-двойников {orth:.1f}")

    def test_overlap_monotone_in_cosine(self):
        """
        Перекрытие обязано быть монотонной функцией близости эмбеддинга —
        иначе «overlap = семантическое узнавание» снова ложь.
        """
        import numpy as np
        from scipy.stats import spearmanr
        pairs = CLOSE_SYNONYMS + MORPH_VARIANTS + [("холостой", "неженатый"),
                                                   ("быстрый", "скорый")] + UNRELATED
        words = list(dict.fromkeys([w for p in pairs for w in p]))
        self.enc.encode_many(words)
        raw = self.enc._raw_encode(words)
        ov, cs = [], []
        for a, b in pairs:
            i, j = words.index(a), words.index(b)
            ov.append(_ov_pair(self.enc, a, b))
            cs.append(float(raw[i] @ raw[j]))
        rho = spearmanr(ov, cs).statistic
        self.assertGreater(rho, 0.75,
                           f"overlap почти не связан с косинусом (rho={rho:.2f})")


class TestScaleSafety(unittest.TestCase):
    """
    Главный вопрос: СУЩЕСТВУЕТ ЛИ порог, отделяющий «это уже есть в графе» от
    «этого нет», и живёт ли он при росте словаря.

    Методика (и главная ловушка): «неизвестное» слово нельзя брать из того же
    словаря, что и граф. Русское слово из hunspell почти всегда имеет в том же
    словаре настоящего двойника — и тогда его best-overlap = 19-20, что выглядит
    как 20-30% ложных узнаваний, хотя ошибки нет: слово действительно узналось
    Correctно — слова, которых в словаре НЕТ (исковерканные/редкие/выдуманные).

    Итог замера на 15 000 настоящих слов (experiments/noise_floor.py):
        размер графа   пол(медиана)   best у ОТСУТСТВУЮЩЕГО слова
               500           10                  8
             2 000            9                 11
             8 000           10                 12
            15 000           10                 12
    Пол НЕ растёт со словарём → фиксированный порог живёт. У буквенного хэша
    пол рос лавинообразно, поэтому UNKNOWN_THRESHOLD=2 молча обесценивался.
    """

    N = 15000
    # Которых нет в hunspell ru_RU — настоящие «неизвестные»
    OUT_OF_VOCAB = ["шизокарпий", "гипотенуза", "квантор", "реквизит", "флюгер",
                    "балласт", "адажио", "иероглиф", "пломбир", "скальп", "фиаско",
                    "юниор", "гексагон", "трюмсель", "куматронь", "велофасад",
                    "орнитоптер", "ксілофон", "квастум", "пневмокот", "шаромыжник",
                    "фибриллярный"]
    # Форма слова, которое в графе ЕСТЬ → обязано узнаваться
    FORMS = [("кошка", "кошки"), ("кошка", "кошке"), ("грусть", "грустью"),
             ("книга", "книги"), ("лес", "лесу"), ("кот", "кота"), ("стол", "столы"),
             ("дом", "доме"), ("друг", "друзья"), ("город", "города"),
             ("река", "реке"), ("огонь", "огнем"), ("вода", "воды"), ("свет", "света"),
             ("ночь", "ночью"), ("камень", "камни"), ("окно", "окна")]
    THRESHOLD = 13        # = Cortex.UNKNOWN_CEILING

    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.enc = get_encoder()
        if not os.path.exists("experiments/ru_words.txt"):
            raise unittest.SkipTest("нет словаря experiments/ru_words.txt")
        every = [w.strip() for w in open("experiments/ru_words.txt", encoding="utf-8")
                 if 3 <= len(w.strip()) <= 10]
        cls.allwords = set(every)
        # Основы форм ОБЯЗАНЫ попасть в «граф», иначе тест про формы бессмысленен.
        bases = [a for a, _ in cls.FORMS if a in cls.allwords]
        pool, pref = list(bases), {a[:5] for a in bases}
        for w in every:                      # дальше — разные корни, без дублей
            if w in pool or w[:5] in pref or any(w[:4] == b[:4] for b in bases):
                continue
            pref.add(w[:5])
            pool.append(w)
            if len(pool) >= cls.N:
                break
        cls.pool = pool
        cls.mask = cls.enc.encode_many(pool)
        cls.rows, cls.coll = np.nonzero(cls.mask)
        cls._csr_cache = None

    def _sparse(self):
        import numpy as np, scipy.sparse as sp
        if self.__class__._csr_cache is None:
            self.__class__._csr_cache = sp.csr_matrix(
                (np.ones(self.coll.size, np.uint8), (self.rows, self.coll)),
                shape=(len(self.pool), TOTAL_BITS))
        return self.__class__._csr_cache

    def test_threshold_zero_false_positives(self):
        """Неизвестное слово НЕ должно «узнаваться» на рабочем пороге."""
        import numpy as np
        if not self.enc.semantic:
            self.skipTest("у буквенного хэша разделяющего порога нет по построению")
        unk = [w for w in self.OUT_OF_VOCAB if w not in self.allwords]
        self.assertGreaterEqual(len(unk), 8, "слишком мало вне-словарных проб")
        U = self.enc.encode_many(unk).astype(np.int32)
        best = np.asarray(self._sparse() @ U.T).max(1)
        fp = float((best >= self.THRESHOLD).mean())
        self.assertLessEqual(fp, 0.05,
                             f"{fp*100:.0f}% неизвестных слов ложно «узнались» "
                             f"(best={sorted(map(int, best))}, порог={self.THRESHOLD})")

    def test_word_forms_are_recognized(self):
        """Форма слова, чья основа в графе, обязана перешагнуть порог."""
        import numpy as np
        if not self.enc.semantic:
            self.skipTest("буквенный хэш не узнаёт формы")
        gi = {w: i for i, w in enumerate(self.pool)}
        S = self._sparse()
        pairs = [(a, b) for a, b in self.FORMS if a in gi]
        self.assertGreaterEqual(len(pairs), 8, "мало форм в тестовом пуле")
        misses = []
        for a, b in pairs:
            ov = int(np.asarray(S[gi[a]] @ self.enc.encode(b).astype(np.int32)).ravel()[0])
            if ov < self.THRESHOLD:
                misses.append(f"{a}<-{b}:{ov}")
        # Единичные промахи допустимы: морфологию бинаризация не гарантирует.
        self.assertLessEqual(len(misses), max(2, len(pairs) // 4),
                             f"формы слов не узнались: {misses}")

    def test_floor_does_not_grow_with_graph_size(self):
        """
        Пол шума не должен расти со словарём. Именно из-за его роста
        фиксированная константа в конфиге раньше молча устаревала.
        """
        import numpy as np
        if not self.enc.semantic:
            self.skipTest("у буквенного хэша пол растёт — это и был баг")
        unk = [w for w in self.OUT_OF_VOCAB if w not in self.allwords]
        U = self.enc.encode_many(unk).astype(np.int32)
        S = self._sparse()
        res = {}
        for n in (500, 4000, len(self.pool)):
            res[n] = float(np.median(np.asarray(S[:n] @ U.T).max(1)))
        self.assertLessEqual(res[max(res)] - res[min(res)], 5,
                             f"пол пополз со словарём: {res}")

    def test_letterhash_cannot_separate(self):
        """
        Памятник. На буквенном хэше «несвязанные» слова дают overlap НЕ меньше,
        чем синонимы, поэтому никаким порогом качество не вытаскивалось:
        кошка~кот = 3, кошка~табуретка = 4.
        """
        h = LetterHashEncoder()
        syn = sum(_ov_pair(h, a, b) for a, b in CLOSE_SYNONYMS) / len(CLOSE_SYNONYMS)
        uns = sum(_ov_pair(h, a, b) for a, b in UNRELATED) / len(UNRELATED)
        self.assertLessEqual(syn, uns + 1,
                             f"буквенный хэш вдруг стал семантичным ({syn} vs {uns})?")


