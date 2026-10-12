"""
Контракт побитового кэша SDR (core/bitsstore.py) и еговязки с кодировщиком.

ЗАЩИЩАЕТ ТРИ ВЕЩИ, ДОРОГИЕ ДЛЯ ПРОЕКТА

  1. ФОРМАТ = КОНТРАКТ, а не побочный продукт сериализатора. Переносимая часть
     ядра обязана читаться без Python: фиксированные ширины, little-endian,
     длина перед строкой, 128-байтовая запись бит. Тест сверяет Python-упаковку
     с ЭТАЛОНОМ, посчитанным отдельно (struct + побитовый И), а не с самой собой.
  2. Биты вне своего кодировщика бессмысленны: несовпадение cfg-хэша обязано
     ОТКАЗЫВАТЬ ГРОМКО (store.reason), а не молча подставлять чужие би́ты.
  3. Кэш не имеет права менять РЕЗУЛЬТАТ. Би́ты из кэша обязаны совпасть с
     битми, посчитанными моделью, до бита — иначе кэш стал источником знания,
     а не ускорителем, и порог 13 повиснет на нём молча.

Формат и детерминизм проверяются БЕЗ torch на индексах-константах; блок с
моделью помечен пропуском, если .venv нет (и тогда он КРИЧИТ пропуском, а не
принимается «автоматически зелёным»).
"""
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.bitsstore import (BITS_BYTES, BitsStore, BitsStoreError, MAGIC,
                            overlap_bytes, pack_bits, unpack_bits)


class TestBitFormat(unittest.TestCase):
    """Упаковка/распаковка и overlap — на целых числах, без модели."""

    def test_pack_roundtrip(self):
        for bits in ([0], [1023], [0, 511, 1023], list(range(0, 1024, 5))):
            self.assertEqual(unpack_bits(pack_bits(bits)), sorted(bits),
                             f"roundtrip поехал на {bits[:5]}...")

    def test_record_is_exactly_128_bytes_little_endian(self):
        blob = pack_bits([0])
        self.assertEqual(len(blob), 128)
        # бит 0 — это младший бит ПЕРВОГО слова little-endian
        self.assertEqual(struct.unpack("<Q", blob[:8])[0], 1)
        self.assertEqual(blob[8:], b"\x00" * 120)
        # бит 64 — первый бит ВТОРОГО слова, не нулевого
        blob2 = pack_bits([64])
        self.assertEqual(struct.unpack("<Q", blob2[:8])[0], 0)
        self.assertEqual(struct.unpack("<Q", blob2[8:16])[0], 1)

    def test_overlap_matches_popcount_reference(self):
        a, b = pack_bits([1, 2, 3, 70]), pack_bits([2, 3, 70, 999])
        self.assertEqual(overlap_bytes(a, b), 3)
        self.assertEqual(overlap_bytes(a, a), 4)
        self.assertEqual(overlap_bytes(a, pack_bits([1023])), 0)
        # совпадение с определением «через множества»
        for bits_a, bits_b in (([5] * 1, [5, 6]), (list(range(20)), list(range(10, 30)))):
            ref = len(set(bits_a) & set(bits_b))
            self.assertEqual(overlap_bytes(pack_bits(bits_a), pack_bits(bits_b)), ref)

    def test_rejects_out_of_range_bit(self):
        for bad in (-1, 1024, 5000):
            with self.assertRaises(BitsStoreError):
                pack_bits([bad])

    def test_rejects_short_record(self):
        with self.assertRaises(BitsStoreError):
            unpack_bits(b"\x00" * 127)
        with self.assertRaises(BitsStoreError):
            overlap_bytes(b"\x00" * 8, b"\x00" * 128)


class TestBitsStoreFile(unittest.TestCase):
    """Файл: заголовок, проверка кодировщика, испорченный вход."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lipid_bits_")
        self.path = os.path.join(self.dir, "sdr_words.bin")

    def _store(self, cfg="abc123", **kw):
        return BitsStore(cfg_hash=cfg, path=self.path, **kw)

    def test_write_then_read(self):
        st = self._store()
        st.put("кошка", pack_bits([1, 2, 3]))
        st.put("джойстик", pack_bits(list(range(20))))
        self.assertTrue(st.flush())
        st2 = self._store()
        self.assertEqual(st2.load(), 2)
        self.assertEqual(unpack_bits(st2.get("кошка")), [1, 2, 3])
        self.assertEqual(len(unpack_bits(st2.get("джойстик"))), 20)
        self.assertIsNone(st2.reason)

    def test_header_fields_are_where_declared(self):
        st = self._store(cfg="deadbeef")
        st.put("стол", pack_bits([7]))
        raw = st.to_bytes()
        magic, ver, on_bits, total, cfg, count = struct.unpack_from("<4sBBH16sI", raw, 0)
        self.assertEqual(magic, MAGIC)
        self.assertEqual(ver, 1)
        self.assertEqual(on_bits, 20)
        self.assertEqual(total, 1024)
        self.assertEqual(cfg.rstrip(b"\x00").decode(), "deadbeef")
        self.assertEqual(count, 1)

    def test_word_is_utf8_with_explicit_length(self):
        st = self._store()
        st.put("флюгер", pack_bits([1]))
        raw = st.to_bytes()
        head = 4 + 1 + 1 + 2 + 16 + 4            # layout из docstring модуля
        (wl,) = struct.unpack_from("<H", raw, head)
        self.assertEqual(raw[head + 2: head + 2 + wl].decode("utf-8"), "флюгер")

    def test_foreign_encoder_is_loud_refusal(self):
        st = self._store(cfg="OLD")
        st.put("кошка", pack_bits([1]))
        st.flush()
        other = self._store(cfg="NEW")
        self.assertEqual(other.load(), 0, "чужие би́ты обязаны быть ОТВЕРГНУТЫ")
        self.assertIsNotNone(other.reason)
        self.assertIn("ДРУГИМ кодировщиком", other.reason)

    def test_changed_on_bits_refused(self):
        st = self._store(on_bits=20)
        st.put("кошка", pack_bits([1]))
        st.flush()
        other = self._store(on_bits=32)
        self.assertEqual(other.load(), 0)
        self.assertIn("активных бит", other.reason)

    def test_truncated_file_is_refused_not_garbage(self):
        st = self._store()
        for w in ("а", "б", "в", "г"):
            st.put(w, pack_bits([1]))
        st.flush()
        with open(self.path, "rb") as fh:
            raw = fh.read()
        with open(self.path, "wb") as fh:
            fh.write(raw[: len(raw) - 40])
        other = self._store()
        self.assertEqual(other.load(), 0)
        self.assertIn("обрезан", other.reason)

    def test_garbage_magic(self):
        with open(self.path, "wb") as fh:
            fh.write(b"XXXX" + b"\x00" * 64)
        st = self._store()
        self.assertEqual(st.load(), 0)
        self.assertIn("magic", st.reason)

    def test_missing_file_is_not_an_error_state(self):
        st = self._store()
        self.assertEqual(st.load(), 0)
        self.assertIn("нет", st.reason)

    def test_flush_is_idempotent_and_atomic(self):
        st = self._store()
        st.put("дом", pack_bits([5]))
        self.assertTrue(st.flush())
        self.assertFalse(st.flush(), "без изменений писать нечего")
        st.put("дом", pack_bits([5]))
        self.assertFalse(st.flush(), "повторная та же запись — не изменение")
        st.put("дом", pack_bits([6]))
        self.assertTrue(st.flush())
        again = self._store()
        again.load()
        self.assertEqual(unpack_bits(again.get("дом")), [6])

    def test_repr_reports_reason(self):
        st = self._store()
        st.load()
        self.assertIn("reason", repr(st),
                      "причина пустого кэша обязана быть видна из repr, "
                      "иначе «кэш не работает» выглядит как «тормозит»")


class TestEncoderUsesBitsStore(unittest.TestCase):
    """
    Кэш обязан УСКОРЯТЬ, НЕ МЕНЯЯ результат. Требует боевого кодировщика,
    то есть .venv с torch — вне его тест честно ПРОПУСКАЕТСЯ.
    """

    #тяжёлая часть общая на класс: модель грузится ~9 с, калибровка ~12 с.
    _shared = None

    @classmethod
    def setUpClass(cls):
        try:
            from core.encoder import HAS_NUMPY
            if not HAS_NUMPY:
                raise ImportError("нет numpy")
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(
                f"нужен .venv с torch/sentence-transformers (запускай "
                f".venv/bin/python): {exc}")
        from core.encoder import EmbeddingEncoder
        # калибровка — в общий временный каталог, один раз на класс
        cls._cal_dir = tempfile.mkdtemp(prefix="lipid_enc_cal_")
        cls._shared = EmbeddingEncoder(
            cache_path=os.path.join(cls._cal_dir, "cal.pkl"),
            bits_path=os.path.join(cls._cal_dir, "warm.bin"),
            lazy=False)

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lipid_enc_")

    def _encoder(self):
        """
        КАЖДЫЙ тест получает СВОЙ кодировщик со СВОИМИ файлами (иначе кэш
        одного теста просочится в другой), но с ОБЩЕЙ моделью: кэш и калибровка
        — это то, что проверяется, а весовая часть здесь декорация.
        """
        from core.encoder import EmbeddingEncoder
        # lazy=True ОБЯЗАТЕЛЬНО: с lazy=False конструктор сам грузит модель,
        # и «общая модель» превращается в 9 секунд на каждый тест.
        enc = EmbeddingEncoder(cache_path=os.path.join(self.dir, "cal.pkl"),
                               bits_path=os.path.join(self.dir, "w.bin"),
                               lazy=True)
        enc.warm_from(self._shared)
        return enc

    def test_cached_bits_equal_model_bits(self):
        import numpy as np
        words = ["кошка", "флюгера", "джойстик", "шизокарпий", "гроз"]
        enc = self._encoder()
        fresh = enc.encode_many(words)                  # считала модель
        enc.flush()
        enc2 = self._encoder()                          # новый «процесс»
        cached = enc2.encode_many(words)
        for i, w in enumerate(words):
            self.assertTrue((fresh[i] == cached[i]).all(),
                            f"«{w}»: кэш дал ДРУГИЕ би́ты — он стал источником "
                            f"знания, а не ускорителем")
            self.assertEqual(int(cached[i].sum()), 20)
        self.assertEqual(enc2.bits_hits, len(words))

    def test_duplicates_and_case_use_one_entry(self):
        enc = self._encoder()
        out = enc.encode_many(["Кошка", "кошка", "  кошка ", "кошка"])
        ref = enc.encode("кошка")
        for i in range(4):
            self.assertTrue((out[i] == ref).all(),
                            f"строка {i}: нормализация поехала, кэш развалится "
                            f"на копии одного слова")
        self.assertTrue((out[3] == ref).all())
        # пустая строка — нулевой SDR, и она НЕ должна попадать в кэш
        blank = enc.encode_many([""])
        self.assertEqual(int(blank[0].sum()), 0)
        enc.flush()
        self.assertIsNone(enc._ensure_bits().get(""))

    def test_partial_cache_hits_only_misses(self):
        import numpy as np
        enc = self._encoder()
        enc.encode_many(["кошка", "стол"])
        enc.flush()
        enc2 = self._encoder()
        out = enc2.encode_many(["кошка", "стол", "новыйслов"])
        self.assertEqual(enc2.bits_hits, 2)
        self.assertEqual(enc2.bits_misses, 1)
        self.assertEqual([int(x.sum()) for x in out], [20, 20, 20])
        self.assertTrue((out[0] == enc.encode("кошка")).all())
        self.assertEqual(out[2].sum(), 20, "новое слово обязано быть посчитано")

    def test_calibration_file_no_longer_stores_floats(self):
        """Пункт 0.1 очереди: 44 МБ float32 -> только среднее (384 float)."""
        enc = self._encoder()
        enc.encode_many(["кошка", "кот"])
        enc._cache_dirty = True             # warm_from не помечает калибровку новой
        enc.flush()
        import pickle
        with open(os.path.join(self.dir, "cal.pkl"), "rb") as fh:
            blob = pickle.load(fh)
        self.assertIn("mean", blob)
        self.assertNotIn("emb", blob,
                         "float-кэш вернулся в калибровку — файл опять вырастет "
                         "до десятков мегабайт")
        size = os.path.getsize(os.path.join(self.dir, "cal.pkl"))
        self.assertLess(size, 10_000, f"файл калибровки {size} байт — должен быть ~1 КБ")

    def test_bit_file_size_is_bytes_not_vectors(self):
        enc = self._encoder()
        words = [f"слово{i}" for i in range(200)]
        enc.encode_many(words)
        enc.flush()
        size = os.path.getsize(os.path.join(self.dir, "w.bin"))
        # заголовок + 200 × (128 байт бит + длина слова + сама строка)
        self.assertLess(size, 200 * 200, "разрядность бит ушла в разы — проверь layout")
        self.assertGreater(size, 200 * 128, "бит в файле меньше, чем должно быть")


class TestNoDepsDegradation(unittest.TestCase):
    """core/bitsstore обязан работать на голом stdlib: это слой переноса в C."""

    def test_import_without_numpy(self):
        code = ("import sys, importlib;"
                "sys.modules['numpy']=None;"
                "import core.bitsstore as b;"
                "print(len(b.pack_bits([1,64,1023])))")
        import subprocess
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "128")


if __name__ == "__main__":
    unittest.main(verbosity=2)
