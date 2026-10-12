"""
Золотые би́ты: защита от ТИХОЙ смены кодировщика.

ЗАЧЕМ ЭТОТ ФАЙЛ СУЩЕСТВУЕТ
--------------------------
Все пороги узнавания (13/17, шумовой пол, unknown_threshold) вымерены на
КОНКРЕТНЫХ битах, которые даёт paraphrase-multilingual-MiniLM-L12-v2 в
КОНКРЕТНЫХ версиях sentence-transformers/transformers/torch. На других версиях
эмбеддинги ДРУГИЕ — значит пороги другие, а код об этом не догадается: SDR
выглядит «семантическим», overlap выглядит «в норме», и молча это вскрывается
только на живом ответе системы.

Единственная защита, которую нельзя подправить «по ходу дела», — зафиксированные
би́ты курируемого списка слов. Тест сверяет их побитово и краснеет, если:
  1. обновлены версии библиотек (сменились эмбеддинги);
  2. заменена модель или её веса;
  3. изменён бинаризатор (центрирование, top-k, seed, число гиперплоскостей).
Во всех трёх случаях пороги обязаны быть ПЕРЕСЧИТАНЫ, а тест — починен только
после этого, осознанно и отдельным коммитом.

Эталон лежит в tests/golden_bits.json (текстовый, читается человеком, в git).

ОБНОВЛЕНИЕ ОСОЗНАННО:
    .venv/bin/python tests/test_golden_bits.py --write
"""
import argparse
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN_PATH = os.path.join(ROOT, "tests", "golden_bits.json")

# Слова ЗАДАНЫ ВРУЧНУЮ, а не «первые N из мира»: набор должен покрывать то, на
# чём строились пороги — морфологию, синонимию, орфографические ловушки, редкие
# и несуществующие слова, одно длинное слово.
GOLDEN_WORDS = [
    "кошка", "кошки", "кошке",                          # формы одного слова
    "кот", "котёнок",                                   # синонимия / диминутив
    "флюгер", "флюгера",                                # редкое + чередование
    "гроз", "гроза", "грозы",                           # беглая гласная
    "табуретка", "алгебра",                             # несвязанные слова
    "джойстик", "шизокарпий",                           # редкое / вымышленное
    "дать", "дуб",                                      # части фразеологизма
    "сельскохозяйственно-машиностроительный",           # длинное слово
]


def bits_for(words):
    """{слово: [индексы активных бит]} через БОЕВОЙ singleton + cfg-хэш."""
    from core.encoder import get_encoder
    enc = get_encoder()
    masks = enc.encode_many(list(words))
    out = {}
    for w, m in zip(words, masks):
        try:
            import numpy as np
            idx = sorted(int(i) for i in np.flatnonzero(m))
        except ImportError:                                     # pragma: no cover
            idx = sorted(i for i, b in enumerate(m) if b)
        out[w] = idx
    return out, enc.config_hash()


def write_golden(path: str = GOLDEN_PATH) -> dict:
    from core.sdr import encoder_info
    from core.encoder import TARGET_ON_BITS
    name, semantic, cfg = encoder_info()
    bits, cfg2 = bits_for(GOLDEN_WORDS)
    assert cfg == cfg2, f"cfg-хэш поехал внутри одного вызова: {cfg} != {cfg2}"
    doc = {"encoder": name, "semantic": semantic, "config_hash": cfg,
           "on_bits": TARGET_ON_BITS, "words": bits}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1, sort_keys=True)
    return doc


class TestGoldenBits(unittest.TestCase):
    """Би́ты известных слов обязаны совпадать побитово."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GOLDEN_PATH):
            raise unittest.SkipTest(
                "нет эталона tests/golden_bits.json — сгенерируй: "
                ".venv/bin/python tests/test_golden_bits.py --write")
        with open(GOLDEN_PATH, encoding="utf-8") as fh:
            cls.gold = json.load(fh)

    def test_encoder_identity(self):
        from core.sdr import encoder_info
        name, semantic, cfg = encoder_info()
        self.assertEqual(name, self.gold["encoder"],
                         "сменился сам кодировщик — би́ты несравнимы")
        self.assertTrue(semantic,
                        "проверка золотых битов на буквенном хэше бессмысленна: "
                        "запускай из .venv, где стоит sentence-transformers")
        self.assertEqual(cfg, self.gold["config_hash"],
                         f"КОНФИГУРАЦИЯ КОДИРОВЩИКА СМЕНИЛАСЬ: было "
                         f"{self.gold['config_hash']}, стало {cfg}. Пороги "
                         f"(13/17, шумовой пол) вымерены на старых битах и "
                         f"требуют пересчёта.")

    def test_bits_match_bit_for_bit(self):
        current, _ = bits_for(list(self.gold["words"].keys()))
        bad = []
        for word, want in self.gold["words"].items():
            got = current.get(word, [])
            if got != sorted(want):
                bad.append(f"  {word}: совпало {len(set(got) & set(want))}"
                           f"/{len(want)} бит")
        self.assertFalse(bad,
                         "БИТЫ СМЕНИЛИСЬ — пороги узнавания больше не имеют силы, "
                         "нужен новый замер (experiments/RESULTS.md):\n"
                         + "\n".join(bad))

    def test_density_invariant_holds(self):
        from core.encoder import TARGET_ON_BITS
        self.assertEqual(self.gold["on_bits"], TARGET_ON_BITS,
                         "инвариант архитектуры (20 активных бит) поехал")
        for word, bits in self.gold["words"].items():
            self.assertEqual(len(bits), TARGET_ON_BITS,
                             f"«{word}»: разреженность {len(bits)} != "
                             f"{TARGET_ON_BITS}")
            self.assertTrue(all(0 <= b < 1024 for b in bits),
                            f"«{word}»: би́ты вылезли за 1024")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="записать новый эталон (после этого ПЕРЕМЕРИТЬ пороги)")
    args, rest = ap.parse_known_args()
    if args.write:
        doc = write_golden()
        print(f"записано {GOLDEN_PATH}: слов {len(doc['words'])}, "
              f"cfg={doc['config_hash']}, on_bits={doc['on_bits']}")
        print("⚠️  если би́ты отличались от прежнего эталона — пороги "
              "(UNKNOWN_CEILING/KNOWN_FLOOR/HIT_THRESHOLD) обязаны быть "
              "перемерены, а не унаследованы")
    else:
        unittest.main(argv=[sys.argv[0]] + rest)
