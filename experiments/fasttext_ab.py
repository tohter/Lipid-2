# -*- coding: utf-8 -*-
"""
Пробник A/B: MiniLM (текущая сетчатка) против fastText (кандидат).

ЧТО МЕРИМ
---------
1. ФОРМЫ одного слова (узнавание флюгера): косинус форм против косинуса
   несвязанных слов — разделение должно быть резким.
2. СИНОНИМЫ против несвязанных: «кот~кошка» должен быть ближе к 1,
   «кот~тумбочка» — к 0.
3. Поведение после проекции на те же 1024 гиперплоскости: перекрытие
   20-битных SDR у синонимов против несвязанных пар (это уже язык ядра).

ВЕРДИКТ ПО ЦИФРАМ, не по вкусу: кандидат обязан не путать классы
(синоним/форма/несвязанные) на обеих метриках.

Запуск: LIPID2_STATUS=0 .venv/bin/python experiments/fasttext_ab.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LIPID2_STATUS", "0")

import numpy as np

from core.encoder import (EmbeddingEncoder, TOTAL_BITS, TARGET_ON_BITS,
                          _FALLBACK_CALIB, LetterHashEncoder)

# ── тестовые классы слов ────────────────────────────────────────
FORMS = {
    "флюгер": ["флюгер", "флюгера", "флюгером", "флюгеры", "флюгеров"],
    "кошка": ["кошка", "кошки", "кошкой", "кошек", "кошку"],
    "освещать": ["освещать", "освещал", "освещала", "освещало", "освещали"],
}

SYN = [  # синонимические пары (s1, s2) и их антисинонимы
    ("кот", "кошка"), ("собака", "пёс"), ("дом", "домик"),
    ("город", "град"), ("чудовище", "монстр"), ("тайна", "секрет"),
    ("смелый", "храбрый"), ("холодный", "стужа"), ("мчаться", "мчался"),
    ("говорить", "сказать"),
]

UNREL = [  # намеренно несвязанные пары
    ("кот", "тумбочка"), ("флюгер", "тумбочка"), ("тайна", "самовар"),
    ("храбрый", "клей"), ("мчаться", "скатерть"), ("дом", "гнев"),
    ("город", "клей"), ("собака", "бумага"), ("чудовище", "ложка"),
    ("говорить", "пятно"),
]

CALIB_POOL = list(_FALLBACK_CALIB) * 8  # пул для вычитания среднего


def split_stats(pairs, sim_fn, label):
    sims = np.array([sim_fn(a, b) for a, b in pairs])
    return sims


def sdr_overlap(enc, pairs):
    """Перекрытие 20-битных SDR пары слов (на 1024 битах)."""
    M = enc.encode_many([w for p in pairs for w in p])
    ov = []
    for i in range(0, len(pairs) * 2, 2):
        a, b = M[i], M[i + 1]
        ov.append(int((a & b).sum()))
    return np.array(ov)


def main():
    t0 = time.time()
    print("=" * 68)
    print("A/B: MiniLM против fastText (одна схема: центр→гиперплоскости→SDR)")
    print("=" * 68)

    # ── MiniLM ─────────────────────────────────────────────
    print("\n[1/3] Загрузка MiniLM…")
    mini = EmbeddingEncoder(
        cache_path=os.path.join(ROOT, "data", "memory", "sdr_calibration.pkl"),
        bits_path=os.path.join(ROOT, "data", "memory", "sdr_words.bin"))
    mini.calibrate(CALIB_POOL)

    # ── fastText ───────────────────────────────────────────
    vec_path = os.path.join(ROOT, "data", "fasttext", "cc.ru.300.vec.gz")
    if not os.path.exists(vec_path):
        print(f"\n!!! Нет файла {vec_path} — скачивание ещё идёт?")
        print("    Сначала: wget cc.ru.300.vec.gz (1,3 ГБ) в data/fasttext/")
        sys.exit(1)
    print("[2/3] Загрузка fastText-векторов (это может занять пару минут)…")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from fasttext_probe import FastTextEncoder
    ft = FastTextEncoder(vec_path)
    ft.calibrate(CALIB_POOL)

    # ── сравнение ──────────────────────────────────────────
    print("\n[3/3] Замеры…")
    results = {}
    for name, enc in (("MiniLM", mini), ("fastText", ft)):
        r = {}
        # косинусы (после центрирования — единая метрика обоих)
        for label, pairs in (("ФОРМЫ", None), ("СИНОНИМЫ", SYN), ("НЕСВЯЗ.", UNREL)):
            if label == "ФОРМЫ":
                sims = np.concatenate([
                    np.array([enc.similarity(w, forms[0]) for w in forms[1:]])
                    for forms in FORMS.values()])
            else:
                sims = split_stats(pairs, enc.similarity, label)
            r[label] = sims
        results[name] = r
        print(f"\n--- {name} (косинус после центрирования) ---")
        for label in ("ФОРМЫ", "СИНОНИМЫ", "НЕСВЯЗ."):
            s = r[label]
            print(f"  {label:9s}: min {s.min():.3f}  max {s.max():.3f}  "
                  f"средн {s.mean():.3f}")
        sep_syn = r["СИНОНИМЫ"].min() - r["НЕСВЯЗ."].max()
        sep_frm = r["ФОРМЫ"].min() - r["НЕСВЯЗ."].max()
        print(f"  РАЗДЕЛЕНИЕ синоним/несвяз: {sep_syn:+.3f} "
              f"(мин син - макс несвяз)")
        print(f"  РАЗДЕЛЕНИЕ форма/несвяз:   {sep_frm:+.3f}")

        # SDR-перекрытия (язык ядра)
        ov_syn = sdr_overlap(enc, SYN)
        ov_unrel = sdr_overlap(enc, UNREL)
        form_pairs = [(f, forms[0]) for forms in FORMS.values() for f in forms[1:]]
        ov_form = sdr_overlap(enc, form_pairs)
        r["ov_syn"] = ov_syn
        r["ov_unrel"] = ov_unrel
        r["ov_form"] = ov_form
        print(f"  SDR-перекрытие (из 20 бит): синонимы медиана "
              f"{int(np.median(ov_syn))}, диапазон {ov_syn.min()}..{ov_syn.max()}")
        print(f"                              формы медиана "
              f"{int(np.median(ov_form))}, диапазон {ov_form.min()}..{ov_form.max()}")
        print(f"                              несвязанные медиана "
              f"{int(np.median(ov_unrel))}, диапазон "
              f"{ov_unrel.min()}..{ov_unrel.max()}")

    # ── скорость ───────────────────────────────────────────
    words = [f"слово{i}" for i in range(200)] + \
        [w for p in SYN + UNREL for w in p]
    print("\n--- Скорость (после прогрева) ---")
    for name, enc in (("MiniLM", mini), ("fastText", ft)):
        enc.encode_many(words[:50])  # прогрев
        t0 = time.time()
        enc.encode_many(words)
        dt = time.time() - t0
        print(f"  {name}: {dt*1000:.1f} мс на {len(words)} слов "
              f"(с учётом кэша повторов)")

    print("\n" + "=" * 68)
    print("Замер окончен. Вердикт: классы обязаны НЕ пересекаться.")
    print("=" * 68)


if __name__ == "__main__":
    main()
