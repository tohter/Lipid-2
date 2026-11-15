# -*- coding: utf-8 -*-
"""
FastTextEncoder — ПОДКЛАСС EmbeddingEncoder, меняется ТОЛЬКО источник
плотных векторов (.vec-файл fastText вместо SentenceTransformer).

Зачем так: вся SDR-механика (вычитание среднего → top-k гиперплоскостей →
ровно 20 бит из 1024 → битовый кэш bitsstore) остаётся РОДНОЙ, покрытой
тестами. A/B тогда честный: сравниваем сетчатки, а не «две системы».

Зачем вообще: MiniLM — трансформер (33 млн параметров, 449 МБ, ~9,5 с
загрузки). fastText — неглубокая сеть, словарь ужимается в компактный пак,
загрузка — чтение текста, кодирование — сумма векторов.

ВЕКТОРА: cc.ru.300.vec (Facebook, CC-BY-SA). Формат: строка «N dim», далее
«слово v1..v300». Загружаем верхний частотный срез. Слово без вектора —
детерминированный субсловный вектор (n-граммы 3..6 по хэшу, та же идея
fastText, реализована на месте, без .bin).
"""
import gzip
import hashlib
import os
import sys
import threading
import time
from typing import Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from core.encoder import (EmbeddingEncoder, TOTAL_BITS, TARGET_ON_BITS,
                          EncodeError)

# Сколько слов держать в RAM (наиболее частотные идут первыми в .vec).
MAX_WORDS = int(os.environ.get("LIPID2_FT_MAXWORDS", "2100000"))
# Размерность векторов (cc.ru.300 → 300).
DIM = 300


class _VecTable:
    """Таблица векторов: word -> строка матрицы, плюс субсловный резерв."""

    def __init__(self, path: str, max_words: int = MAX_WORDS):
        self.word2idx: dict = {}
        self.vecs: np.ndarray | None = None
        self.dim = DIM
        self._load(path, max_words)

    def _load(self, path: str, max_words: int):
        t0 = time.time()
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().split()
            n, dim = int(header[0]), int(header[1])
            self.dim = dim
            n = min(n, max_words)
            self.vecs = np.zeros((n, dim), dtype=np.float32)
            i = 0
            while i < n:
                line = fh.readline()
                if not line:
                    break
                parts = line.rstrip("\n").split(" ")
                if len(parts) != dim + 1:
                    continue
                self.word2idx[parts[0]] = i
                self.vecs[i] = parts[1:]
                i += 1
        self.vecs = self.vecs[:i]
        print(f"[fastText] словарь: {len(self.word2idx)} слов, dim={self.dim}, "
              f"{time.time()-t0:.1f} с")

    def has(self, w: str) -> bool:
        return w in self.word2idx

    def raw_vector(self, word: str) -> np.ndarray:
        """Вектор слова: из словаря, иначе субсловный (n-граммы по хэшу)."""
        if word in self.word2idx:
            return self.vecs[self.word2idx[word]]
        v = np.zeros(self.dim, dtype=np.float32)
        w = f"<{word}>"
        grams = [w[i:i+n] for n in (3, 4, 5, 6)
                 for i in range(0, max(len(w) - n + 1, 0))]
        if not grams:
            grams = [w]
        for g in grams:
            h = hashlib.sha1(g.encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.default_rng(seed)
            v += rng.standard_normal(self.dim).astype(np.float32)
        v /= max(len(grams), 1)
        return v


class FastTextEncoder(EmbeddingEncoder):
    """
    Тот же контракт: центр → 1024 гиперплоскостей → ровно 20 бит → bitsstore.
    Источник векторов — таблица .vec fastText.
    """
    name = "embedding"          # имя НЕ меняем: формат мира и битов тот же
    semantic = True

    def __init__(self, vec_path: str, n_hyper: int = None, seed: int = None,
                 on_bits: int = TARGET_ON_BITS,
                 calibration_texts: Sequence[str] = None,
                 cache_path: str = None, bits_path: str = None,
                 lazy: bool = True):
        self.vec_path = vec_path
        self._table = None
        self._table_lock = threading.RLock()
        super().__init__(model_name=f"fasttext:{os.path.basename(vec_path)}",
                         n_hyper=n_hyper, seed=seed, on_bits=on_bits,
                         calibration_texts=calibration_texts,
                         cache_path=cache_path, bits_path=bits_path,
                         lazy=True)
        if not lazy:
            self._warm()

    # ── источник векторов и гиперплоскости ───────────────────
    def _warm(self):
        with self._lock:
            if self._R is not None:
                return
            if not np:
                raise EncodeError("FastTextEncoder требует numpy")
            rng = np.random.default_rng(self.seed)
            self._R = rng.standard_normal((self.n_hyper, DIM)).astype(np.float32)
            self._R /= np.linalg.norm(self._R, axis=1, keepdims=True)
            self._load_calibration()
            if self._table is None:
                with self._table_lock:
                    self._table = _VecTable(self.vec_path)

    def _raw_encode(self, texts: Sequence[str]) -> "np.ndarray":
        """Dense-вектора из таблицы (без torch, без сети, без HF)."""
        with self._table_lock:
            if self._table is None:
                self._table = _VecTable(self.vec_path)
        out = np.stack([self._table.raw_vector(t.strip().lower())
                        for t in texts]).astype(np.float32)
        # нормализация — как у SentenceTransformer(normalize_embeddings=True)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        out /= np.maximum(n, 1e-9)
        return out
