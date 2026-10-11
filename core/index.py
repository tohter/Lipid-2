"""
Индекс SDR-перекрытий.

Зачем. Предиктивному ядру на КАЖДОМ тике нужен argmax overlap по всему графу.
Прежний _recall был чистым Python-циклом: для графа из N узлов это N вызовов
overlap(), каждый — 16 итераций с bin().count(). На N=10^4 это ~160 000 тяжёлых
операций на тик, а тиков в потоке тысячи. Граф упрётся в скорость раньше, чем
в качество распознавания.

Здесь: то же перекрытие, но матрицей (numpy/scipy, если есть) и с точным
фоллбэком на чистом Python — ядро обязано работать и без科学-стека.

Дополнительно отдаёт ВТОРОЙ лучший результат. Это не украшение: «узнан» — это
отношение, а не абсолют. При 20 000 реальных русских слов ЛУЧШИЙ чужой узел
даёт 9-11 совпавших бит из 20 (медиана), и абсолютный порог this перестаёт
существовать. Маргар best - second — единственная величина, которая растёт,
когда концепт действительно в графе (overlap = 20 против 9), и не растёт,
когда его там нет.
"""
import random
import threading
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:                                    # pragma: no cover
    np = None
    HAS_NUMPY = False

try:
    import scipy.sparse as _sp
    HAS_SCIPY = True
except ImportError:                                    # pragma: no cover
    _sp = None
    HAS_SCIPY = False

from core.sdr import SDR1024, TOTAL_BITS


class SdrIndex:
    """
    Позиционный индекс: узлы добавляются по одному, id = порядок добавления.
    Хранит плотную булеву матрицу (n, 1024) и, если есть scipy, CSR-разреженную
    копию для быстрых поисков.
    """

    def __init__(self):
        self._keys: List = []                 # node_id -> внешний ключ
        self._pos = {}                        # внешний ключ -> позиция
        self._rows: List = []                 # плотные uint8-строки (fallback)
        self._dense = None                    # кэш склеенной матрицы
        self._csr = None                      # кэш CSR
        self._dirty = False
        self._lock = threading.RLock()

    # ── изменение ───────────────────────────────────────────
    def add(self, key, sdr: SDR1024) -> int:
        with self._lock:
            if key in self._pos:
                return self._pos[key]
            pos = len(self._keys)
            self._keys.append(key)
            self._pos[key] = pos
            self._rows.append(SdrIndex._as_mask(sdr))
            self._dirty = True
            return pos

    def build_from(self, items: Sequence[Tuple[object, SDR1024]]):
        """Пачкой (загрузка графа с диска): один проход вместо N вызовов add()."""
        with self._lock:
            for key, sdr in items:
                self.add(key, sdr)

    def remove_all(self):
        with self._lock:
            self.__init__()

    @staticmethod
    def _mask_from_bits(bits: Sequence[int]):
        mask = np.zeros(TOTAL_BITS, dtype=np.uint8)
        for w, blk in enumerate(bits):
            for k in range(64):
                if (blk >> k) & 1:
                    mask[w * 64 + k] = 1
        return mask

    @staticmethod
    def _as_mask(sdr: SDR1024):
        """SDR -> плотная uint8-строка (или sorted-список битов на fallback)."""
        if HAS_NUMPY:
            return SdrIndex._mask_from_bits(sdr.bits)
        return sdr.active_indices()

    # ── сборка представлений ────────────────────────────────
    def _materialize(self):
        if not self._dirty:
            return
        if not HAS_NUMPY or not self._keys:
            self._dirty = False
            return
        self._dense = np.stack(self._rows) if self._rows else np.zeros((0, TOTAL_BITS), np.uint8)
        if HAS_SCIPY:
            self._csr = _sp.csr_matrix(self._dense)
        self._dirty = False

    def __len__(self):
        return len(self._keys)

    # ── поиск ───────────────────────────────────────────────
    def topk(self, sdr: SDR1024, k: int = 3) -> List[Tuple[object, int]]:
        """k лучших узлов по перекрытию: [(ключ, overlap), ...] по убыванию."""
        with self._lock:
            self._materialize()
            n = len(self._keys)
            if n == 0:
                return []
            k = max(1, min(k, n))
            if HAS_NUMPY:
                q = SdrIndex._as_mask(sdr)
                if HAS_SCIPY:
                    vec = np.asarray(self._csr @ q).ravel()
                else:
                    vec = self._dense @ q.astype(np.int32)
                vec = np.asarray(vec, dtype=np.int32)
                order = np.argpartition(-vec, k - 1)[:k]
                order = order[np.argsort(-vec[order], kind="stable")]
                return [(self._keys[i], int(vec[i])) for i in order]
            qset = set(sdr.active_indices())
            scored = sorted(((len(qset & set(r)), i) for i, r in enumerate(self._rows)),
                            key=lambda x: -x[0])[:k]
            return [(self._keys[i], int(s)) for s, i in scored]

    def best_and_second(self, sdr: SDR1024) -> Tuple[Optional[object], int, int]:
        """
        Возвращает (лучший_узел, best_overlap, second_overlap).
        second_overlap — перекрытие с ЛУЧШИМ ЧУЖИМ, то есть локальный шумовой пол.
        """
        top = self.topk(sdr, 2)
        if not top:
            return None, 0, 0
        best_key, best_ov = top[0]
        second_ov = top[1][1] if len(top) > 1 else 0
        return best_key, best_ov, second_ov

    # ── оценка шумового пола самого графа ───────────────────
    def self_probe_floor(self, sample: int = 120) -> dict:
        """
        Насколько сильно ТИПИЧНОЕ слово графа пересекается со своим ближайшим
        соседом. Это шумовой пол, от которого надо отталкиваться, выбирая порог
        «узнали / не узнали».

        Оценивается узлами САМОГО графа: каждый узел ищется против остальных,
        сам себя он исключает. Поэтому оценка НЕ зависит от того, какое слово
        пришло на текущем тике. Прежняя версия меряла пол от текущего входа —
        и когда вход узнавался, пол выпрыгивал на 19-20, унося порог за потолок.

        Замер на 15 000 настоящих русских слов (hunspell ru_RU):
            размер графа   медиана пола
                   500            10
                 2 000             9
                 8 000            10
                15 000            10
        Пол держится на ~10 битах и почти НЕ растёт со словарём — в отличие от
        буквенного хэша, где шум рос лавинообразно и любое фиксированное число
        в конфиге молча устаревало по мере роста графа.
        """
        with self._lock:
            self._materialize()
            n = len(self._keys)
            if n < 3:
                return {"n": 0, "median": 0, "p90": 0, "max": 0}
            idx = list(range(n))
            if n > sample:
                idx = random.Random(0xC0FFEE).sample(idx, sample)
            bests = []
            for i in idx:
                if HAS_NUMPY:
                    if HAS_SCIPY:
                        vec = np.asarray(self._csr[i].dot(self._csr.T).todense()).ravel()
                    else:
                        vec = self._dense @ self._dense[i].astype(np.int32)
                    vec = np.asarray(vec, dtype=np.int32)
                    vec[i] = -1                       # сам себя не считаем
                    bests.append(int(vec.max()))
                else:
                    me = set(self._rows[i])
                    bests.append(max((len(me & set(r)) for j, r in enumerate(self._rows)
                                      if j != i), default=0))
            if not bests:
                return {"n": 0, "median": 0, "p90": 0, "max": 0}
            if HAS_NUMPY:
                b = np.asarray(bests)
                return {"n": len(b), "median": float(np.median(b)),
                        "p90": float(np.percentile(b, 90)), "max": int(b.max())}
            b = sorted(bests)
            return {"n": len(b), "median": b[len(b) // 2],
                    "p90": b[min(len(b) - 1, int(len(b) * 0.90))], "max": b[-1]}

    def key_at(self, pos: int):
        return self._keys[pos] if 0 <= pos < len(self._keys) else None
