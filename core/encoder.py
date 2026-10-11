"""
Кодировщики SDR для Lipid-2.

История вопроса. Прежний кодировщик строил 20 активных бит из SHA-256 буквенных
биграмм/триграмм. Он кодирует НАПИСАНИЕ, а не СМЫСЛ:
    кошка~кот = 3/20,  кошка~табуретка = 4/20
а весь downstream (_recall, «узнал/не узнал», bootstrap) трактовал overlap как
семантическое узнавание. Это неустранимо порогом — нужна смена кодировщика.

Здесь:
  LetterHashEncoder   — прежнее поведение. Оставлен как fallback без зависимостей
                        и как точка отсчёта в тестах.
  EmbeddingEncoder    — dense-эмбеддинг -> центрирование -> top-k гиперплоскостей
                        (Random Projection Pursuit) -> ровно 20 бит из 1024.

ВАЖНО ПРО ЦЕНТРИРОВАНИЕ. Мультиязычные модели (MiniLM-L12-v2) сильно анизотропны:
у «сырых» эмбеддингов косинус у ЛюБой пары слов >= 0.9 («цыкающий» и «беломорец» —
0.95). Без вычитания среднего компонента ЛЮБЫЕ два SDR дают overlap ~15/20, и
узнавание превращается в шум хуже прежнего. Вычитание среднего возвращает
несвязанные пары к нулю и оставляет смысл.

ВАЖНО ПРО ВЫБОР БИТ. Знаковый LSH (bit = sign(v·r)) даёт ~500 активных бит из 1024,
и прунинг до 20 по |проекции| убивает разреженность перекрытий (несвязанные пары
лезут за 200 бит). Работает именно top-k по проекции: перекрытие тогда равно
числу общих «ближайших гиперплоскостей», то есть мере близости в эмбеддинг-пространстве.
"""
import hashlib
import os
import pickle
import struct
import threading
import zlib
from typing import Iterable, List, Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:                                   # pragma: no cover
    np = None
    HAS_NUMPY = False

TOTAL_BITS = 1024
NUM_U64 = TOTAL_BITS // 64
TARGET_ON_BITS = 20

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
# Число гиперплоскостей ОБЯЗАНО быть <= 1024: бит SDR — это номер гиперплоскости,
# соответствие 1:1. Иначе (проверено) индексы вылезают за 1024, а схлопывание
# по mod рождает коллизии и «съедает» часть из 20 активных бит.
DEFAULT_NHYPER = 1024
DEFAULT_SEED = 20240611
DEFAULT_CALIBRATION_N = 8192

_env = os.environ.get


class EncodeError(RuntimeError):
    pass


# ══════════════════════════════════════════════════════════════════════
#  БАЗА
# ══════════════════════════════════════════════════════════════════════
class BaseEncoder:
    """Интерфейс: тексты -> булева матрица (n, TOTAL_BITS) с ровно ON битами."""

    name = "base"
    semantic = False

    def config_hash(self) -> str:
        return self.name

    def encode_many(self, texts: Sequence[str]):
        raise NotImplementedError

    def encode(self, text: str):
        return self.encode_many([text])[0]

    # -- утилиты паковки/распаковки (общие для всех кодировщиков) ------
    @staticmethod
    def mask_to_u64(mask) -> List[int]:
        """bool/uint8 вектор длины 1024 -> список из 16 uint64."""
        idx = np.flatnonzero(mask) if HAS_NUMPY else [i for i, b in enumerate(mask) if b]
        bits = [0] * NUM_U64
        for b in idx:
            bits[int(b) // 64] |= (1 << (int(b) % 64))
        return bits

    @staticmethod
    def u64_to_mask(bits: Sequence[int]):
        mask = np.zeros(TOTAL_BITS, dtype=np.uint8)
        for w, blk in enumerate(bits):
            for k in range(64):
                if blk >> k & 1:
                    mask[w * 64 + k] = 1
        return mask


# ══════════════════════════════════════════════════════════════════════
#  КОДИРОВЩИК №1 — буквенный хэш (исторический, fallback)
# ══════════════════════════════════════════════════════════════════════
class LetterHashEncoder(BaseEncoder):
    """
    Прежняя схема: SHA-256 по символьным 2-/3-граммам. Детерминирована, без
    зависимостей, кодирует орфографию. Смысловые тесты на ней КРАСНЫЕ — это
    ожидаемое и задокументированное поведение, а не регрессия.
    """

    name = "letterhash"
    semantic = False

    def encode_many(self, texts: Sequence[str]):
        return [self._one(t) for t in texts]

    def _one(self, text: str):
        norm = text.strip().lower()
        mask = bytearray(TOTAL_BITS)
        if not norm:
            return np.asarray(mask, dtype=np.uint8) if HAS_NUMPY else mask

        padded = f"#{norm}#"
        ngrams = []
        for i in range(len(padded) - 1):
            ngrams.append(padded[i:i + 2])
            if i < len(padded) - 2:
                ngrams.append(padded[i:i + 3])
        ngrams.append(norm)

        active = []
        seen = set()
        for ng in ngrams:
            h = hashlib.sha256(ng.encode("utf-8")).digest()
            idx = struct.unpack(">H", h[:2])[0] % TOTAL_BITS
            if idx not in seen:
                seen.add(idx)
                active.append(idx)
            if len(active) >= TARGET_ON_BITS:
                break
        counter = 0
        while len(active) < TARGET_ON_BITS:
            h = hashlib.sha256(f"{norm}_{counter}".encode("utf-8")).digest()
            idx = struct.unpack(">H", h[:2])[0] % TOTAL_BITS
            if idx not in seen:
                seen.add(idx)
                active.append(idx)
            counter += 1
        for b in sorted(active)[:TARGET_ON_BITS]:
            mask[b] = 1
        return np.asarray(mask, dtype=np.uint8) if HAS_NUMPY else mask


# ══════════════════════════════════════════════════════════════════════
#  КОДИРОВЩИК №2 — эмбеддинговый (основной)
# ══════════════════════════════════════════════════════════════════════
class EmbeddingEncoder(BaseEncoder):
    """
    text -> SentenceTransformer -> вычитание среднего -> top-k гиперплоскостей.

    model_name  — любая мультиязычная модель sentence-transformers.
    n_hyper     — число случайных гиперплоскостей (кандидатов на 20 активных бит).
    seed        — фиксирован, иначе кэш графа и бинарники .lipid рассинхронизируются.
    calibration_texts — пул для оценки «мусорного» среднего направления.
    cache_path  — .npz/.pkl с эмбеддингами калибровки; без него модель грузится
                  и калибруется заново при каждом старте.
    """

    name = "embedding"
    semantic = True

    def __init__(self, model_name: str = None, n_hyper: int = None, seed: int = None,
                 on_bits: int = TARGET_ON_BITS, calibration_texts: Sequence[str] = None,
                 cache_path: str = None, lazy: bool = True):
        if not HAS_NUMPY:
            raise EncodeError("EmbeddingEncoder требует numpy (pip install numpy)")
        self.model_name = model_name or _env("LIPID2_SDR_MODEL", DEFAULT_MODEL)
        self.n_hyper = int(n_hyper or _env("LIPID2_SDR_NHYPER", DEFAULT_NHYPER))
        self.seed = int(seed or _env("LIPID2_SDR_SEED", DEFAULT_SEED))
        self.on_bits = int(on_bits)
        self.cache_path = cache_path
        self._model = None
        self._R = None
        self._mean = None
        self._calib_texts = list(calibration_texts) if calibration_texts else None
        self._lock = threading.RLock()
        self._emb_cache = {}          # text -> np.ndarray (dense), LRU-подобный
        self._cache_dirty = False
        if not lazy:
            self._warm()

    # ── конфигурация и кэш ───────────────────────────────────
    def config_hash(self) -> str:
        raw = f"emb|{self.model_name}|{self.n_hyper}|{self.seed}|{self.on_bits}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _warm(self):
        with self._lock:
            if self._R is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:                 # pragma: no cover
                raise EncodeError(
                    "EmbeddingEncoder требует sentence-transformers:\n"
                    "  pip install sentence-transformers\n"
                    "Либо выставьте LIPID2_SDR_BACKEND=letterhash."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            getter = getattr(self._model, "get_sentence_embedding_dimension", None)
            dim = getter() if callable(getter) else self._model.get_embedding_dimension()
            if self.n_hyper > TOTAL_BITS:
                print(f"[SDR] n_hyper={self.n_hyper} урезан до {TOTAL_BITS}: "
                      f"бит SDR = номер гиперплоскости, больше битов физически нет")
                self.n_hyper = TOTAL_BITS
            rng = np.random.default_rng(self.seed)
            self._R = rng.standard_normal((self.n_hyper, dim)).astype(np.float32)
            self._R /= np.linalg.norm(self._R, axis=1, keepdims=True)
            self._load_calibration()

    def _calibration_file(self):
        if self.cache_path:
            return self.cache_path
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "data", "memory",
                            f"sdr_calibration_{self.config_hash()}.pkl")

    def _load_calibration(self):
        path = self._calibration_file()
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    blob = pickle.load(fh)
                if blob.get("config_hash") == self.config_hash():
                    self._mean = np.asarray(blob["mean"], dtype=np.float32)
                    self._emb_cache = dict(blob.get("emb", {}))
                    return
            except Exception:
                pass                                    # битый кэш -> пересчитаем
        self._mean = None
        self.calibrate()

    def calibrate(self, texts: Sequence[str] = None):
        """
        Оценивает «мусорное» направление = среднее по representative-пулу.
        Пул должен быть широким и частотным: среднее по 40 словам недооценивает
        сдвиг, и несвязанные пары снова лезут в overlap.
        """
        with self._lock:
            self._warm()
            pool = list(texts if texts is not None else (self._calib_texts or []))
            if not pool:
                pool = self._default_pool()
            pool = pool[:DEFAULT_CALIBRATION_N]
            E = self._raw_encode(pool)
            self._mean = E.mean(axis=0).astype(np.float32)
            self._mean /= max(float(np.linalg.norm(self._mean)), 1e-9)
            self._cache_dirty = True
            self._save_calibration()

    def _default_pool(self) -> List[str]:
        """
        Калибровочный пул: частотные русские слова. Ищем русский словарь системы
        (hunspell/myspell/dict), при отсутствии — скромный встроенный список.
        """
        for path in ("/usr/share/hunspell/ru_RU.dic", "/usr/share/myspell/ru_RU.dic",
                     "/usr/share/dict/ru"):
            if not os.path.exists(path):
                continue
            try:
                out, seen = [], set()
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        w = line.split("/")[0].strip()
                        if 3 <= len(w) <= 10 and w.isalpha() and w[0].islower():
                            root5 = w[:5]
                            if root5 in seen:
                                continue
                            seen.add(root5)
                            out.append(w)
                        if len(out) >= DEFAULT_CALIBRATION_N:
                            break
                if len(out) >= 2000:
                    return out
            except OSError:
                pass
        return list(_FALLBACK_CALIB)

    def _save_calibration(self):
        path = self._calibration_file()
        if self._mean is None or not os.path.isdir(os.path.dirname(path)):
            return
        try:
            with open(path, "wb") as fh:
                pickle.dump({"config_hash": self.config_hash(),
                             "mean": self._mean,
                             "emb": self._emb_cache}, fh, protocol=4)
            self._cache_dirty = False
        except OSError:
            pass

    def flush(self):
        if self._cache_dirty:
            self._save_calibration()

    # ── ядро ────────────────────────────────────────────────
    def _raw_encode(self, texts: Sequence[str]) -> "np.ndarray":
        """Dense-эмбеддинги с LRU-кэшем: граф перeкодировается при каждой загрузке."""
        todo = [t for t in texts if t not in self._emb_cache]
        if todo:
            vecs = self._model.encode(list(dict.fromkeys(todo)), normalize_embeddings=True,
                                      batch_size=256, show_progress_bar=False)
            vecs = np.asarray(vecs, dtype=np.float32)
            for t, v in zip(dict.fromkeys(todo), vecs):
                self._emb_cache[t] = v
            self._cache_dirty = True
        return np.stack([self._emb_cache[t] for t in texts]).astype(np.float32)

    def centered(self, texts: Sequence[str]) -> "np.ndarray":
        """
        Эмбеддинги БЕЗ анизотропного сдвига, нормированные. Это ЕДИНСТВЕННАЯ
        корректная метрика смысловой близости у этой модели.

        Почему нельзя брать «сырой» косинус: у мультиязычных SentenceTransformer
        все векторы набиты в один узкий конус, поэтому косинус ЛЮБОЙ пары слов
        лежит в 0.7-1.0 (замер: «булка~человек» = 0.736, «флюгер~тумбочка» = 0.825).
        На сыром косинусе любые два слова «похожи», и любой порог даёт 100%
        ложных срабатываний. Снятие среднего (проекция на ортогональное
        дополнение + перевыпенормировка) раздвигает классы: формы одного слова
        0.89-0.999, несвязанные 0.0-0.5.

        Та же центровка применяется внутри бинаризатора, поэтому биты SDR и эта
        метрика согласованы.
        """
        self._warm()
        E = self._raw_encode([t.strip().lower() for t in texts])
        if self._mean is not None:
            E = E - (E @ self._mean)[:, None] * self._mean
            E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        return E.astype(np.float32)

    def similarity(self, a: str, b: str) -> float:
        """Смысловая близость 0..1 после снятия анизотропии."""
        E = self.centered([a, b])
        return float(E[0] @ E[1])

    def encode_many(self, texts: Sequence[str]):
        texts = [t.strip().lower() for t in texts]
        out = np.zeros((len(texts), TOTAL_BITS), dtype=np.uint8)
        live = [i for i, t in enumerate(texts) if t]
        if not live:
            return out
        self._warm()
        sub = [texts[i] for i in live]
        E = self.centered(sub)
        proj = E @ self._R.T
        k = min(self.on_bits, proj.shape[1])
        idx = np.argpartition(-proj, k - 1, axis=1)[:, :k]
        # Индекс гиперплоскости == номер бита (n_hyper <= TOTAL_BITS), коллизий нет:
        # ровно k различных активных бит.
        out[np.arange(len(live))[:, None], idx] = 1
        return out


# Минимальный встроенный пул на случай отсутствия системных словарей.
_FALLBACK_CALIB = """
кошка кот собака дерево стол табуретка дом окно дверь книга слово буква звук свет тень
бег шаг сон еда хлеб вода огонь дождь снег лес поле гора море река рыба птица конь
волк лиса мышь крыша стена пол потолок город улица площадь машина колесо дорога
ночь день утро вечер год месяц неделя час минута друг враг любовь грусть страх гнев
радость мысль идея речь голос слух глаз рука нога голова сердце кровь кость кожа
нож молоток кирпич цемент стекло бумага перо чернила школа урок закон война мир
""".split()


# ══════════════════════════════════════════════════════════════════════
#  ФАбрика singleton
# ══════════════════════════════════════════════════════════════════════
_ENCODER = None
_ENCODER_LOCK = threading.RLock()


def get_encoder(force: BaseEncoder = None) -> BaseEncoder:
    """
    Кодировщик выбирается один раз на процесс: SDR узла обязан совпадать с SDR
    того же слова, полученным часом позже, иначе кэш и .lipid распадаются.

    LIPID2_SDR_BACKEND = letterhash | embedding (по умолчанию embedding).

    ОТКАТ НА LETTERHASH НЕ ТИХИЙ И НЕ БЕСПЛАТНЫЙ. Буквенный хэш — это НЕ
    «упрощённый режим», а сломанный режим:Overlap несвязанных слов в нём ВЫШЕ,
    чем синонимов (кошка~кот = 3, кошка~табуретка = 4), то есть вся логика
    узнавания в коре и памяти становится ложью. Молчаливый откат был самым
    опасным местом проекта: запуск системным python3 (где нет numpy) давал
    РАБОТАЮЩЕЕ приложение с неработающим смыслом.
    """
    global _ENCODER
    if force is not None:
        _ENCODER = force
        return force
    with _ENCODER_LOCK:
        if _ENCODER is not None:
            return _ENCODER
        want = _env("LIPID2_SDR_BACKEND", "embedding").strip().lower()
        if want not in ("embedding", "letterhash"):
            print(f"[SDR]: неизвестный LIPID2_SDR_BACKEND='{want}', использую embedding")
            want = "embedding"
        if want == "letterhash":
            # Явный выбор — не ошибка, но и не «нормальный режим»: пусть будет
            # видно, что система осознанно переведена в несемантический режим.
            _warn_letterhash("выбрано явно: LIPID2_SDR_BACKEND=letterhash", explicit=True)
            _ENCODER = LetterHashEncoder()
            return _ENCODER
        if want == "embedding" and HAS_NUMPY:
            try:
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _ENCODER = EmbeddingEncoder(
                    cache_path=os.path.join(root, "data", "memory", "sdr_calibration.pkl"),
                    lazy=False)
                return _ENCODER
            except EncodeError as exc:
                reason = str(exc)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
            _warn_letterhash(reason)
        elif want == "embedding":
            _warn_letterhash("нет numpy/torch в этом интерпретаторе")
        _ENCODER = LetterHashEncoder()
        return _ENCODER


def letterhash_reason() -> str:
    """Почему сейчас буквенный хэш — для GUI и диагностики, без угадывания."""
    return _FALLBACK_REASON


def _warn_letterhash(reason: str, explicit: bool = False):
    """Один раз на процесс, но ГРОМКО: см. docstring get_encoder."""
    global _FALLBACK_ANNOUNCED, _FALLBACK_REASON
    _FALLBACK_REASON = reason
    if explicit:
        # Явный выбор не печатаем как аварию, но и молчать нельзя.
        print(f"[SDR]: выбран буквенный хэш ({reason}). Смысл НЕ кодируется.")
        return
    if _FALLBACK_ANNOUNCED:
        return
    _FALLBACK_ANNOUNCED = True
    import sys
    sys.stderr.write(
        "\n" + "!" * 74 + "\n"
        "⚠️  SDR РАБОТАЕТ НА БУКВЕННОМ ХЭШЕ — СМЫСЛ НЕ КОДИРУЕТСЯ\n"
        f"   причина: {reason}\n"
        "   overlap у несвязанных слов будет ВЫШЕ, чем у синонимов\n"
        "   (кошка~кот = 3, кошка~табуретка = 4). Узнавание, bootstrap и\n"
        "   пороги в этом режиме бессмысленны, но приложение запустится.\n"
        "   Запускайте окружение проекта:\n"
        "       .venv/bin/python run.py\n"
        "   (в системном python3 нет numpy/torch/sentence-transformers)\n"
        + "!" * 74 + "\n")


_FALLBACK_ANNOUNCED = False
_FALLBACK_REASON = "не инициализирован"


def reset_encoder():
    """Только для тестов: сброс singleton."""
    global _ENCODER
    with _ENCODER_LOCK:
        if _ENCODER is not None and isinstance(_ENCODER, EmbeddingEncoder):
            try:
                _ENCODER.flush()
            except Exception:
                pass
        _ENCODER = None
