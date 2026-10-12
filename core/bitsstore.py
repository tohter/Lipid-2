"""
core/bitsstore.py — побитовый кэш SDR в ПЕРЕНОСИМОМ бинарном формате.

ЗАЧЕМ, ЕСЛИ УЖЕ ЕСТЬ sdr_calibration.pkl
----------------------------------------
Калибровочный файл хранил float32-эмбеддинги 30 055 слов — 44 МБ из 47.9 МБ.
Онлайн-ядру (и будущему C-коду) эти векторы НЕ НУЖНЫ: ему нужны би́ты, а это
128 байт на слово вместо 1536. Плюс pickle — это Python: перенести его в C
нельзя ни при каких условиях, а договорённость проекта (memory.txt, раздел
«КОМПИЛИРУЕМОЕ ЯДРО») требует ЯВНОГО layout для всего переносимого.

ФОРМАТ = КОНТРАКТ (не побочный продукт сериализатора)
-----------------------------------------------------
little-endian, фиксированные ширины, длина перед каждой строкой, UTF-8:

    offset  size   поле
    0       4      magic  b"L2SB"
    4       1      version (u8) = 1
    5       1      on_bits (u8)          — сколько активных бит в записи
    6       2      total_bits (u16)      — 1024
    8       16     encoder cfg_hash      — ASCII, 0-дополненный
    24      4      count (u32)           — число записей
    28      ...    записи:
                     u16  len_word_bytes
                     bytes word (UTF-8, уже нормализованный: lower/strip)
                     128 bytes  bits      = 16 × u64 little-endian

Би́ты хранятся как 16 слов uint64 — ровно то, что держит SDR1024.bits, и ровно
то, что в C читается одним memcpy. overlap = AND + popcount без единого
преобразования.

ЗАЧЕМ cfg-хэш ВНУТРИ ФАЙЛА
--------------------------
Би́ты вне своего кодировщика бессмысленны: те же 20 позиций, полученные другой
моделью, — ДРУГОЕ знание. Хэш проверяется при чтении, несовпадение — ГРОМКИЙ
отказ с причиной (файл выбрасывается, всё кодируется заново), а не тихая
подстановка. Это то же правило, что уже стоит в core/storage.py для .lipid.

ЦЕНОЙ КАКОГО КОМПРОМИССА ЭТО ДАЁТСЯ
-----------------------------------
Файл побитово зависим от версии кодировщика и бесполезен без него. Поэтому он
в .gitignore: артефакт, регенерируется, в VCS ему не место.
"""
from __future__ import annotations

import os
import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MAGIC = b"L2SB"
VERSION = 1
HEADER = struct.Struct("<4sBBH16sI")       # magic, version, on_bits, total, cfg, count
WORD_LEN = struct.Struct("<H")
U64S = struct.Struct("<16Q")

BITS_BYTES = 128                          # 16 × uint64


class BitsStoreError(RuntimeError):
    pass


def default_path(root: str = None) -> str:
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "memory", "sdr_words.bin")


def pack_bits(active: Iterable[int]) -> bytes:
    """Индексы активных бит -> 16 слов uint64 (little-endian), 128 байт."""
    words = [0] * 16
    for b in active:
        b = int(b)
        if not 0 <= b < 1024:
            raise BitsStoreError(f"бит {b} вне 0..1023")
        words[b // 64] |= (1 << (b % 64))
    return U64S.pack(*words)


def unpack_bits(blob: bytes) -> List[int]:
    """128 байт -> список индексов активных бит (отсортированный)."""
    if len(blob) != BITS_BYTES:
        raise BitsStoreError(f"размер записи {len(blob)} != {BITS_BYTES}")
    words = U64S.unpack(blob)
    out: List[int] = []
    for w, blk in enumerate(words):
        for k in range(64):
            if blk >> k & 1:
                out.append(w * 64 + k)
    return out


def overlap_bytes(a: bytes, b: bytes) -> int:
    """
    overlap ДВУМ записям без распаковки в mask — ровно то, что делает C:
    побитовый И по 16 словам + popcount. Нужен как эталон в тестах: если
    Python-реализация и распаковка через mask расходятся, формат неверен.
    """
    if len(a) != BITS_BYTES or len(b) != BITS_BYTES:
        raise BitsStoreError("overlap: размер записи должен быть 128 байт")
    total = 0
    for x, y in zip(U64S.unpack(a), U64S.unpack(b)):
        total += bin(x & y).count("1")
    return total


class BitsStore:
    """
    Кэш «слово -> 20 активных бит» с явным layout и проверкой кодировщика.

    Использование:
        st = BitsStore(cfg_hash="d39733936ebd", on_bits=20)
        st.load()                       # несовпадение cfg -> тихо пусто + reason
        st.get("кошка")                 # bytes(128) или None
        st.put("кошка", bits128)
        st.flush()                      # атомарно: temp + os.replace

    Файл — производный артефакт. Любой сбой чтения означает «кэша нет», а не
    аварию, НО причину молча проглатывать нельзя: store.reason объясняет, почему
    кодировщик будет работать без ускорения (или почему би́ты пересоздаются).
    """

    def __init__(self, cfg_hash: str, on_bits: int = 20, total_bits: int = 1024,
                 path: str = None):
        self.cfg_hash = (cfg_hash or "")[:16]
        self.on_bits = int(on_bits)
        self.total_bits = int(total_bits)
        self.path = path or default_path()
        self._map: Dict[str, bytes] = {}
        self._dirty = False
        self.reason: Optional[str] = None     # почему кэш пуст/не тот
        self.loaded_from: Optional[str] = None

    # ── чтение ──────────────────────────────────────────────
    def load(self) -> int:
        if not os.path.exists(self.path):
            self.reason = "файла ещё нет"
            return 0
        try:
            with open(self.path, "rb") as fh:
                blob = fh.read()
            n = self._parse(blob)
        except BitsStoreError as exc:
            # НЕСОВПАДЕНИЕ КОДИРОВЩИКА — рядовая ситуация (сменили модель или
            # версию библиотеки), а не авария. Но и молчать нельзя: иначе
            # «кэш не работает» выглядит как «система тормозит без причины».
            self.reason = str(exc)
            self._map = {}
            return 0
        except Exception as exc:
            self.reason = f"не читается: {type(exc).__name__}: {exc}"
            self._map = {}
            return 0
        self.loaded_from = self.path
        self.reason = None
        return n

    def _parse(self, blob: bytes) -> int:
        if len(blob) < HEADER.size:
            raise BitsStoreError("файл короче заголовка")
        magic, ver, on_bits, total_bits, cfg_raw, count = HEADER.unpack_from(blob, 0)
        if magic != MAGIC:
            raise BitsStoreError(f"magic {magic!r} != {MAGIC!r} — это не sdr_words.bin")
        if ver != VERSION:
            raise BitsStoreError(f"версия формата {ver}, ожидается {VERSION}")
        cfg = cfg_raw.rstrip(b"\x00").decode("ascii", "replace")
        if cfg != self.cfg_hash:
            raise BitsStoreError(
                f"биты построены ДРУГИМ кодировщиком (cfg={cfg}), сейчас "
                f"{self.cfg_hash}: би́ты вне своего кодировщика бессмысленны, "
                f"кэш отброшен, всё будет перекодировано")
        if on_bits != self.on_bits:
            raise BitsStoreError(
                f"в файле {on_bits} активных бит, сейчас {self.on_bits} — "
                f"инвариант архитектуры поехал")
        if total_bits != self.total_bits:
            raise BitsStoreError(
                f"в файле {total_bits} бит, сейчас {self.total_bits}")
        out: Dict[str, bytes] = {}
        pos = HEADER.size
        for i in range(count):
            if pos + WORD_LEN.size > len(blob):
                raise BitsStoreError(f"обрезан на записи {i} из {count}")
            (wl,) = WORD_LEN.unpack_from(blob, pos)
            pos += WORD_LEN.size
            if pos + wl + BITS_BYTES > len(blob):
                raise BitsStoreError(f"обрезана запись {i} (нет {wl}+{BITS_BYTES} байт)")
            word = blob[pos:pos + wl].decode("utf-8", "replace")
            pos += wl
            bits = blob[pos:pos + BITS_BYTES]
            pos += BITS_BYTES
            out[word] = bits
        if pos != len(blob):
            raise BitsStoreError(f"в хвосте файла {len(blob) - pos} лишних байт")
        self._map = out
        return len(out)

    # ── доступ ──────────────────────────────────────────────
    def get(self, word: str) -> Optional[bytes]:
        return self._map.get(word)

    def __contains__(self, word: str) -> bool:
        return word in self._map

    def __len__(self) -> int:
        return len(self._map)

    def put(self, word: str, bits: bytes) -> None:
        if len(bits) != BITS_BYTES:
            raise BitsStoreError(f"put: размер {len(bits)} != {BITS_BYTES}")
        if self._map.get(word) == bits:
            return
        self._map[word] = bits
        self._dirty = True

    def put_many(self, pairs: Sequence[Tuple[str, bytes]]) -> int:
        n = 0
        for w, b in pairs:
            self.put(w, b)
            n += 1
        return n

    # ── запись ──────────────────────────────────────────────
    def to_bytes(self) -> bytes:
        parts = [HEADER.pack(MAGIC, VERSION, self.on_bits, self.total_bits,
                             self.cfg_hash.encode("ascii")[:16].ljust(16, b"\x00"),
                             len(self._map))]
        for word in sorted(self._map):
            raw = word.encode("utf-8")
            if len(raw) > 0xFFFF:
                raise BitsStoreError(f"слово длиной {len(raw)} байт не влезет в u16")
            parts.append(WORD_LEN.pack(len(raw)))
            parts.append(raw)
            parts.append(self._map[word])
        return b"".join(parts)

    def flush(self, force: bool = False) -> bool:
        """
        АТОМДАРНО: temp + os.replace. Полуполный файл после обрыва питания
        обязан читаться как «кэша нет» (проверка хвоста в _parse), но лучше
        вообще его не оставлять.
        """
        if not self._dirty and not force:
            return False
        d = os.path.dirname(self.path)
        try:
            os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(self.to_bytes())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            self.reason = f"не записан: {type(exc).__name__}: {exc}"
            return False
        self._dirty = False
        return True

    def __repr__(self):
        state = "empty" if not self._map else f"{len(self._map)} слов"
        why = f", reason={self.reason}" if self.reason else ""
        return f"<BitsStore cfg={self.cfg_hash} {state}{why}>"
