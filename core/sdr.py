"""
Модуль Восприятия (SDR-Парсер) для проекта Липид-2.

1024-битный разреженный вектор, ровно 20 активных бит (~2% плотности),
хранится как 16 слов uint64 — popcount/AND остаются дешёвыми.

ЧТО ИЗМЕНИЛОСЬ И ПОЧЕМУ
------------------------
Раньше биты строились из SHA-256 буквенных 2-/3-грамм. Это хэш НАПИСАНИЯ:

    кошка ~ кот        overlap  3/20
    кошка ~ табуретка  overlap  4/20      <-- больше, чем с «кот»
    грусть ~ любовь    overlap  2/20
    фон 820 реальных пар: среднее 0.68, максимум 6

А весь downstream (_recall, «узнал/не узнал», bootstrap по UNKNOWN_THRESHOLD)
трактовал overlap как СЕМАНТИЧЕСКОЕ узнавание. Он им не был и быть не мог:
это лечится только сменой кодировщика, не порогом.

Теперь кодировщик подключается из core/encoder.py (по умолчанию — эмбеддинговый,
семантический). Интерфейс класса не изменился: SDR1024(text), .overlap(),
.similarity(), .bits, .active_bit_count(). Битый хэш доступен как
LIPID2_SDR_BACKEND=letterhash — он нужен тестам как точка отсчёта.

Заглавный кэш заполняется пачкой через prewarm(), иначе граф из 10^4 узлов
перебирает модель по слову.
"""
import threading
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.encoder import (LetterHashEncoder, BaseEncoder, TARGET_ON_BITS,
                          TOTAL_BITS, NUM_U64, get_encoder)

# Переизлучаем константы: старый код ссылается на SDR1024.TOTAL_BITS и т.п.
__all__ = ["SDR1024", "prewarm", "encoder_info"]


class SDR1024:
    """
    Один SDR концепта. Биты отдаёт выбранный кодировщик; сам класс — только
    обёртка над popcount-арифметикой и кэшем.
    """
    TOTAL_BITS = TOTAL_BITS
    NUM_U64 = NUM_U64
    TARGET_ON_BITS = TARGET_ON_BITS

    # Общий кэш text -> SDR1024. Кодировщик детерминирован на время процесса,
    # поэтому кэш между узлами графа безопасен.
    _CACHE: Dict[str, "SDR1024"] = {}
    _CACHE_LOCK = threading.RLock()

    def __init__(self, text: str = "", encoder: Optional[BaseEncoder] = None):
        self.raw_text = (text or "").strip().lower()
        self.bits: List[int] = [0] * self.NUM_U64

        if not self.raw_text:
            return

        enc = encoder or get_encoder()
        # Повторное SDR1024("кошка") обязано вернуть тот же вектор — иначе
        # рассинхронизируются кэш графа и скомпилированные .lipid.
        cached = self._CACHE.get(self.raw_text)
        if cached is not None and (encoder is None or encoder is cached._encoder):
            self.bits = cached.bits
            self._encoder = cached._encoder
            return

        mask = enc.encode(self.raw_text)
        self.bits = self._mask_to_u64(mask)
        self._encoder = enc
        if encoder is None:
            with self._CACHE_LOCK:
                self._CACHE[self.raw_text] = self

    # ── построение ───────────────────────────────────────────
    @staticmethod
    def _mask_to_u64(mask) -> List[int]:
        try:
            import numpy as np
            idx = np.flatnonzero(mask)
        except ImportError:
            idx = [i for i, b in enumerate(mask) if b]
        bits = [0] * SDR1024.NUM_U64
        for b in idx:
            b = int(b)
            bits[b // 64] |= (1 << (b % 64))
        return bits

    # ── арифметика ───────────────────────────────────────────
    def overlap(self, other: "SDR1024") -> int:
        """Побитовый AND + popcount: число совпавших единичных бит."""
        shared = 0
        for i in range(self.NUM_U64):
            shared += bin(self.bits[i] & other.bits[i]).count("1")
        return shared

    def similarity(self, other: "SDR1024") -> float:
        return self.overlap(other) / float(self.TARGET_ON_BITS)

    def active_bit_count(self) -> int:
        return sum(bin(block).count("1") for block in self.bits)

    def active_indices(self) -> List[int]:
        out = []
        for w, blk in enumerate(self.bits):
            for k in range(64):
                if blk >> k & 1:
                    out.append(w * 64 + k)
        return out

    def __eq__(self, other):
        return isinstance(other, SDR1024) and self.bits == other.bits

    def __hash__(self):
        return hash(tuple(self.bits))

    def __repr__(self):
        kind = getattr(getattr(self, "_encoder", None), "name", "?")
        return (f"<SDR1024 text='{self.raw_text}' "
                f"active_bits={self.active_bit_count()} enc={kind}>")


# ══════════════════════════════════════════════════════════════════════
#  ПАЧЕЧНЫЙ ПРОГРЕВ
# ══════════════════════════════════════════════════════════════════════
def prewarm(texts: Iterable[str]) -> int:
    """
    Кодирует пачкой и кладёт в кэш. Возвращает число новых SDR.
    Без этого загрузка графа на 10^4 узлов = 10^4 вызовов модели по одному.
    """
    words = list(dict.fromkeys((t or "").strip().lower() for t in texts))
    words = [w for w in words if w]
    todo = [w for w in words if w not in SDR1024._CACHE]
    if not todo:
        return 0
    enc = get_encoder()
    try:
        masks = enc.encode_many(todo)
    except Exception as exc:                            # не роняем загрузку графа
        print(f"[SDR] пачечный прогрев не удался ({exc}); буду кодировать по слову")
        return 0
    with SDR1024._CACHE_LOCK:
        for w, m in zip(todo, masks):
            s = SDR1024.__new__(SDR1024)
            s.raw_text = w
            s.bits = SDR1024._mask_to_u64(m)
            s._encoder = enc
            SDR1024._CACHE[w] = s
    try:
        enc.flush()
    except Exception:
        pass
    return len(todo)


def encoder_info() -> Tuple[str, bool, str]:
    """(имя кодировщика, семантический ли, хэш конфигурации) — для логов и GUI."""
    enc = get_encoder()
    return enc.name, enc.semantic, enc.config_hash()


def clear_cache():
    """Только для тестов."""
    with SDR1024._CACHE_LOCK:
        SDR1024._CACHE.clear()


# ==============================================================================
# БЫСТРАЯ САМОПРОВЕРКА:  python3 -m core.sdr
# ==============================================================================
if __name__ == "__main__":
    name, semantic, cfg = encoder_info()
    print("--- ТЕСТИРОВАНИЕ МОДУЛЯ SDR1024 (ЛИПИД-2) ---")
    print(f"Кодировщик: {name} (семантический={semantic}, cfg={cfg})\n")

    groups = {
        "СИНОНИМЫ": [("кошка", "кот"), ("грусть", "печаль"), ("храбрый", "смелый")],
        "ОРФО-двойники (смысл разный)": [("кот", "код"), ("дом", "том")],
        "НЕ СВЯЗАНЫ": [("кошка", "табуретка"), ("любовь", "гвоздь"), ("стол", "река")],
    }
    prewarm([w for gs in groups.values() for p in gs for w in p])
    for title, pairs in groups.items():
        print(f"  {title}:")
        for a, b in pairs:
            sa, sb = SDR1024(a), SDR1024(b)
            ov = sa.overlap(sb)
            bar = "█" * ov + "░" * (20 - ov)
            print(f"    {a:9s} ~ {b:11s} {bar} {ov:2d}/20")

    long_word = "сельскохозяйственно-машиностроительный"
    sdr_long = SDR1024(long_word)
    print(f"\n  Длинное слово '{long_word}': {sdr_long}")
    assert sdr_long.active_bit_count() == TARGET_ON_BITS, "разреженность поехала"
    print("  Разреженность 20/1024 держится — OK")
