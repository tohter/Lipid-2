"""
core/transitions.py — Сила ПЕРЕХОДА «концепт A -> концепт B». Отдельная величина.

ПОЧЕМУ ЭТОТ ФАЙЛ ВО ВСЕ СОСТОИТ (замер 2026-07-13, а не «архитектурно правильно»)
----------------------------------------------------------------------------------
В проекте ОДНО хранилище связей — Node.out_synapses_typed[..]["weight"] — и в
нём живёт величина с ДВУМЯ разными смыслами:

  1. ВЕС = УВЕРЕННОСТЬ в факте. Так его трактует brain/learner.py:
         CAN_DO/HAS_PROPERTY = 90+130*conf, прочие = 60+100*conf,
         повторное свидетельство НЕ ПОНИЖАЕТ вес (правило «вес никогда не
         понижаем», learner._connect).
  2. ЧАСТОТА ПЕРЕХОДА. Так его же трактует предиктивный цикл core/cortex.py:
         попадание -> +дофамин()/4, промах -> -кортизол()/2.

Пока цикл был отключён, со конфликтом никто не встречался. стоило его подключить
— цикл начал ПЕРЕПИСЫВАТЬ УВЕРЕННОСТЬ онтологии. ЗАМЕРЕНО на связи
«кошка -> кошачьи» (IS_A, вес 180 = онтология из Вселенной):

    один промах цикла  ->  180 -> 116     (уверенность упала на 36%)
    одно попадание     ->  116 -> 231

После этого вес перестаёт читаться как уверенность: по числу 116 нельзя сказать,
«факт сомнительный» или «просто чаще не угадывали следующее слово». Это тихая
порча знаний ровно того класса, который проект считает багом.

РЕШЕНИЕ: РАЗНЫЕ ВЕЩИ ХРАНЯТСЯ РАЗНЫМИ ЧИСЛАМИ.
  weight      -> уверенность в факте. Пишут: learner, merge_entity, wiki, LLM.
  transition  -> «насколько B ожидаемо после A». Пишет: только предиктивный цикл.

ЧЕМ ПЕРЕХОД НЕ ЯВЛЯЕТСЯ, нарочно:
  * НЕ отрицательное знание: «не было» превращает силу в 0, знак минуса ниже
    нуля не уходит (для отрицания у наблюдения есть отдельный флаг negated);
  * НЕ замена наблюдению: наблюдение — факт из текста с evidence, переход —
    обобщённая статистика ожиданий. Одно без другого неполно.

ПЛОТНОСТЬ И ПЕРЕНОСИМОСТЬ
-------------------------
Сила — saturating-счётчик 4 бита (0..15): «в тысячный раз подтвердить переход»
смысла нет, а неограниченный int не переносится в C.
  * память: 4 перехода на байт. 100 000 переходов = 25 КБ против ~979 МБ у
    словаря Python-объектов (замер дельты RSS на узел: 9.79 КБ, memory.txt);
  * НУМЕРА СЛОТОВ ЛОКАЛЬНЫ ДЛЯ КАЖДОГО head. Это не микрооптимизация, а
    исправление собственного дефекта первой версии: при ГЛОБАЛЬНЫХ номерах
    хвостов рядок каждого head растягивался до размера всего словаря (100 000
    хвостов -> 25 КБ на head вместо ~1 байта на его 4 перехода), то есть
    разреженная таблица молча становилась плотной. Рядок теперь =
    ceil(своих_хвостов/2), ни байтом больше;
  * в сериализацию (JSON и бинарный образ) номера НЕ попадают — только ИМЕНА и
    силы. Порядок вставки не влияет на байты, ключом знания остаётся текст
    (memory.txt: «НЕ терять ключи: имя узла обязано оставаться текстом»).
"""
from __future__ import annotations

import struct
import sys
from typing import Dict, Iterable, List, Optional, Tuple

MAGIC = b"LPD2TX"
VERSION = 1
PACK = 2              # переходов в байте: 4 бита на силу, потолок 15
BITS = 8 // PACK      # бит на силу: в байте ВОСЕМЬ бит, не двести пятьдесят
MAX_STRENGTH = (1 << BITS) - 1


def _complain(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")
    sys.stderr.flush()


class _Row:
    """Рядок одного head: имена хвостов + упакованные силы. Нумерация ЛОКАЛЬНА."""

    def __init__(self, pack: int = PACK):
        self.pack = pack
        self.bits = 8 // pack
        self.names: List[str] = []      # индекс -> имя (порядок = порядок хранения)
        self.tails: Dict[str, int] = {}  # имя -> локальный индекс
        self.data = bytearray()

    def index(self, tail: str) -> Tuple[int, bool]:
        """Локальный индекс хвоста и признак «заведён только что\"."""
        i = self.tails.get(tail)
        if i is not None:
            return i, False
        i = len(self.names)
        self.tails[tail] = i
        self.names.append(tail)
        need = i // self.pack + 1
        if need > len(self.data):
            self.data.extend(bytes(need - len(self.data)))
        return i, True


class TransitionTable:
    """
    Разреженная таблица «head -> (tail -> сила 0..15)», 4 бита на переход.

    ПУБЛИЧНЫЙ КОНТРАКТ КЛАССА: get/confirm/surprise/top работают по ИМЕНАМ и
    ничего не знают о упаковщике. Всё, что вне класса, обязано читать имена из
    to_json(), а не вычислять байты руками — иначе упаковщик перестанет быть
    деталью хранения и станет второй схемой данных.
    """

    def __init__(self, pack: int = PACK):
        if pack < 1 or 8 % pack:
            raise ValueError(f"pack={pack}: в байте 8 бит, "
                             f"число переходов должно делить его нацело")
        self.pack = pack
        self.bits = 8 // pack
        self.max_strength = (1 << self.bits) - 1
        self._rows: Dict[str, _Row] = {}
        # Сколько записей с ненулевой силой. Публично: отчёт и тесты обязаны
        # видеть объём знания числом, а не «по виду журнала».
        self.entries = 0
        # Пусто = таблица цела. Непусто = ЧТО ИМЕННО не прочитано.
        self.degraded_reason = ""

    # ── ЧТЕНИЕ ────────────────────────────────────────────────────────────
    def get(self, head: str, tail: str) -> int:
        """Сила перехода 0..max_strength. 0 = «перехода нет» или «его стёрли»."""
        row = self._rows.get(head)
        if row is None:
            return 0
        i = row.tails.get(tail)
        if i is None:
            return 0
        idx, shift = divmod(i, row.pack)
        if idx >= len(row.data):
            return 0
        return (row.data[idx] >> (row.bits * shift)) & self.max_strength

    def strength(self, head: str, tail: str) -> float:
        """Нормированная сила 0.0..1.0 — для отчётов и порога «ожидалось»."""
        return self.get(head, tail) / float(self.max_strength)

    def has(self, head: str, tail: str) -> bool:
        return self.get(head, tail) > 0

    def known_tails(self, head: str) -> List[str]:
        """Имена хвостов с ненулевой силой, сила DESC затем имя ASC."""
        row = self._rows.get(head)
        if row is None:
            return []
        pairs = [(name, self.get(head, name)) for name in row.names]
        pairs = [(n, s) for n, s in pairs if s > 0]
        # Tie-break по алфавиту ОБЯЗАТЕЛЕН: предсказание обязано быть одним и
        # тем же при одном состоянии. Порядок вставки словаря Python разный на
        # разных прогонах, и без сортировки «точность цикла» зависела бы от него.
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return [n for n, _ in pairs]

    def top(self, head: str, k: int = 1) -> List[Tuple[str, int]]:
        return [(t, self.get(head, t)) for t in self.known_tails(head)[:k]]

    def top_name(self, head: str) -> Optional[str]:
        t = self.top(head, 1)
        return t[0][0] if t else None

    def heads(self) -> Iterable[str]:
        return sorted(self._rows)

    def size(self) -> int:
        return self.entries

    def stats(self) -> dict:
        """Публичные числа: без них «таблица работает» нечем проверить."""
        by_strength = {s: 0 for s in range(self.max_strength + 1)}
        for head, row in self._rows.items():
            for name in row.names:
                by_strength[self.get(head, name)] += 1
        return {
            "heads": len(self._rows),
            "entries": self.entries,
            "bytes": sum(len(r.data) for r in self._rows.values()),
            "by_strength": by_strength,
            "degraded": self.degraded_reason,
        }

    # ── ЗАПИСЬ (единственный писатель — предиктивный цикл) ───────────────
    def _cell(self, head: str, tail: str) -> Tuple[_Row, int, int]:
        row = self._rows.get(head)
        if row is None:
            row = self._rows[head] = _Row(self.pack)
        i, _fresh = row.index(tail)
        idx, shift = divmod(i, row.pack)
        return row, idx, shift

    def confirm(self, head: str, tail: str, delta: int = 1) -> int:
        """Переход подтвердился: сила +delta с ПОТОЛКОМ. Возвращает новую силу."""
        return self._adjust(head, tail, delta)

    def surprise(self, head: str, tail: str, delta: int = 1) -> int:
        """
        Переход НЕ подтвердился: сила -delta с ПОЛОМ 0.

        Пол 0 = «переход стёрт», а НЕ «отрицательный опыт»: отрицательной
        уверенности в проекте нет, есть отдельный флаг negated у наблюдения.
        """
        return self._adjust(head, tail, -delta)

    def _adjust(self, head: str, tail: str, delta: int) -> int:
        if not head or not tail or not delta:
            return self.get(head, tail)
        row, idx, shift = self._cell(head, tail)
        mask = self.max_strength << (self.bits * shift)
        cur = (row.data[idx] & mask) >> (self.bits * shift)
        new = max(0, min(self.max_strength, cur + delta))
        row.data[idx] = (row.data[idx] & ~mask | (new << (self.bits * shift))) & 0xFF
        if cur == 0 and new > 0:
            self.entries += 1
        elif cur > 0 and new == 0:
            # Запись стёрта (сила 0), но имя остаётся до compact(): перенумерация
            # на каждый сюрприз дороже сэкономленного байта, а сдвиг индексов
            # перемешал бы силы соседей. Счётчик entries при этом честный:
            # «переходов» показывает ЗНАНИЕ, а не занятое место.
            self.entries = max(0, self.entries - 1)
        return new

    def compact(self) -> int:
        """
        Выбрасывает нулевые переходы и упаковывает рядки заново. Возвращает
        число удалённых записей. Вызывается пачкой (после загрузки/обучения),
        не на каждый сюрприз: перенумерация дороже сэкономленного байта.
        """
        dropped = 0
        for head in list(self._rows):
            row = self._rows[head]
            keep = [n for n in row.names if self.get(head, n) > 0]
            if len(keep) == len(row.names):
                continue
            dropped += len(row.names) - len(keep)
            fresh = _Row(self.pack)
            strengths = {n: self.get(head, n) for n in keep}
            for name in sorted(keep):
                i, _ = fresh.index(name)
                idx, shift = divmod(i, fresh.pack)
                fresh.data[idx] |= (strengths[name] & self.max_strength) << (
                    fresh.bits * shift)
            if fresh.names:
                self._rows[head] = fresh
            else:
                del self._rows[head]
        return dropped

    # ── СЕРИАЛИЗАЦИЯ В МИР (JSON — читается человеком, memory.txt п.15) ──
    def to_json(self) -> dict:
        """
        {"version":1,"pack":2,"rows":[{"head":"кошка",
                                       "tails":["молоко","подоконник"],
                                       "strengths":[7,3]}]}
        Два выровненных массива вместо словаря: имя остаётся ТЕКСТОМ, силы
        читаются глазами построчно. head — по алфавиту, хвосты — сила DESC затем
        имя: две записи одного состояния дают ПОБАЙТОВО одинаковый файл, иначе
        git показывает диффы там, где данных не меняли.
        """
        rows = []
        for head in sorted(self._rows):
            pairs = self.top(head, k=len(self._rows[head].names) + 1)
            if not pairs:
                continue
            rows.append({"head": head,
                         "tails": [t for t, _ in pairs],
                         "strengths": [s for _, s in pairs]})
        return {"version": VERSION, "pack": self.pack, "rows": rows}

    def from_json(self, data: dict) -> "TransitionTable":
        """
        Загрузка. Любое несоответствие формата — ГРОМКОЕ и с флагом, а не
        молчаливая пустота: «не смог прочитать» и «переходов ещё нет» — РАЗНЫЕ
        статусы (урок проекта: asked_empty != not_asked).
        """
        self._reset()
        if data in (None, {}, ""):
            return self                       # «ещё не учились» — это не поломка
        if not isinstance(data, dict):
            self._degrade(f"ожидался объект, получен {type(data).__name__}")
            return self
        ver = data.get("version")
        if ver != VERSION:
            self._degrade(f"версия таблицы {ver!r}, ожидаю {VERSION} — НЕ читаю")
            return self
        pack = data.get("pack", self.pack)
        if pack != self.pack:
            self._degrade(f"плотность pack={pack}, у меня {self.pack} — НЕ читаю")
            return self
        for row in data.get("rows") or []:
            if not isinstance(row, dict):
                self._degrade(f"строка таблицы {type(row).__name__}, жду объект")
                continue
            head = row.get("head")
            tails = row.get("tails")
            strengths = row.get("strengths")
            if not isinstance(head, str) or not isinstance(tails, list) \
                    or not isinstance(strengths, list):
                self._degrade(f"строка без head/tails/strengths: {str(row)[:60]}")
                continue
            if len(tails) != len(strengths):
                self._degrade(f"head «{head}»: хвостов {len(tails)}, сил "
                              f"{len(strengths)} — строка пропущена")
                continue
            for tail, s in zip(tails, strengths):
                if not isinstance(tail, str) or not tail:
                    self._degrade(f"«{head}\": хвост {tail!r} не строка")
                    continue
                try:
                    s = int(s)
                except (TypeError, ValueError):
                    self._degrade(f"«{head}->{tail}»: сила {s!r} не число")
                    continue
                if s <= 0:
                    continue
                if s > self.max_strength:
                    # Молча срезать до потолка = принять чужие данные за свои:
                    # сила 99 означает «кто-то писал в таблицу не understanding
                    # шкалой», и узнать об этом надо сейчас, а не после
                    # перезаписи файла.
                    self._degrade(f"«{head}->{tail}»: сила {s} выше потолка "
                                  f"{self.max_strength} — записана как потолок")
                self.confirm(head, tail, min(s, self.max_strength))
        return self

    def _reset(self) -> None:
        self._rows.clear()
        self.entries = 0
        self.degraded_reason = ""

    def _degrade(self, text: str) -> None:
        """Деградация громкая И запомненная: отчёт обязан её показать."""
        if not self.degraded_reason:
            self.degraded_reason = text       # первая причина — настоящая
        _complain(f"[transitions] ⛔ {text}")

    # ── ПЕРЕНОСИМЫЙ БИНАРНЫЙ ОБРАЗ (для компилируемого ядра) ─────────────
    def encode(self) -> bytes:
        """
        LPD2TX | uint8 version | uint8 pack | uint32 n_heads |
          per head: uint32 len + UTF-8 head, uint32 n_tails,
                    per tail: uint32 len + UTF-8 tail, uint8 strength
        Номера слотов в образ НЕ попадают: только имена и силы. Тот же образ
        читается на C без знания нашего packing.
        """
        out = bytearray(MAGIC)
        out += struct.pack("<BBI", VERSION, self.pack, len(self._rows))
        for head in sorted(self._rows):
            hb = head.encode("utf-8")
            out += struct.pack("<I", len(hb)) + hb
            pairs = sorted((t, self.get(head, t)) for t in self.known_tails(head))
            out += struct.pack("<I", len(pairs))
            for tail, s in pairs:
                tb = tail.encode("utf-8")
                out += struct.pack("<I", len(tb)) + tb + struct.pack("<B", s)
        return bytes(out)

    def decode(self, blob: bytes) -> "TransitionTable":
        self._reset()
        if not blob:
            return self
        if blob[:len(MAGIC)] != MAGIC:
            self._degrade("образ переходов без сигнатуры LPD2TX — НЕ читаю")
            return self
        off = len(MAGIC)
        if off + 6 > len(blob):
            self._degrade("заголовок образа обрезан")
            return self
        ver, pack, n_heads = struct.unpack_from("<BBI", blob, off)
        off += 6
        if ver != VERSION or pack != self.pack:
            self._degrade(f"образ: version={ver} pack={pack}, жду "
                          f"{VERSION}/{self.pack}")
            return self
        for _ in range(n_heads):
            head, off = self._read_str(blob, off)
            if head is None:
                self._degrade("образ оборвался на имени head")
                return self
            if off + 4 > len(blob):
                self._degrade("образ оборвался на числе хвостов")
                return self
            count = struct.unpack_from("<I", blob, off)[0]
            off += 4
            for _ in range(count):
                tail, off = self._read_str(blob, off)
                if tail is None or off + 1 > len(blob):
                    self._degrade(f"образ оборвался на паре «{head}»")
                    return self
                s = struct.unpack_from("<B", blob, off)[0]
                off += 1
                if s > self.max_strength:
                    # Молча вписать в шкалу чужое число = выдать чужие данные
                    # за свой опыт (тот же дефект, что закрыт в from_json).
                    self._degrade(f"образ: сила {s} для «{head}->{tail}» выше "
                                  f"потолка {self.max_strength} — записана как "
                                  f"потолок, образ от другого кодера?")
                self.confirm(head, tail, max(1, min(int(s), self.max_strength)))
        return self

    @staticmethod
    def _read_str(blob: bytes, off: int) -> Tuple[Optional[str], int]:
        if off + 4 > len(blob):
            return None, off
        n = struct.unpack_from("<I", blob, off)[0]
        off2 = off + 4
        if off2 + n > len(blob):
            return None, off2
        return blob[off2:off2 + n].decode("utf-8", "replace"), off2 + n

    def __len__(self) -> int:
        return self.entries

    def __contains__(self, pair) -> bool:
        head, tail = pair
        return self.get(head, tail) > 0


def merge_json(base: Optional[dict], other: Optional[dict]) -> dict:
    """
    Слияние двух сериализованных таблиц ПО МАКСИМУМУ.

    ЗАЧЕМ: мир читают и пишут из нескольких мест (мозг, чистильщик, Учитель),
    а запись выглядит как «прочитал всё -> изменил -> записал всё». Обычное
    замешивание dict стёрло бы переходы, наработанные другим путём, а сила
    перехода имеет смысл ТОЛЬКО накопительный. Максимум — единственная операция,
    совместимая с таким порядком: результат не зависит от того, кто записался
    первым (коммутативно), и опыт от перезаписи не уменьшается.

    Кривые записи не роняют слияние, но и не исчезают молча: они считаются и
    объявляются в stderr. Слияние НЕ сокращает — урезать таблицу можно только
    явным compact(). Формат тот же, что пишет to_json(), чтобы merge можно было
    проверить сравнением файлов, а не только по объекту.
    """
    acc: Dict[str, Dict[str, int]] = {}
    dropped = 0
    for source in (base, other):
        if not isinstance(source, dict):
            continue
        for row in source.get("rows") or []:
            if not isinstance(row, dict):
                dropped += 1
                continue
            head = row.get("head")
            tails = row.get("tails")
            strengths = row.get("strengths")
            if not isinstance(head, str) or not isinstance(tails, list) \
                    or not isinstance(strengths, list) or len(tails) != len(strengths):
                dropped += 1
                continue
            target = acc.setdefault(head, {})
            for tail, strength in zip(tails, strengths):
                if not isinstance(tail, str) or not tail or isinstance(strength, bool) \
                        or not isinstance(strength, int) \
                        or not 0 < strength <= MAX_STRENGTH:
                    dropped += 1
                    continue
                target[tail] = max(target.get(tail, 0), strength)
    if dropped:
        _complain(f"[transitions] ⚠ при слиянии отброшено кривых записей: "
                  f"{dropped} (ожидается целое 1..{MAX_STRENGTH} и имена "
                  f"строками) — проверьте секцию transitions")
    rows = [{"head": head,
             "tails": [t for t, _ in sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))],
             "strengths": [s for _, s in sorted(vals.items(), key=lambda kv: (-kv[1], kv[0]))]}
            for head, vals in sorted(acc.items()) if vals]
    return {"version": VERSION, "pack": PACK, "rows": rows}



def table_from_world(world: dict) -> TransitionTable:
    """Таблица из секции мира. Пусто = «ещё не учились», а не «повреждено»."""
    t = TransitionTable()
    t.from_json((world or {}).get("transitions") or {})
    return t


if __name__ == "__main__":
    # Контрольный прогон: арифметика, потолок, память, переносимость.
    checks = []
    t = TransitionTable()
    for _ in range(1000):
        t.confirm("кошка", "подоконник")
    checks.append(("потолок 15 после 1000 подтверждений",
                   t.get("кошка", "подоконник") == MAX_STRENGTH))
    for _ in range(15):
        t.surprise("кошка", "подоконник")
    checks.append(("15 сюрпризов стёрли переход в 0",
                   t.get("кошка", "подоконник") == 0))
    for w, s in (("молоко", 3), ("подоконник", 3), ("спать", 5)):
        t.confirm("кошка", w, s)
    checks.append(("топ: сила DESC, затем имя",
                   t.top("кошка", 3) == [("спать", 5), ("молоко", 3),
                                         ("подоконник", 3)]))
    checks.append(("топ предсказуем при равных силах",
                   t.top_name("кошка") == "спать"))
    checks.append(("per-head компактность: 1 рядок = 2 байта на 4 хвоста",
                   t.stats()["bytes"] <= 2))
    checks.append(("entries считает ненулевые переходы", t.size() == 3))
    t.surprise("кошка", "спать", 5)
    checks.append(("сюрприз до нуля снимает запись из entries",
                   t.get("кошка", "спать") == 0 and t.size() == 2))
    dropped = t.compact()
    checks.append(("compact() выкидывает стёртый переход",
                   dropped == 1 and t.top_name("кошка") == "молоко"))

    t2 = TransitionTable().from_json(t.to_json())
    checks.append(("json round-trip сохраняет силы",
                   all(t2.get("кошка", w) == t.get("кошка", w)
                       for w in ("молоко", "подоконник"))))
    t3 = TransitionTable().decode(t.encode())
    checks.append(("бинарный round-trip сохраняет силы",
                   all(t3.get("кошка", w) == t.get("кошка", w)
                       for w in ("молоко", "подоконник"))))
    bad = TransitionTable().from_json({"version": 99, "rows": []})
    checks.append(("чужая версия — громкий флаг, а не пустота",
                   bool(bad.degraded_reason)))
    empty = TransitionTable().from_json({})
    checks.append(("пустая таблица НЕ считается повреждённой",
                   not empty.degraded_reason))
    ragged = TransitionTable().from_json(
        {"version": VERSION, "pack": PACK,
         "rows": [{"head": "x", "tails": ["a", "b"], "strengths": [3]}]})
    checks.append(("рваная строка — флаг, остальные читаются",
                   bool(ragged.degraded_reason)))

    ok = True
    for name, res in checks:
        print(f"  {'OK  ' if res else 'FAIL'} {name}")
        ok &= bool(res)
    print(f"\nпереходов: {t.size()}, packed-байт: {t.stats()['bytes']}, "
          f"образ: {len(t.encode())} байт")
    raise SystemExit(0 if ok else 1)
