"""
brain/sleep.py — SLEEP CYCLE: единственный код, где ОПЫТ получает право
править ВЕС онтологии (очередь memory.txt, п.23-доработка, P1).

ЧТО ТАКОЕ «СОН» В ЭТОМ ПРОЕКТЕ
------------------------------
Днём работают два кода с РАЗНЫМИ правами:
  * Learner читает текст и пишет ЗНАНИЕ (observations/entities) — но видит
    только то, что сказано словами;
  * PredictiveCortex наблюдает последовательности и копит СИЛУ ПЕРЕХОДА
    (core/transitions.py) — но не имеет права создать ни узла, ни факта
    (закрыто тестами tests/test_cortex_experience_contract.py).
Между «текст повторялся снова и снова» и «это знание» не было НИКАКОГО
мостика: цикл мог накопить силу 15/15 на переходе, а онтология не становилась
умнее НИКОГДА. Сон — этот мостик: он вызывается НЕ во время урока, а пачкой, и
только он имеет право переводить опыт в вес.

ПОЧЕМУ ПРАВА ВЫДАНЫ ЗДЕСЬ, а не разданы по коду
-----------------------------------------------
memory.txt: «это ЕДИНСТВЕННОЕ место, где опыт получит право править weight
онтологии — оформлять строго по контракту и с замером». Числа взяты из
experiments/PLASTICITY_CONTRACT.md (замер experiments/plasticity_sweep.json),
а не на глаз; каждое названо константой и повторено в тесте.

ТРИ ОПЕРАЦИИ СНА (и только три)
-------------------------------
1. создание — опыт СИЛЬНО ждёт (strength >= 8 из 15), онтологии НЕТ
   -> создать CAN_DO с весом 80 + 8*strength (144..200).
2. повышение — онтология есть, опыт её подтверждает -> вес + (strength-7)*10,
   но НЕ ВЫШЕ 200 (201+ значит «неоспоримо», а опыт неоспорим быть не может).
3. примирение — онтология УТВЕРЖДАЕТ (weight >= 150), а опыт почти не видел
   связи (strength <= 3) -> вес -30, но НЕ НИЖЕ 80 (базовое «может»).

ЗАПРЕЩЕНО и проверяется тестом: трогать что-либо кроме CAN_DO; трогать
ОТРИЦАННЫЕ наблюдения (опыт, ждущий связи, которой текст сказал «не делает», —
это противоречие, а не знание: рапортуется и оставляется на лицо); создавать
узлы; писать в entities; менять порядок «сначала консолидация, потом decay».

ЗАМЕР, ОПРЕДЕЛИВШИЙ КОНСТРУКЦИЮ СОХРАНЕНИЯ (важнее, чем сам код)
----------------------------------------------------------------
Brain сохраняет опыт через core.transitions.merge_json, а тот сливает ПО
МАКСИМУМУ (иначе соседний процесс стирал бы чужой опыт). Для накопления это
правильно, для ЗАТУХАНИЯ — убийственно. ЗАМЕРЕНО этой сессии:

    сила в памяти после decay:            5
    та же сила после merge_json при записи: 10     <-- decay ПРОПАЛ

то есть затухание, сохранённое обычным путём, отменяется собственным
сохранением: файл помнит бОльшую силу. Поэтому сон пишет секцию transitions
не «максимумом», а через reconcile_saved: наша затухшая сила берётся СВОЕЙ, но
чужой опыт (наработанный соседом поверх нашего снимка или вовсе не виденный
нами) НЕ откатывается. Окно «прочитали -> записали» сужено транзакцией (тот же
_LOCK, что у всех писателей мира), baseline (снимок ДО decay) снимается до
затухания, а каждая сохранённая чужая сила рапортуется числом в
SleepReport.file_conflict.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.records import observation_weight
from core.transitions import (MAX_STRENGTH, PACK, TransitionTable,
                                  VERSION, table_from_world)
from core.worldfile import world_transaction

# ── ЧИСЛА КОНТРАКТА (experiments/PLASTICITY_CONTRACT.md, НЕ на глаз) ──────
# Порог «опыта достаточно, чтобы оформить знание»: 8 из 15 = половина шкалы.
CONSOLIDATION_THRESHOLD = 8
# Вес создаваемой связи: 80 + 8*strength -> 144..200 на силе 8..15.
WEIGHT_BASE = 80
WEIGHT_PER_STRENGTH = 8
# Потолок повышения. 201+ означает «факт неоспорим»; опыт, каким бы частым он
# ни был, неоспоримым быть не может — для неоспоримого есть каналы знания.
WEIGHT_CAP = 200
# Шаг повышения за сон: (strength - 7) * 10. Семёрка = «порог минус один»,
# чтобы на самом пороге прирост был минимальным (10), а не нулевым.
RAISE_OFFSET = 7
RAISE_STEP = 10
# Пороги конфликта — ТЕ ЖЕ, что читает brain.lipid2_brain.experience_conflicts:
# инспектор и сон обязаны судить по одному критерию, иначе отчёт и лечение
# разошлись бы числами (отчёт «расхождений 3», а полечено 1).
CONFLICT_WEIGHT_MIN = 150
CONFLICT_STRENGTH_MAX = 3
CONFLICT_WEIGHT_DOWN = 30
# Пол понижения: 80 = базовое «может». Ниже — «не может», а такое опыту
# не принадлежит: отсутствие наблюдений НЕ является опровержением.
WEIGHT_FLOOR = 80
# Затухание силы переходов за один сон: 5%. Медленнее — редкий шум живёт
# вечно и продолжает побеждать в предсказании; быстрее — редкие, но настоящие
# переходы вымываются, не успев дорасти до порога 8.
DECAY_RATE = 0.05
# ── КОГДА СПАТЬ (замер, а не «по настроению») ───────────────────────
# Сон дорог и НЕ должен вызываться на каждое предложение: он правит веса.
#
# TICKS_PER_STRENGTH_POINT — ИЗМЕРЕННАЯ величина: сколько тиков цикла стоит
# сильнейшей паре опыта +1 силы. ЗАМЕР (рассказ из 6 предложений, 16 прогонов
# БЕЗ снов): прогон даёт ровно 10 тиков, сила самой частой пары растёт ровно
# на 1 за прогон: 2,3,3,4,5,6,7,8,... Порог 8 достигнут на 8-м прогоне,
# то есть на 80 накопленных тиках.
#
# Порог сна ВЫВОДИТСЯ из этой величины, а не выбирается на глаз. Замер при
# пороге 64 (первая версия константы, выведенная от неверных «8 тиков»):
#   снов 2, осмысленных 1, created=4 — один сон прошёл ВПУСТУЮ,
#   и впустую он не был: «затухло 8, стёрто 1» при created=0.
# То есть пустой сон не бывает бесплатным: он списывает именно тот опыт, из
# которого позже должно родиться знание. При 80: снов 2, осмысленных 2,
# created=4, raised=4, пустых 0 (замер той же методики).
TICKS_PER_STRENGTH_POINT = 10
SLEEP_TRIGGER_TICKS = int(os.environ.get(
    "LIPID2_SLEEP_TICKS",
    str(CONSOLIDATION_THRESHOLD * TICKS_PER_STRENGTH_POINT)))
# Имя секции мира, где живёт статистика снов. Выбрано отдельным ключом, а не
# «припиской» к transitions, потому что tools/clean_world.py переносит
# НЕИЗВЕСТНЫЕ ему секции нетронутыми (замерено: список KNOWN у него короткий,
# а незнакомая секция иначе молча стиралась). Значит новый ключ — единственный
# способ оставить след в мире, не отняв чужие данные.
SLEEP_SECTION = "sleep"
# Отношение, которым говорит опыт (см. docstring модуля).

RELATION = "CAN_DO"          # единственное отношение, которым говорит опыт
SOURCE = "experience:consolidation"
# Сколько последних снов хранить в секции мира. Число, а не «все": мир
# перечитывается на каждый запрос, а полная история всё равно не нужна —
# итоговые счётчики живут рядом с историей.
SLEEP_HISTORY = 20


def _complain(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")
    sys.stderr.flush()


@dataclass
class SleepReport:
    """
    Что сон СДЕЛАЛ, числом и поимённо.

    Отчёт обязателен: правка веса без перечисления неотличима от «что-то
    поменялось». Каждая правка лежит в .changes с head/tail/старым/новым и
    причиной — чтобы спор «почему у связи вес 170» решался чтением, а не
    догадкой.
    """
    examined: int = 0            # пар «опыт <-> онтология» рассмотрено
    created: int = 0             # новых CAN_DO, рождённых из опыта
    raised: int = 0              # поднято веса (опыт подтверждает)
    lowered: int = 0             # понижено веса (конфликт с опытом)
    skipped_negated: int = 0     # опыт против ОТРИЦАННОЙ связи — не трогаем
    decayed: int = 0             # переходов уменьшено затуханием
    compacted: int = 0           # переходов удалено (сошли к нулю)
    file_conflict: int = 0       # чужих сил в файле, которые сон перетёр
    dry_run: bool = False
    changes: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        """Одна строка для журнала/GUI. Пустой сон обязан читаться пустым."""
        bits = [f"просмотрено {self.examined}", f"создано {self.created}",
                f"поднято {self.raised}", f"понижено {self.lowered}",
                f"затухло {self.decayed}", f"стерто {self.compacted}"]
        out = "СОН: " + ("(пробный, не записано) " if self.dry_run else "") \
            + ", ".join(bits)
        if self.skipped_negated:
            out += f"; пропущено отрицанных {self.skipped_negated}"
        if self.file_conflict:
            out += f"; ⚠ заменой перетёрто чужих записей {self.file_conflict}"
        return out


# ── ЧТЕНИЕ ОНТОЛОГИИ (только данные: никакого LLM и никакого графа) ───────
def _can_do_index(world: dict) -> Dict[tuple, dict]:
    """
    CAN_DO-наблюдения Вселенной как словарь (head, tail) -> запись.

    Берём ИЗ ДАННЫХ, а не из графа: граф пересобирается из наблюдений при
    старте (Learner.replay_observations), и решение, основанное на графе,
    зависело бы от того, успел ли данный процесс его собрать. Наблюдения —
    единственный источник, доступный ВСЕГДА.
    """
    out: Dict[tuple, dict] = {}
    for obs in (world.get("observations") or {}).values():
        if not isinstance(obs, dict):
            continue
        if str(obs.get("relation", "")).upper() != RELATION:
            continue
        head = str(obs.get("head", "")).lower().strip()
        tail = str(obs.get("tail", "")).lower().strip()
        if not head or not tail:
            continue
        out[(head, tail)] = obs
    return out


def weight_of(obs: Optional[dict]) -> int:
    """Уверенность наблюдения — через core.records (вывод для старых файлов)."""
    return observation_weight(obs)


# ── 1+2+3. КОНСОЛИДАЦИЯ ──────────────────────────────────────────────────
def consolidate(world: dict, transitions: Optional[TransitionTable] = None,
                *, threshold: int = CONSOLIDATION_THRESHOLD,
                dry_run: bool = False) -> SleepReport:
    """
    Перевести накопленный опыт в вес онтологии. Правит world НА МЕСТЕ
    (секция observations), возвращает SleepReport.

    Мир принимается СЛОВАРЁМ, а не путём: тест обязан видеть результат без
    файловой системы, а мозг обязан уметь вызвать это внутри своей транзакции
    (иначе два read-modify-write в одном вызове перемешались бы).
    """
    rep = SleepReport(dry_run=dry_run)
    table = transitions if transitions is not None else table_from_world(world)
    can_do = _can_do_index(world)
    obs = world.setdefault("observations", {})
    now = int(time.time())

    for head in table.heads():
        for tail in table.known_tails(head):
            strength = table.get(head, tail)
            if strength <= 0:
                continue
            rep.examined += 1
            existing = can_do.get((head, tail))

            if existing is not None and existing.get("negated"):
                # Опыт ждёт связи, а текст сказал «не делает». Создать CAN_DO
                # нельзя: это противоречие, а не знание. Понизить вес тоже
                # нельзя: weight отрицанной записи отвечает за ОТРИЦАНИЕ, а не
                # за частоту. Молча проглотить — значит оставить противоречие
                # невидимым навсегда, поэтому считается и рапортуется.
                rep.skipped_negated += 1
                rep.changes.append({"kind": "skipped_negated", "head": head,
                                    "tail": tail, "strength": strength})
                continue

            if existing is None:
                # ── ПРАВИЛО 1: опыта достаточно, знания нет ──
                if strength < threshold:
                    continue
                w = min(WEIGHT_CAP, WEIGHT_BASE + WEIGHT_PER_STRENGTH * strength)
                key = f"{head}|{RELATION}|{tail}"
                rec = {"head": head, "relation": RELATION, "tail": tail,
                       # confidence здесь — доля СИЛЫ ОПЫТА (strength/потолок),
                       # а не уверенность разбора: у опыта разбора нет.
                       "confidence": round(strength / float(MAX_STRENGTH), 3),
                       "negated": False,
                       "evidence": f"опыт: переход «{head}->{tail}» "
                                   f"{strength}/{MAX_STRENGTH}",
                       "source": SOURCE, "support": 1, "first_seen": now,
                       "weight": w,
                       "consolidated_strength": strength}
                if not dry_run:
                    obs[key] = rec
                    can_do[(head, tail)] = rec
                rep.created += 1
                rep.changes.append({"kind": "created", "head": head,
                                    "tail": tail, "new_weight": w,
                                    "strength": strength})
                continue

            # ── ПРАВИЛО 3: конфликт (онтология утверждает, опыт молчит) ──
            cur_w = weight_of(existing)
            if cur_w >= CONFLICT_WEIGHT_MIN and strength <= CONFLICT_STRENGTH_MAX:
                new_w = max(WEIGHT_FLOOR, cur_w - CONFLICT_WEIGHT_DOWN)
                if new_w != cur_w:
                    if not dry_run:
                        existing["weight"] = new_w
                        existing["weight_reason"] = (
                            f"сон: опыт {strength}/{MAX_STRENGTH} при весе "
                            f"{cur_w} — понижение до {new_w}")
                    rep.lowered += 1
                    rep.changes.append({"kind": "lowered", "head": head,
                                        "tail": tail, "old_weight": cur_w,
                                        "new_weight": new_w,
                                        "strength": strength})
                continue

            # ── ПРАВИЛО 2: подтверждение ──
            if strength < threshold:
                continue
            new_w = min(WEIGHT_CAP,
                        cur_w + (strength - RAISE_OFFSET) * RAISE_STEP)
            if new_w > cur_w:
                if not dry_run:
                    existing["weight"] = new_w
                    existing["weight_reason"] = (
                        f"сон: опыт {strength}/{MAX_STRENGTH} подтверждает")
                rep.raised += 1
                rep.changes.append({"kind": "raised", "head": head,
                                    "tail": tail, "old_weight": cur_w,
                                    "new_weight": new_w, "strength": strength})
    if rep.skipped_negated:
        rep.notes.append(
            f"опыт ждёт {rep.skipped_negated} связок, которым текст ПРОТИВОРЕЧИТ "
            f"(«не делает») — CAN_DO НЕ создан, противоречие оставлено на лицо")
    return rep


# ── 4. ЗАТУХАНИЕ СИЛЫ ПЕРЕХОДОВ ──────────────────────────────────────────
def decay_transitions(transitions: TransitionTable,
                      rate: float = DECAY_RATE,
                      *, dry_run: bool = False) -> SleepReport:
    """
    Умножить силу КАЖДОГО перехода на (1 - rate), целая часть; затем compact().

    ЗАЧЕМ. Потолок силы 15, шаг подтверждения +2 (замер). Переход, один раз
    случайно угадавший, остаётся в топе предсказания НАВСЕГДА: он уже не
    подтверждается, но и не уменьшается, и вытеснить его может только другой
    шум. Без затухания «предсказуемость системы» — это память о случайности.

    ПОЛ НЕЛЯ: 5% от силы 1 даёт int(0.95) = 0 — переход уходит целиком. Это не
    огрубление, а смысл: редкий переход, простоявший без подтверждения, обязан
    иметь шанс исчезнуть, иначе таблица забивается шумом, который никогда не
    подтвердится (речь о качестве предсказания, а не о 4 битах памяти).

    Сильные переходы обязаны ОСТАВАТЬСЯ РАЗЛИЧИМЫМИ после одного сна
    (15 -> 14, 8 -> 7): иначе затухание сплющило бы шкалу и стёрло порог 8.
    Проверяется тестом.
    """
    rep = SleepReport()
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"rate={rate}: доля затухания обязана быть в [0, 1)")
    pairs = [(h, t) for h in transitions.heads()
             for t in transitions.known_tails(h)]
    for head, tail in pairs:
        s = transitions.get(head, tail)
        if s <= 0:
            continue
        new_s = int(s * (1.0 - rate))
        if new_s == s:
            continue
        if not dry_run:
            transitions.set_strength(head, tail, new_s)
        rep.decayed += 1
        rep.changes.append({"kind": "decay", "head": head, "tail": tail,
                            "old_strength": s, "new_strength": new_s})
    if not dry_run:
        # compact() ОБЯЗАТЕЛЕН: обнулённая сила оставляет имя хвоста в ряду до
        # утраковки (core.transitions._adjust), а сон — как раз место, где
        # пачка убрана дешевле всего.
        rep.compacted = transitions.compact()
    return rep


# ── СОХРАНЕНИЕ ОПЫТА С УЧЁТОМ ЧУЖИХ ПРАВОК ──────────────────────────────
# ── ПОРОГ И СТАТИСТИКА СНА ──────────────────────────────────────────
def has_meaningful_work(rep: SleepReport) -> bool:
    """
    Был ли сон НЕ холостым.

    Нужен для порога срабатывания: «сон состоялся» и «сон что-то сделал» —
    РАЗНЫЕ события. Затухание происходит всегда (оно и задумано как уборка),
    поэтому проверка по rep.decayed не отличил бы уборку от продвижения и счётчик
    «сколько снов реально двинули знание» был бы бессмысленным.
    """
    return bool(rep.created or rep.raised or rep.lowered)


def record_in_world(world: dict, rep: SleepReport, *,
                    now: Optional[int] = None) -> dict:
    """
    Записать итог сна в секцию мира. Возвращает то, что записал.

    Зачем это в ДАННЫХ, а не только в журнале: вопрос «спит ли система вообще
    и что ей это дало» должен решаться чтением файла памяти, а не догадкой по
    лог-строке, которая живёт до перезапуска окна.

    Поля — числа и одна строка на сон; история обрезается (последние
    SLEEP_HISTORY), потому что мир читается целиком на каждый запрос и
    бесконечный журнал замедлял бы его, не добавляя ничего, чего нет в
    счётчиках.
    """
    section = world.setdefault(SLEEP_SECTION, {})
    hist = section.setdefault("history", [])
    entry = {"at": int(now if now is not None else time.time()),
             "examined": rep.examined, "created": rep.created,
             "raised": rep.raised, "lowered": rep.lowered,
             "decayed": rep.decayed, "compacted": rep.compacted,
             "skipped_negated": rep.skipped_negated,
             "file_conflict": rep.file_conflict,
             "meaningful": has_meaningful_work(rep)}
    hist.append(entry)
    if len(hist) > SLEEP_HISTORY:
        del hist[:-SLEEP_HISTORY]
    section["cycles"] = int(section.get("cycles", 0)) + 1
    section["meaningful_cycles"] = int(
        section.get("meaningful_cycles", 0)) + (1 if entry["meaningful"] else 0)
    for key in ("created", "raised", "lowered", "decayed", "compacted",
                "skipped_negated", "file_conflict"):
        section[key] = int(section.get(key, 0)) + entry[key]
    section["last"] = entry
    return section


def sleep_due(ticks_since_sleep: int, *,
              threshold: int = SLEEP_TRIGGER_TICKS) -> bool:
    """
    Пора ли спать. Только накопленные ТИКИ, и вот почему.

    Казалось бы правильнее спать по факту «накоплен опыт >= 8»: сон тогда
    гарантированно не холостой. Но это ЗАКРЫЛО бы дверь навсегда: опыт
    доращивает именно предсказание, а предсказание читает таблицу, которую
    сон точит. Ждать, пока «точно будет что продвинуть», — значит ждать, что
    опыт накопится без сна, при том что сон этот опыт и уменьшает.

    Поэтому порог — по объёму ПРОЙДЕННОГО материала (тики цикла), а отчёт
    честно покажет created=0, если материала не хватило. Холостой сон — это
    измеримый факт, а не ошибка: см. meaningful_cycles.
    """
    return int(ticks_since_sleep) >= threshold


def table_of_section(section: dict) -> TransitionTable:
    """
    Таблица из СЕКЦИИ мира (не из мира целиком).

    Причина отдельной функции — собственный дефект этой же правки: сон собирал
    таблицу вызовом table_from_world(merged_section), а table_from_world ждёт
    СЛОВАРЬ МИРА и читает из него ключ "transitions". Секция — это уже и есть
    transitions, поэтому .get("transitions") отдавал None, таблица выходила
    ПУСТОЙ, и сон, ничего не увидев, перезаписывал живой опыт пустотой
    (замерена потеря перехода силой 12). Ошибка молчала: пустая таблица —
    законное состояние («ещё не учились»), и проверить её нельзя ничем, кроме
    того, что кто-то передаёт не тот словарь.

    Отсюда правило: СЕКЦИЯ -> table_of_section, МИР -> table_from_world.
    Смешать их теперь можно только через эти две имена, и они разные.
    """
    t = TransitionTable()
    t.from_json(section or {})
    return t


def _section_map(data) -> dict:
    """Секция transitions -> {(head, tail): сила}.

    Кривых строк здесь просто нет из чего построить пару (не строковый head,
    хвост не строка, сила не число). МОЛЧА они не пропадают: их видит и
    считает тот, кто читает секцию целиком (core.transitions.from_json /
    merge_json) — эта функция служебная и дублировать их счёт не обязана,
    но и не делает вид, что их не было."""
    out = {}
    if not isinstance(data, dict):
        return out
    for row in data.get("rows") or []:
        if not isinstance(row, dict):
            continue
        head = row.get("head")
        tails = row.get("tails") or []
        strengths = row.get("strengths") or []
        if not isinstance(head, str):
            continue
        for t, s in zip(tails, strengths):
            if isinstance(t, str) and t:
                try:
                    out[(head, t)] = int(s)
                except (TypeError, ValueError):
                    continue
    return out


def reconcile_saved(baseline: dict, ours: dict, file_: dict) -> tuple:
    """
    Собрать секцию transitions ДЛЯ ЗАПИСИ СНА. Возвращает (секция, потери).

    baseline — таблица ДО затухания (то, на чём сон считал консолидацию);
    ours     — таблица ПОСЛЕ консолидации и затухания (чего хочет сон);
    file_    — что лежит в файле в момент транзакции.

    ТРИ случая РАЗНЫЕ. Слить их одним merge_json по максимуму — значит
    отменить собственный decay (см. ЗАМЕР в docstring модуля):

      а) пара есть у нас и в файле, сила в файле НЕ ВЫШЕ baseline — никто не
         трогал: пишем НАШУ (затухшую) силу;
      б) пара есть, сила в файле ВЫШЕ baseline — сосед НАРАБОТАЛ опыта, пока
         мы спали. Откатывать чужой опыт нашим decay нельзя: берём бОльшую
         силу и сообщаем потерю ЧИСЛОМ;
      в) пары у нас нет, в файле есть — чужой опыт, которого мы не видели:
         сохраняем КАК ЕСТЬ (выкинуть = потерять знание).

    «Потери» — это случай б). Он не «всегда ноль»: это счётчик, который обязан
    быть виден в отчёте, а не проглочен.
    """
    base, our, cur = _section_map(baseline), _section_map(ours), _section_map(file_)
    lost = 0
    final: Dict[tuple, int] = dict(our)              # наш опыт, уже затухший
    for pair, strength in cur.items():
        if pair not in base:
            final.setdefault(pair, strength)          # (в) чужая пара
            continue
        if strength > base[pair]:
            lost += 1                                 # (б) сосед дорастив
            final[pair] = max(final.get(pair, 0), strength)
        # (а) не менялось — остаётся наша сила из `our`
    rows = []
    for head in sorted({h for h, _ in final}):
        pairs = sorted(((t, s) for (h, t), s in final.items() if h == head),
                       key=lambda p: (-p[1], p[0]))
        pairs = [(t, s) for t, s in pairs if s > 0]
        if not pairs:
            continue
        rows.append({"head": head, "tails": [t for t, _ in pairs],
                     "strengths": [s for _, s in pairs]})
    return {"version": VERSION, "pack": PACK, "rows": rows}, lost


def save_transitions(path: str, transitions: TransitionTable,
                     report: Optional[SleepReport] = None,
                     baseline: Optional[dict] = None) -> int:
    """
    Записать секцию transitions: применить СВОЁ затухание и СОХРАНИТЬ чужое.

    Причина, по которой нельзя обычный merge_json, — ЗАМЕР, а не вкус: тот
    берёт максимум, поэтому decay отменяется собственным сохранением (в памяти
    5, в файле снова 10). Сон — единственный код, которому разрешено УМЕНЬШАТЬ
    опыт, значит только он обязан писать не «максимумом».

    baseline (снимок ДО decay) обязателен: без него собственный decay
    неотличим от чужой дорастивки, и сон откатывал бы чужой опыт.

    Запись идёт ВНУТРИ world_transaction — тот же _LOCK, что у всех писателей
    мира, поэтому окно «прочитали -> записали» не растягивается на весь сон.
    Живая таблица становится ТЕМ ЖЕ, что легло в файл: иначе процесс продолжал
    бы работать силой, которой нет уже нигде.
    """
    ours = transitions.to_json()
    with world_transaction(path) as world:
        section, lost = reconcile_saved(baseline or ours, ours,
                                        world.get("transitions") or {})
        world["transitions"] = section
        if lost:
            _complain(f"[sleep] ⚠ опыта соседа больше, чем помнил сон: {lost} "
                      f"переходов нарастили во время сна — взята БОЛЬШАЯ сила "
                      f"(откатывать чужой опыт decay-ом нельзя)")
    transitions.from_json(section)
    if report is not None:
        report.file_conflict = lost
    return lost


# ── ПОЛНЫЙ СОН ───────────────────────────────────────────────────────────
def sleep(world: dict, path: Optional[str] = None,
          transitions: Optional[TransitionTable] = None,
          *, decay_rate: float = DECAY_RATE,
          dry_run: bool = False) -> SleepReport:
    """
    Один полный цикл: консолидация -> затухание -> (если задан путь) запись.

    ПОРЯДОК НЕ СЛУЧАЕН и менять его нельзя: затухание идёт ПОСЛЕ консолидации,
    иначе сон сам обнулировал бы опыт, из которого только что сделал знание, и
    порог 8 «прыгал» бы в зависимости от порядка вызовов. Тест проверяет:
    связка, рождённая этим сном, сохраняет ровно 80 + 8*сила, без поправок на
    собственный decay.

    baseline (снимок ДО затухания) снимается ЗДЕСЬ: без него запись не отличила
    бы собственный decay от чужого нарастания опыта (см. save_transitions).
    """
    table = transitions if transitions is not None else table_from_world(world)
    baseline = table.to_json()
    rep = consolidate(world, table, dry_run=dry_run)
    d = decay_transitions(table, decay_rate, dry_run=dry_run)
    rep.decayed = d.decayed
    rep.compacted = d.compacted
    rep.changes.extend(d.changes)
    rep.notes.extend(d.notes)
    if path and not dry_run:
        save_transitions(path, table, rep, baseline=baseline)
    return rep


if __name__ == "__main__":
    # Контрольный прогон контракта на глазах: три правила + затухание.
    ok = True

    def check(name, res):
        global ok
        print(f"  {'OK  ' if res else 'FAIL'} {name}")
        ok = ok and bool(res)

    def obs(head, tail, weight, negated=False, conf=0.9, support=1):
        return {f"{head}|CAN_DO|{tail}": {
            "head": head, "relation": "CAN_DO", "tail": tail,
            "confidence": conf, "support": support, "weight": weight,
            "_NEGATED": negated}}

    def make_obs(rec):
        k = list(rec)[0]
        rec[k]["negated"] = rec[k].pop("_NEGATED")
        return rec

    world = {"observations": {}}
    for r in (obs("кит", "нырять", 220),      # конфликт -> понижение
              obs("кошка", "лазать", 100, conf=0.5),   # подтверждение
              obs("рыба", "бегать", 200, negated=True)):   # трогать нельзя
        world["observations"].update(make_obs(r))

    t = TransitionTable()
    t.confirm("кошка", "лазать", 12)      # подтверждение
    t.confirm("птенец", "летать", 11)     # создание знания
    t.confirm("рыба", "бегать", 14)       # против отрицания
    t.confirm("кит", "нырять", 1)         # конфликт

    r = sleep(world, None, t)
    print(r.as_text())
    O = world["observations"]
    check("создан CAN_DO из опыта (птенец->летать)",
          "птенец|CAN_DO|летать" in O)
    check("вес созданной связи = 80 + 8*сила",
          O.get("птенец|CAN_DO|летать", {}).get("weight") == 80 + 8 * 11)
    check("созданная связь помечена источником опыта",
          O.get("птенец|CAN_DO|летать", {}).get("source") == SOURCE)
    check("подтверждение подняло вес (100 -> 150)",
          O["кошка|CAN_DO|лазать"]["weight"] == 150)
    check("конфликт понизил вес (220 -> 190)",
          O["кит|CAN_DO|нырять"]["weight"] == 190)
    check("понижение не ушло под пол 80",
          all(v["weight"] >= WEIGHT_FLOOR for v in O.values()))
    check("отрицанная связь НЕ тронута и заявлена",
          r.skipped_negated == 1 and O["рыба|CAN_DO|бегать"]["weight"] == 200)
    check("затухание уменьшило силу (12 -> 11)", t.get("кошка", "лазать") == 11)

    tt = TransitionTable()
    tt.confirm("a", "b", 15)
    tt.confirm("a", "c", 8)
    tt.confirm("a", "d", 1)
    sleep({"observations": {}}, None, tt)
    check("15 и 8 остаются РАЗЛИЧИМЫМИ после одного сна (14 и 7)",
          tt.get("a", "b") == 14 and tt.get("a", "c") == 7)
    check("сила 1 гаснет полностью, compact() убирает имя",
          tt.get("a", "d") == 0 and "d" not in tt.known_tails("a"))

    w2 = {"observations": {"x|CAN_DO|y": {
        "head": "x", "relation": "CAN_DO", "tail": "y", "confidence": 1.0,
        "support": 1, "weight": 220, "negated": False}}}
    t2 = TransitionTable()
    t2.confirm("p", "q", 15)
    sleep(w2, None, t2, dry_run=True)
    check("dry_run не меняет ни мира, ни таблицы",
          w2["observations"]["x|CAN_DO|y"]["weight"] == 220
          and t2.get("p", "q") == 15)
    check("dry_run помечен в отчёте", "пробный" in SleepReport(
        dry_run=True).as_text())

    print(f"\nчисла контракта: порог {CONSOLIDATION_THRESHOLD}/{MAX_STRENGTH}, "
          f"вес создания {WEIGHT_BASE}+{WEIGHT_PER_STRENGTH}*s, "
          f"потолок {WEIGHT_CAP}, конфликт {CONFLICT_WEIGHT_MIN}/"
          f"<={CONFLICT_STRENGTH_MAX} -{CONFLICT_WEIGHT_DOWN}, "
          f"пол {WEIGHT_FLOOR}, decay {DECAY_RATE}")
    raise SystemExit(0 if ok else 1)
