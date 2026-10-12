"""
brain/learner.py — Обучение из текста: предложение -> знание в графе.

МЕСТО В АРХИТЕКТУРЕ
  Мозг (Lipid2Brain) решает «что спросить и что ответить». Learner решает
  «что ИЗВЛЕЧЬ из того, что дали, и как это записать, не испортив уже
  известное». Он не знает про IR и Handlers и не ходит в HTTP: все внешние
  способности приходят функциями в конструкторе, поэтому тестируется без
  llama.cpp и без загрузки эмбеддингов.

ТРИ ГЛАВНЫХ ПРАВИЛА ЗАПИСИ (нарушение любого = тихая порча знаний)

  1. НЕ ПЕРЕЗАПИСЫВАТЬ. Существовала сущность с пятью свойствами — после
     обучения должно быть пять + новые, а не «одно, зато свежее». То же к
     иерархии и действиям. LLM выдаёт неполные записи; «обновить всёцело»
     означает деградировать базу при каждом удачном ответе модели.

  2. НЕ ВЫДУМЫВАТЬ СУЩНОСТЬ. Если слова нет в мире и Учитель молчит, узел
     всё равно участвует в связях («кошка сидит на подоконнике» — подоконник
     нужен в графе как вершина). Но в entities он не появляется: сущность без
     иерархии — это ложное «мы про неё знаем». Такие слова уходят в
     unknown_words с доказательством и счётчиком.

  3. ПОМНИТЬ ПРОИСХОЖДЕНИЕ. У каждого факта: источник, цитата, уверенность,
     число подтверждений. Знание из одного рассказа и онтология из Вики
     выглядят в графе одинаково — и через месяц нельзя отличить факт от
     правдоподобной выдумки локалки, а именно это решает, можно ли на знание
     опираться.

УВЕРЕННОСТЬ -> ВЕС СИНАПСА
  Граф оперирует весами 0..255, экстрактор — вероятностями 0..1. Это НЕ
  одно и то же, и переводить их молча нельзя: правдоподобная догадка из
  омонима не должна иметь вес проверенного факта.
      CAN_DO/HAS_PROPERTY:  90 + 130*conf   (уверенное — до 220)
      прочие отношения:     60 + 100*conf   (обстоятельства слабее по жизни)
      отрицание:            вес делится на 3 — «не боится» описывает отсутствие
  Повторное свидетельство adds support и слегка поднимает вес: два независимых
  предложения — уже не случайность.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

if __package__ in (None, ""):        # прямой запуск: python brain/learner.py
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from core.extract import Fact, SimpleExtractor
from core.logchannel import status as _status
from core.morph import Morph
from core.records import merge_entity
from core.storage import NodeType, RelationType

try:
    from core.actions import profile_of
except Exception as _exc:                            # pragma: no cover
    # Без справочника глагол всё равно закрепится узлом ACTION, но БЕЗ профиля
    # K/M/S. Оставлять это НЕВидимым нельзя: предиктивный цикл перестанет
    # отличать одно действие от другого, а выглядит это как обычная работа.
    print(f"[learner] ⚠️ core.actions недоступен ({_exc}) — глаголы пишутся "
          "БЕЗ профиля K/M/S, предиктивный цикл их не различает",
          file=sys.stderr, flush=True)

    def profile_of(verb):
        return None

# Имя отношения в тексте -> enum. Обратная таблица для чтений из файла.
REL: Dict[str, RelationType] = {r.name: r for r in RelationType}
# Имя отношения экстрактора -> enum графа. Совесть требует НЕ сваливать всё
# в HAS_PROPERTY/CAUSES: «кошка боится грозы» и «кошка пушистая» — разные
# утверждения, и, сведя их к одному типу, мы навсегда потеряем различие
# «действие/объект» vs «свойство». IS_ROLE == IS_A по смыслу — это единственное
# допустимое слияние.
REL_ALIAS = {"IS_ROLE": "IS_A"}

# Тип узла по роли слова в предложении.
ROLE_TO_NODETYPE = {"subject": NodeType.ENTITY, "property": NodeType.PROPERTY,
                    "action": NodeType.ACTION}

# Два разных состояния «не знаю» (см. цикл обучения ниже): молча смешивать их
# нельзя — «спросили и не нашли» трогать больше не надо, «не спрашивали» надо
# спросить при первой возможности.
ASKED_EMPTY = "asked_empty"
NOT_ASKED = "not_asked"
# Глагол, профиля которого нет в справочнике. Это НЕ ask_empty: «не нашли
# онтологию действия» было бы враньём — мы её и не искали, потому что
# Википедия и LLM описывают СУЩНОСТИ. Слово помечено как ЖДУЩЕЕ профиля: оно
# стоит в графе узлом ACTION, его связи работают, а K/M/S придут с замером.
ACTION_NO_PROFILE = "action_no_profile"


class TeacherUnavailable(RuntimeError):
    """
    Учитель НЕ ОТВЕТИЛ по causes вне его компетенции: канал выключен, сервер
    лёг, ответ обрезан по токенам, парсер упал.

    Существо — в том, что вызывающий обязан ОТЛИЧИТЬ это от честного «не знаю».
    «Не нашли» => asked_empty, слово больше не спрашиваем (экономия минут на
    запрос). «Не спросили» => not_asked, слово ОБЯЗАНО вернуться в очередь.
    Смешать их — значит либо молча сжигать часы на повторные запросы, либо
    НАВСЕГДА снять слово с изучения. Замерено в обоих направлениях.
    """


class TeacherDisabled(TeacherUnavailable):
    """
    Учителя СПРОСИТЬ БЫЛО НЕЛЬЗЯ: канал выключен пользователем или исчерпан
    лимит обращений урока.

    Выделиться отдельным типом нужно потому, что исход у этих случаев РАЗНЫЙ:
    при выключенном канале мы не имели права вообще записывать слово в
    unknown_words, а при исчерпанном лимите — имели и должны. Первое — наш
    собственный отказ, второе — обычная экономия бюджета урока.
    """


def _mark_unknown(rep: "LearningReport", lemma: str) -> None:
    """Одно место для счёта «осталось неизвестным»: поле unknown означает
    именно ИТОГ урока, а не «слово было незнакомым». Раньше в него попадали и
    выученные слова, поэтому отчёт говорил «неизвестных 1» про слово, которое
    только что выучил, и «распознано 0%» — про успешно пройденный урок."""
    rep.unknown_words += 1
    if lemma not in rep.unknown:
        rep.unknown.append(lemma)


@dataclass
class LearningReport:
    """Измеримый итог одного сеанса обучения. Цифры — не косметика: без них
    невозможно сказать, стало ли системе лучше после правки правил."""
    text_len: int = 0
    sentences: int = 0
    content_words: int = 0
    known_words: int = 0
    unknown_words: int = 0
    teacher_calls: int = 0
    teacher_success: int = 0
    facts: int = 0
    entities_created: int = 0
    entities_updated: int = 0
    observations_new: int = 0
    observations_seen: int = 0
    skipped: int = 0
    low_confidence: int = 0
    new_concepts: List[str] = field(default_factory=list)
    learned_concepts: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    # Глаголы урока, закреплённые узлами ACTION. Отдельно от learned_concepts:
    # глагол учится ДРУГИМ способом (отношение + профиль), и сваливать его в
    # общий список — значит позже снова перепутать его с сущностью.
    learned_actions: List[str] = field(default_factory=list)
    # Снятые со слова диагнозы asked_empty, выставленные НЕУМЕСТНЫМ вопросом
    # (искали сущность там, где действие). Удаление из unknown_words обязано
    # пережить слияние с чужим снимком — иначе неверный диагноз возвращается.
    retracted_unknown: List[str] = field(default_factory=list)
    # Сколько слов ПРИШЛО незнакомыми (в т.ч. те, что удалось выучить). Нужно
    # вместе с unknown_words, иначе «распознано 0 из 1» рядом с «неизвестных 0»
    # выглядит как вранье отчёта, а не как «не знал — выучил».
    seen_unknown_words: int = 0
    # Слова, оставленные неизвестными БЕЗ вопроса Учителю: канал был выключен
    # или исчерпан лимит. Это НЕ «мы не нашли» — это «мы не искали», и их надо
    # выучить позже. Терять это различие = терять слова навсегда.
    not_asked: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    quality: Dict[str, bool] = field(default_factory=dict)

    @property
    def not_asked_words(self) -> int:
        return len(self.not_asked)

    def as_text(self) -> str:
        acc = (self.known_words / self.content_words) if self.content_words else 0.0
        lines = [
            "📚 Обучение из текста",
            f"  предложений: {self.sentences}   значимых слов: {self.content_words}"
            f"   распознано: {self.known_words} ({acc:.0%})   незнакомых: {self.seen_unknown_words}   выучено сейчас: {len(self.learned_concepts) + len(self.learned_actions)}",
            f"  фактов извлечено: {self.facts}   новых наблюдений: {self.observations_new}"
            f"   подтверждено существующих: {self.observations_seen}",
            f"  сущностей создано: {self.entities_created}   дополнено: {self.entities_updated}",
            f"  обращений к Учителю: {self.teacher_calls}"
            f"   успешно: {self.teacher_success}   пропущено фактов: {self.skipped}"
            f"   низкоуверенных: {self.low_confidence}",
        ]
        if self.learned_concepts:
            lines.append("  выучено: " + ", ".join(self.learned_concepts[:12]))
        if self.learned_actions:
            lines.append("  действия (ACTION): " + ", ".join(self.learned_actions[:12]))
        if self.new_concepts:
            # Не дубль learned_concepts: learned_concepts — слова, которых не было
            # в лексиконе; new_concepts — сущности, созданные ИМЕННО из этого
            # источника. Разница видна на переучивании: слово уже знакомо, а
            # сущность появилась новая (или наоборот).
            lines.append("  сущностей из этого источника: "
                         + ", ".join(self.new_concepts[:12]))
        if self.unknown:
            lines.append("  осталось неизвестным: " + ", ".join(self.unknown[:12]))
        lines.append(f"  из них НЕ спрашивали: {self.not_asked_words}"
                     + (f" — {', '.join(self.not_asked[:8])}" if self.not_asked else ""))
        for n in self.notes[:6]:
            lines.append("  ! " + n)
        return "\n".join(lines)


class Learner:
    """
    Обучение из простых текстов. Зависимости — только функции, переданные
    в конструктор (см. выше «место в архитектуре»).

        load_world() -> dict          чтение Вселенной
        save_world(dict) -> None      запись Вселенной
        core                            GraphMemory (может быть None)
        known_forms(word) -> lemma|None морфологическое/семантическое ядро форм
        teacher(lemma) -> entity|None   канал знания (LLM/Wiki), может быть None
    """

    def __init__(self, load_world: Callable[[], dict],
                 save_world: Callable[[dict], None],
                 core=None,
                 known_forms: Optional[Callable[[str], Optional[str]]] = None,
                 teacher: Optional[Callable[[str], Optional[dict]]] = None,
                 extractor: Optional[SimpleExtractor] = None,
                 max_teacher_calls: int = 3,
                 min_confidence: float = 0.30):
        self.load_world = load_world
        self.save_world = save_world
        self.core = core
        self.known_forms = known_forms
        self.teacher = teacher
        self.extractor = extractor or SimpleExtractor()
        self.max_teacher_calls = max_teacher_calls
        # Факты ниже этого порога не пишутся в вес: правдоподобная догадка на
        # омониме не должна становиться памятью. Они считаются и отчётливы.
        self.min_confidence = min_confidence

    # ── ПУБЛИЧНЫЙ ВХОД ─────────────────────────────────────

    def learn_from_text(self, text: str, source: str = "text",
                        allow_teacher: bool = True) -> LearningReport:
        rep = LearningReport(text_len=len(text or ""),
                             quality={"morph": Morph.available,
                                      "teacher": self.teacher is not None})
        if not text or not text.strip():
            rep.notes.append("пустой текст")
            return rep
        if not Morph.available:
            rep.notes.append("pymorphy3 недоступен: разбор на эвристике, "
                             "часть фактов будет потеряна")
            _status("⚠️ морфология деградировала на эвристику (нет pymorphy3)")

        world = self.load_world()
        entities = world.setdefault("entities", {})
        observations = world.setdefault("observations", {})
        unknowns = world.setdefault("unknown_words", {})

        # Омонимия снимается в том числе известным: «стоит» = 0.33 стоить /
        # 0.33 стоять, но если ядро знает «стоять», выбор однозначен.
        self.extractor.set_known(set(entities.keys()))

        # Слово -> каноническое имя сущности. Строится ОДИН раз на сеанс:
        # поиск по форме — самая дорогая операция (эмбеддинги).
        resolve = self._make_resolver(entities)

        sentences = self.extractor.extract_text(text)
        rep.sentences = len(sentences)
        _status(f"разобрал текст: {rep.sentences} предложений, "
                f"{sum(len(sf.facts) for sf in sentences)} возможных фактов")

        # 1) словарь: кого мы знаем, кого нет
        vocab: Dict[str, dict] = {}
        # Глаголы — вне этой охоты. см. SentenceFacts.action_lemmas.
        verbs: Dict[str, str] = {}
        for sf in sentences:
            for lemma in sf.action_lemmas:
                verbs.setdefault(lemma, sf)
            for lemma in sf.content_lemmas:
                vocab.setdefault(lemma, {"node": self._role_of_lemma(sf, lemma),
                                         "evidence": sf.sentence})
        rep.content_words = len(set(vocab) | set(verbs))

        for lemma in sorted(verbs):
            sf = verbs[lemma]
            if resolve(lemma):
                # Глагол уже стоит в графе — он НЕ незнакомец, иначе процент
                # распознавания падал бы просто из-за глаголов в тексте.
                rep.known_words += 1
            # Снимаем с очереди «неизвестных сущностей», если слово попало туда
            # на УРОКАХ ДО появления глагольного пути.
            if lemma in unknowns and unknowns[lemma].get("status") == ASKED_EMPTY:
                # Прежний диагноз выставлен НЕУМЕСТНЫМ вопросом (искали
                # сущность там, где действие). Силы он не имеет.
                _status(f"«{lemma}»: снимаю прежний asked_empty — это глагол, "
                        "он изучается как действие, а не как сущность")
                del unknowns[lemma]
                rep.retracted_unknown.append(lemma)
            self._store_action(entities, sf, lemma, source, rep)
            if not resolve(lemma):
                # Глагол пришёл незнакомым и закреплен узлом ACTION: он ВЫУЧЕН,
                # но СПОСОБ другой, поэтому учитывается в своей строке отчёта.
                rep.seen_unknown_words += 1
            unknowns.pop(lemma, None)

        for lemma, info in sorted(vocab.items()):
            if lemma in verbs:
                # ГЛАГОЛ НЕ УЧАСТВУЕТ в охоте на «сущность с иерархией».
                # Глагол — не «ещё одна сущность», а отношение: узел ACTION со
                # связями CAN_DO / OBJECT_OF. Гнать его через Википедию или
                # «онтологический разбор» — категориальная ошибка: статью-
                # онтологию на глагол Википедия не даёт, слово получало
                # asked_empty и БОЛЬШЕ не спрашивалось. ЗАМЕРЕНО на
                # «Гладкий шар катится по столу»: «катиться» гонялось в сеть.
                continue
            canon = resolve(lemma)
            if canon:
                rep.known_words += 1
                info["canon"] = canon
                continue
            _status(f"изучаю слово «{lemma}»")
            unknowns_seen = unknowns.get(lemma)
            if unknowns_seen and unknowns_seen.get("status") == ASKED_EMPTY:
                # Спрашивали и ничего не нашли — повторно НЕ гоним Учителя:
                # ответ LLM стоит секунды-минуты, и без этого правила каждое
                # упоминание слова платит за поиск заново.
                unknowns_seen["count"] = int(unknowns_seen.get("count", 1)) + 1
                unknowns_seen["last_seen"] = int(time.time())
                _mark_unknown(rep, lemma)
                info["canon"] = None
                continue
            entity = None
            # Чем КОНЧИЛСЯ вопрос, а не намерение его задать. От этого зависит
            # статус слова — см. TeacherUnavailable.
            outcome, why = "not_asked", "Учитель не подключён"
            if allow_teacher and self.teacher is not None:
                if rep.teacher_calls < self.max_teacher_calls:
                    _status(f"ищу значение «{lemma}» у Учителя "
                            f"({rep.teacher_calls + 1}/{self.max_teacher_calls})")
                    try:
                        entity = self.teacher(lemma)
                        rep.teacher_calls += 1
                        outcome = "empty" if not entity else "ok"
                    except TeacherDisabled as exc:
                        # СПРОСИТЬ БЫЛО НЕЛЬЗЯ (канал выключен). Обращение НЕ
                        # засчитывается: бюджета урока не тратили, а иначе
                        # выключенный канал съедал лимит и остальные слова
                        # оставались вообще без попытки.
                        outcome, why = "disabled", str(exc)
                    except TeacherUnavailable as exc:
                        # Обратились и получили отказ (сервер лёг, ответ
                        # обрезан). Время потрачено — в бюджет идёт.
                        rep.teacher_calls += 1
                        outcome, why = "broken", str(exc)
                    except Exception as exc:
                        rep.teacher_calls += 1
                        outcome, why = "broken", f"{type(exc).__name__}: {exc}"
                else:
                    outcome, why = "limit", (
                        f"лимит обращений к Учителя ({self.max_teacher_calls}) "
                        f"исчерпан")
            asked = (outcome == "ok") or (outcome == "empty")
            usable = bool(entity) and bool(entity.get("hierarchy")
                                        or entity.get("properties"))
            if outcome == "ok" and not usable:
                # Ответ ПРИШЕЛ, но в нём нет ни категории, ни свойств: брать
                # нечего. Это НЕ сбой канала и НЕ «молчание» — Учитель ответил
                # пустой структурой, и называть это падением нельзя.
                outcome = "degenerate"
            if outcome != "ok":
                # Формулировка обязана СООТВЕТСТВОВАТЬ причине. «Учитель упал»
                # про выключенный канал — то же враньё в журнале, что и
                # «не нашли» про невыполненный вопрос.
                text = {
                    "empty": f"«{lemma}»: Учитель ответил «не знаю» — спрашивать "
                             f"повторно не будем",
                    "degenerate": f"«{lemma}»: ответ Учителя ПУСТ по составу "
                                  f"(ни категории, ни свойств) — спрашивать "
                                  f"повторно не будем",
                    "disabled": f"«{lemma}»: Учитель НЕ СПРАШИВАЛСЯ — {why}; "
                                f"слово оставлено КАК НЕСПРОШЕННОЕ",
                    "broken": f"учитель по «{lemma}» НЕ ОТВЕТИЛ: {why}; "
                              f"слово оставлено КАК НЕСПРОШЕННОЕ",
                    "limit": f"«{lemma}»: {why} — слово оставлено "
                             f"КАК НЕСПРОШЕННОЕ, его ещё предстоит выучить",
                    "not_asked": f"«{lemma}»: {why} — слово НЕ изучалось"}[outcome]
                if text not in rep.notes:
                    rep.notes.append(text)
            if usable:
                rep.teacher_success += 1
                _status(f"«{lemma}» получено: {(entity.get('hierarchy') or ['?'])[0]}, "
                        f"свойств {len(entity.get('properties') or [])}")
                self._store_entity(entities, lemma, entity, source,
                                   info["evidence"], rep)
                info["canon"] = lemma
                self._mark_learned(rep, lemma)
                rep.seen_unknown_words += 1
                # Выучили — значит слово больше НЕ неизвестное. Оставлять его
                # в unknown_words нельзя: файл начинает утверждать про одно и то
                # же слово «не знаю» и «вот что оно такое», а GUI показывает обе
                # счётчика рядом («сущностей N · неизвестных M») и выглядит
                # сломанным. ЗАМЕРЕНО: после «Пчела опыляет цветок на лугу»
                # пчела и цветок остались в обоих списках.
                if lemma in unknowns:
                    del unknowns[lemma]
            else:
                # Сущность НЕ создаём (правило 2), но слово помним.
                # СТАТУС пишется, потому что два случая выглядят одинаково, а
                # обращаться с ними надо по-разному:
                #   ASKED_EMPTY — СПРОСИЛИ честным вопросом, ответа нет:
                #                 больше не спрашиваем (экономия минут на запрос);
                #   NOT_ASKED   — НЕ спрашивали (канал выключен, сервер лёг,
                #                 лимит исчерпан): слово ОБЯЗАНО быть спрошено
                #                 позже.
                # Без этого различия «снял галочку на одном уроке» означало
                # «слово запрещено учить всегда»: первая тренировка с выключенным
                # каналом ставила всем встреченным словам пожизненный диагноз.
                # ЗАМЕРЕНО на GUI-прогоне: «опылять/пчела/цветок» остались
                # неизвестными навсегда, хотя Учитель был просто выключен.
                rep.seen_unknown_words += 1
                _mark_unknown(rep, lemma)
                _status(f"«{lemma}» осталось неизвестным "
                        + ("(спросили — пусто)" if asked else
                           f"(НЕ спрашивали: {outcome})"))
                entry = dict(unknowns.get(lemma) or {"first_seen": int(time.time())})
                entry.update({
                    "count": int(entry.get("count", 0)) + 1,
                    "last_seen": int(time.time()),
                    "evidence": info["evidence"][:200],
                    "source": source,
                    # Любой исход, кроме честного пустого ответа Учителя,
                    # означает НЕ СПРАШИВАЛИ: слово ОБЯЗАНО вернуться в
                    # очередь следующего урока.
                    "status": ASKED_EMPTY if asked else NOT_ASKED,
                })
                if outcome == "disabled":
                    entry["status_note"] = f"канал недоступен: {why[:120]}"
                unknowns[lemma] = entry
                if not asked:
                    rep.not_asked.append(lemma)
                info["canon"] = None

        # 2) связи
        for sf in sentences:
            _status(f"раскладываю предложение: «{sf.sentence[:80]}»")
            for fact in sf.facts:
                self._apply_fact(fact, sf, resolve, entities, observations,
                                 source, rep)

        # resolved=выученные: удаление из unknown_words — ТОЖЕ изменение,
        # которое обязано пережить слияние, иначе «чужой писатель» resurrect-ит
        # старую запись и файл снова враньё про одно и то же слово.
        # Сюда же попадают действия и СНЯТЫЕ диагнозы: удаление из unknown
        # ничем не отличается от удаления выученного существительного.
        self._commit(world, resolved=list(rep.learned_concepts)
                     + list(rep.learned_actions) + list(rep.retracted_unknown))
        rep.facts = sum(len(sf.facts) for sf in sentences)
        _status(f"записано: фактов {rep.observations_new}, подтверждено "
                f"{rep.observations_seen}, пропущено {rep.skipped}")
        rep.low_confidence = sum(1 for sf in sentences for f in sf.facts
                                 if f.confidence < self.min_confidence)
        # Действия — НЕ «новые концепты»: для них есть своя строка отчёта
        # (learned_actions). Свалить их сюда — значит получить «сущностей 3»
        # там, где сущностей две, а третье — глагол.
        rep.new_concepts = sorted({k for k in entities
                                   if (entities[k] or {}).get("learned_from") == source
                                   and (entities[k] or {}).get("kind") != "action"})
        # Резолвер форм должен увидеть новичков СРАЗУ: иначе следующее
        # предложение того же текста спросит LLM про уже выученное слово.
        if self.core is not None and hasattr(self.core, "sync_lemmas"):
            try:
                self.core.sync_lemmas()
            except Exception:
                pass
        return rep

    # ── РЕЗЕРВНАЯ КОПИЯ И ТОЧКА ВОССТАНОВЛЕНИЯ ─────────────

    def replay_observations(self, world: dict) -> int:
        """
        Наблюдения -> граф. Вызывается при старте мозга: граф собирается из
        JSON, и без этого шага все связи, выученные из рассказов, исчезали бы
        при перезапуске — обучение работало бы «до закрытия окна».
        Возвращает число восстановленных связей.
        """
        if self.core is None:
            return 0
        n = 0
        for key, obs in (world.get("observations") or {}).items():
            try:
                head, rel_name, tail = key.split("|", 2)
            except ValueError:
                continue
            rel = REL.get(rel_name)
            if rel is None:
                continue
            src = obs.get("source", "replay")
            hn = self.core.get_or_create_node(head, self._node_type(obs, "head"),
                                             source=src,
                                             evidence=obs.get("evidence", "")[:200])
            tn = self.core.get_or_create_node(tail, self._node_type(obs, "tail"),
                                             source=src)
            self._connect(hn, tn, rel, self._weight(float(obs.get("confidence", .5)),
                                                    obs.get("negated"), rel,
                                                    int(obs.get("support", 1))))
            n += 1
        return n

    def _commit(self, world: dict, resolved=()) -> None:
        """
        СОХРАНЕНИЕ = ПЕРЕЧИТАТЬ И СЛИТЬ, а не «перезаписать свой снимок».

        Между чтением мира и записью файл может изменить КТО-ТО ДРУГОЙ.
        Это НЕ теоретическая предосторожность: lipid_world.json пишут пятеро —
        этот learner, AgentTeacherV2 (tool-call локальной модели, пока мы
        «думаем» над предложением), teacher_wiki, канал ответа пользователя и
        соседняя копия проекта (/home/tohter/Lipid2) тем же путём. Его ответ был получен за реальные деньги и за секунды
        ожидания, и потерять его молча — значит научиться забывать дороже,
        чем не научиться. Поэтому перед записью файл перечитывается, и обе
        стороны сливаются аддитивно (core.records.merge_entity).
        """
        try:
            fresh = self.load_world()
        except Exception:
            fresh = world                      # читать не смогли — пишем своё
        fe = fresh.setdefault("entities", {})
        for name, ent in (world.get("entities") or {}).items():
            fe[name] = merge_entity(fe.get(name), ent)
        fo = fresh.setdefault("observations", {})
        for key, obs in (world.get("observations") or {}).items():
            cur = fo.get(key)
            if cur is None:
                fo[key] = obs
                continue
            cur["support"] = max(int(cur.get("support", 1)),
                                 int(obs.get("support", 1)))
            cur["confidence"] = max(float(cur.get("confidence", 0)),
                                    float(obs.get("confidence", 0)))
            cur["negated"] = bool(cur.get("negated")) or bool(obs.get("negated"))
        fu = fresh.setdefault("unknown_words", {})
        for key, u in (world.get("unknown_words") or {}).items():
            cur = fu.get(key)
            if cur is None:
                fu[key] = u
            else:
                cur["count"] = max(int(cur.get("count", 1)), int(u.get("count", 1)))
        # Удаление — это тоже правка. Без этого шага снятие статуса «не знаю»
        # пропадало: слияние видело старую запись чужого снимка и возвращало
        # её, так что выученное слово снова числилось неизвестным.
        for key in resolved:
            if key in fu and key not in (world.get("unknown_words") or {}):
                del fu[key]
        self.save_world(fresh)

    # ── ВНУТРЕННЕЕ ─────────────────────────────────────────

    def _make_resolver(self, entities: dict) -> Callable[[str], Optional[str]]:
        """
        Лемма/форма -> имя сущности во Вселенной. Порядок:
          1) точное совпадение леммы;
          2) форма слова сущности (падеж/число) через переданный резолвер;
          3) лемма, найденная по форме среди самих имён сущностей.
        """
        names = list(entities.keys())
        cache: Dict[str, Optional[str]] = {}
        exact = {n.lower(): n for n in names}

        def resolve(word: str) -> Optional[str]:
            w = (word or "").lower().strip()
            if not w:
                return None
            if w in cache:
                return cache[w]
            out = exact.get(w)
            if out is None and self.known_forms is not None:
                try:
                    hit = self.known_forms(w)
                except Exception:
                    hit = None
                if hit:
                    out = exact.get(hit.lower(), hit.lower())
            if out is None:
                # «кошки» -> «кошка» даже без ядра: лемма уже получена морфологией
                lem = Morph.lemma(w)
                out = exact.get(lem)
            cache[w] = out
            return out

        # имена сущностей, которых нет среди лемм («кошки» как имя сущности):
        # добавляем их формы в кэш, чтобы не гонять резолвер на каждое слово
        if self.known_forms is None:
            for n in names:
                cache.setdefault(Morph.lemma(n), exact.get(n))
        resolve._names = names          # type: ignore[attr-defined]
        return resolve

    def _apply_fact(self, fact: Fact, sf, resolve, entities, observations,
                    source: str, rep: LearningReport) -> None:
        if fact.confidence < self.min_confidence:
            rep.skipped += 1
            return
        head = resolve(fact.head) or fact.head
        tail = resolve(fact.tail) or fact.tail
        if head == tail:
            rep.skipped += 1
            return
        if not (head and tail):
            rep.skipped += 1
            return

        rel_name = REL_ALIAS.get(fact.relation, fact.relation)
        if rel_name not in REL:
            rep.skipped += 1
            rep.notes.append(f"неизвестное отношение {rel_name} — факт пропущен")
            return
        key = f"{head}|{rel_name}|{tail}"
        obs = observations.get(key)
        if obs is None:
            observations[key] = {
                "head": head, "relation": rel_name, "tail": tail,
                "confidence": fact.confidence, "negated": bool(fact.negated),
                "evidence": (fact.evidence or sf.sentence)[:220],
                "source": source, "support": 1, "first_seen": int(time.time()),
            }
            rep.observations_new += 1
        else:
            obs["support"] = int(obs.get("support", 1)) + 1
            # Уверенность берём ЛУЧШЕЕ из свидетельств, а не последнее: второй
            # проход с более слабым разбором не должен отменять уверенный.
            obs["confidence"] = max(float(obs.get("confidence", 0.0)),
                                    fact.confidence)
            if fact.negated:
                obs["negated"] = True
            rep.observations_seen += 1

        if self.core is not None:
            hn = self.core.get_or_create_node(head, NodeType.ENTITY, source=source,
                                             evidence=(fact.evidence or "")[:200])
            tn = self.core.get_or_create_node(tail, self._tail_type(fact),
                                              source=source)
            rel = REL[rel_name]
            self._connect(hn, tn, rel, self._weight(fact.confidence,
                                                    fact.negated, rel,
                                                    int(observations[key].get("support", 1))))

    def _store_action(self, entities: dict, sf, verb: str, source: str,
                      rep: LearningReport) -> None:
        """
        Закрепить глагол КАК ДЕЙСТВИЕ, а не как сущность.

        Записывается узел ACTION: категория «действие», участники из этого же
        предложения (кто может / над чем) и профиль K/M.S, ЕСЛИ он есть в
        справочнике core/actions.py. Профиля нет — мы ЕГО НЕ ВЫДУМЫВАЕМ:
        профиль участвует в матчинге предиктивного цикла, и выдуманное число
        молча порождает ложные предсказания. Отсутствие профиля фиксируется
        громко — в notes отчёта и в журнале.
        """
        prof = profile_of(verb)
        agents = [f.head for f in sf.facts
                  if f.relation == "CAN_DO" and f.tail == verb]
        objects = [f.tail for f in sf.facts
                   if f.relation in ("OBJECT_OF", "LOCATED_IN", "DIRECTION")
                   and f.head == verb]
        rec = {
            "concept": verb,
            "kind": "action",
            "hierarchy": ["действие"],
            "participants": {"agent": agents[:6], "object": objects[:6]},
        }
        if prof:
            rec["action_profile"] = prof
        # mark_learned=False: глагол живёт в своей строке отчёта
        # (learned_actions), а не в списке выученных сущностей. Два списка —
        # чтобы «выучено 3» и «действия 1» не складывались в «выучено 4».
        self._store_entity(entities, verb, rec, source, sf.sentence, rep,
                           mark_learned=False)
        if not prof:
            note = (f"«{verb}»: действие закреплено БЕЗ профиля K/M/S — "
                    f"в справочнике core/actions.py такого нет. Узел и связи "
                    f"работают, но предиктивный цикл это действие не оценит")
            if note not in rep.notes:
                rep.notes.append(note)
                _status("⚠️ " + note)
        if verb not in rep.learned_actions:
            rep.learned_actions.append(verb)

    @staticmethod
    def _mark_learned(rep: "LearningReport", lemma: str) -> None:
        """
        Слово в список выученных — НЕ ПОВТОРЯЯСЬ.

        Один и тот же концепт попадает сюда из двух мест: из основного цикла
        («учитель ответил») и из _store_entity («сущность создана»). Без
        проверки отчёт печатал «выучено: вода, вода, гасить, гасить, пламя,
        пламя» на три слова — ЗАМЕРЕНО в живом GUI. Числа отчёта обязаны
        сходиться со словами, иначе весь отчёт перестаёт быть измерением.
        """
        if lemma and lemma not in rep.learned_concepts:
            rep.learned_concepts.append(lemma)

    def _store_entity(self, entities: dict, lemma: str, data: dict,
                      source: str, evidence: str, rep: LearningReport,
                      mark_learned: bool = True) -> None:
        """
        СЛИЯНИЕ БЕЗ СТИРАНИЯ (правило 1). Существующие непустые поля
        дополняются, но не заменяются; списки объединяются без повторов.
        """
        prev = entities.get(lemma) or {}
        provenanced = dict(data)
        provenanced["provenance"] = [{"source": source, "evidence": evidence[:200],
                                      "at": int(time.time())}]
        # Слияние — НЕ здесь: одна функция на все каналы (core/storage.merge_entity),
        # иначе правила «что можно перезаписать» разъедутся по модулям.
        merged = merge_entity(prev, provenanced)
        merged.setdefault("concept", lemma)
        merged["learned_from"] = source
        entities[lemma] = merged
        if prev:
            rep.entities_updated += 1
        else:
            rep.entities_created += 1
            if mark_learned:
                self._mark_learned(rep, lemma)
        if self.core is not None and hasattr(self.core, "integrate_knowledge"):
            try:
                self.core.integrate_knowledge({"concept": lemma,
                                               "properties": merged.get("properties", []),
                                               "actions": [a if isinstance(a, str) else a.get("name", "")
                                                           for a in merged.get("actions", [])],
                                               "category": (merged.get("hierarchy") or [None])[0]},
                                              weight=200)
            except Exception:
                pass

    # ── мелочь, но с решением ──────────────────────────────

    @staticmethod
    def _connect(hn, tn, rel: RelationType, weight: int) -> None:
        """
        Вес НИКОГДА не понижаем. Две фразы об одном факте не должны приводить
        к худшей памяти, чем была: повторное свидетельство либо усиливает,
        либо оставляет как было.
        """
        cur = hn.out_synapses_typed.get(tn.name) or {}
        if int(cur.get("weight", 0)) >= weight and cur:
            return
        hn.connect_to(tn, rel_type=rel, weight=weight)

    @staticmethod
    def _weight(conf: float, negated: Optional[bool], rel: RelationType,
                support: int = 1) -> int:
        base = 90 + 130 * conf if rel in (RelationType.CAN_DO,
                                          RelationType.HAS_PROPERTY) else 60 + 100 * conf
        base += min(40, 10 * max(0, support - 1))
        if negated:
            base /= 3.0
        return int(max(8, min(255, base)))

    @staticmethod
    def _tail_type(fact: Fact) -> int:
        if fact.relation in ("CAN_DO", "PURPOSE"):
            return NodeType.ACTION
        if fact.relation in ("HAS_PROPERTY", "ABOUT", "VALUE", "CIRCUMSTANCE",
                           "CONDITION"):
            return NodeType.PROPERTY
        return NodeType.ENTITY

    @staticmethod
    def _node_type(obs: dict, which: str) -> int:
        """Тип узла при восстановлении из наблюдений (типа в записи нет)."""
        rel = obs.get("relation", "")
        if which == "tail":
            if rel == "CAN_DO":
                return NodeType.ACTION
            if rel in ("HAS_PROPERTY", "ABOUT", "VALUE", "CIRCUMSTANCE", "CONDITION"):
                return NodeType.PROPERTY
        return NodeType.ENTITY

    @staticmethod
    def _role_of_lemma(sentence, lemma: str) -> str:
        for t in sentence.tokens:
            if t["lemma"] == lemma and t["content"]:
                return t["role"]
        return "subject"


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    world = {"entities": {"кошка": {"concept": "кошка", "hierarchy": ["животные"],
                                    "properties": [], "actions": []}},
             "observations": {}, "unknown_words": {}}
    lr = Learner(lambda: world, lambda w: None)
    report = lr.learn_from_text(
        "Кошка сидит на широком подоконнике. Пушистая кошка боится грозы. "
        "У кошки есть хвост. Вася — это инженер.")
    print(report.as_text())
    print("\nнаблюдения:")
    for k, v in world["observations"].items():
        print(f"  {k:52s} conf={v['confidence']} support={v['support']}")
    print("\nнеизвестные слова:", list(world["unknown_words"]))
