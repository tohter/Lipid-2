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
from core.morph import Morph
from core.records import merge_entity
from core.storage import NodeType, RelationType

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
    notes: List[str] = field(default_factory=list)
    quality: Dict[str, bool] = field(default_factory=dict)

    def as_text(self) -> str:
        acc = (self.known_words / self.content_words) if self.content_words else 0.0
        lines = [
            "📚 Обучение из текста",
            f"  предложений: {self.sentences}   значимых слов: {self.content_words}"
            f"   распознано: {self.known_words} ({acc:.0%})   незнакомых: {self.unknown_words}",
            f"  фактов извлечено: {self.facts}   новых наблюдений: {self.observations_new}"
            f"   подтверждено существующих: {self.observations_seen}",
            f"  сущностей создано: {self.entities_created}   дополнено: {self.entities_updated}",
            f"  обращений к Учителю: {self.teacher_calls}"
            f"   успешно: {self.teacher_success}   пропущено фактов: {self.skipped}"
            f"   низкоуверенных: {self.low_confidence}",
        ]
        if self.learned_concepts:
            lines.append("  выучено: " + ", ".join(self.learned_concepts[:12]))
        if self.unknown:
            lines.append("  осталось неизвестным: " + ", ".join(self.unknown[:12]))
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

        # 1) словарь: кого мы знаем, кого нет
        vocab: Dict[str, dict] = {}
        for sf in sentences:
            for lemma in sf.content_lemmas:
                vocab.setdefault(lemma, {"node": self._role_of_lemma(sf, lemma),
                                         "evidence": sf.sentence})
        rep.content_words = len(vocab)

        for lemma, info in sorted(vocab.items()):
            canon = resolve(lemma)
            if canon:
                rep.known_words += 1
                info["canon"] = canon
                continue
            unknowns_seen = unknowns.get(lemma)
            if unknowns_seen:
                unknowns_seen["count"] = int(unknowns_seen.get("count", 1)) + 1
                rep.unknown_words += 1
                rep.unknown.append(lemma)
                info["canon"] = None
                continue
            rep.unknown_words += 1
            rep.unknown.append(lemma)
            entity = None
            if allow_teacher and self.teacher is not None \
                    and rep.teacher_calls < self.max_teacher_calls:
                rep.teacher_calls += 1
                try:
                    entity = self.teacher(lemma)
                except Exception as exc:
                    rep.notes.append(f"учитель по «{lemma}» упал: {type(exc).__name__}")
                    entity = None
            if entity and (entity.get("hierarchy") or entity.get("properties")):
                rep.teacher_success += 1
                self._store_entity(entities, lemma, entity, source,
                                   info["evidence"], rep)
                info["canon"] = lemma
                rep.learned_concepts.append(lemma)
            else:
                # Сущность НЕ создаём (правило 2), но слово помним: иначе тот
                # же вопрос гонит LLM при каждом упоминании.
                unknowns[lemma] = {"first_seen": int(time.time()), "count": 1,
                                   "evidence": info["evidence"][:200],
                                   "source": source}
                info["canon"] = None

        # 2) связи
        for sf in sentences:
            for fact in sf.facts:
                self._apply_fact(fact, sf, resolve, entities, observations,
                                 source, rep)

        self._commit(world)
        rep.facts = sum(len(sf.facts) for sf in sentences)
        rep.low_confidence = sum(1 for sf in sentences for f in sf.facts
                                 if f.confidence < self.min_confidence)
        rep.new_concepts = sorted({k for k in entities
                                   if (entities[k] or {}).get("learned_from") == source})
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

    def _commit(self, world: dict) -> None:
        """
        СОХРАНЕНИЕ = ПЕРЕЧИТАТЬ И СЛИТЬ, а не «перезаписать свой снимок».

        Между чтением мира и записью файл может изменить КТО-ТО ДРУГОЙ:
        локальный Учитель — отдельный процесс, он пишет в lipid_world.json
        своим tool-вызовом ровно в тот момент, когда мы «думаем» над
        предложением. Его ответ был получен за реальные деньги и за секунды
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

    def _store_entity(self, entities: dict, lemma: str, data: dict,
                      source: str, evidence: str, rep: LearningReport) -> None:
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
            rep.learned_concepts.append(lemma)
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
