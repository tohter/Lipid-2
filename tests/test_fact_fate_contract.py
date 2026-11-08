"""СУДЬБА ВЫСКАЗЫВАНИЯ: закон / событие / вымысел — НЕ ХРАНИТЬ ЛОЖЬ В БАЗЕ.

ЗАЧЕМ ЭТОТ ФАЙЛ. Хозяин поставил закон (09.11 ночь): заведомо выдуманная связь не
имеет права осесть в постоянном мире. Симулятор имеет право ВЫДУМАТЬ посылку
(«ботинок читает стихи») в рабочей памяти, но не имеет права записать её во
Вселенную. Замер (experiments/measure_nonsense_footprint.py, 09.11) показал,
что сегодня никакое поле этого не различает:

    текст  «Ботинок читает стихи. Ветер раздувает пламя.»
    мир    ботинок|CAN_DO|читать   ground=observation weight=55 support=1
           читать|OBJECT_OF|стих   ground=observation weight=55 support=1
           ветер|CAN_DO|раздувать  ground=observation weight=55 support=1

Ботинок, читающий стихи, и законный факт выглядят в базе ОДИНАКОВО по всем
полям authority (ground / weight / support / weight_law / source). Поля
«вымысел» в записи нет.

ЗАМЕР, ОПРЕДЕЛИВШИЙ ГРАНИЦУ (не «правда/ложь», а «закон/случай»):

    «Ботинок читает стихи.»    -> 2 статические связи, 0 событий
    «Ботинок прочитал стих.»   -> 0 статических связей, 1 событие
    «Ветер раздувает пламя.»   -> 2 статические связи, 0 событий
    «Ветер раздул пламя.»      -> 0 статических связей, 1 событие
    «Маша была жирная.»        -> 1 СТАТИЧЕСКАЯ связь маша|HAS_PROPERTY|жирный

Последняя строка — дыра, которую запирает этот файл. Время сказуемого решает
судьбу высказывания, но решает НЕДОЛЯ: личный глагол в прошедшем времени
уходит в события, а ПРОШЕДШАЯ СВЯЗКА — нет. «Маша была жирная» становится
вечным свойством Маши в постоянном мире. Маша не обязана быть жирной вечно,
а база уже утверждает это как род.

ВТОРАЯ ПОЛОВИНА (замерена здесь же, ЭТИМ КИРПИЧОМ НЕ ЧИНИТСЯ и заперта как
дыра классом TestCopulaInstrumentHoleLeftAlive): связочная форма в
Творительном падеже — «Вася был инженером» — разбирается вообще БЕЗ
сказуемого (`_predicate()` исключает COPULAS, а ветка нулевой связки требует
тире или «это», которых во фразе нет), и именная часть уезжает косым падежом:

    «Вася был инженер.»    -> вася IS_ROLE инженер     (починка aa5f2e9)
    «Вася был инженером.»  -> вася INSTRUMENT инженер  (ЛОЖЬ: Вася — инструмент)

Память считает пункт закрытым. Замер 09.11 (ночь): закрыта только НЕЕстественная (именительная)
форма, живой русский вариант лжёт. Названо отдельно, чтобы следующий
исполнитель не решил, что одна правка закрывает обе.

ЧТО ЗАПРЕЩЕНО ДЕЛАТЬ С ЭТИМ ФАЙЛОМ. Дешёвый способ сделать его зелёным —
перестать запускать в мир ВСЁ прошедшее время. Тогда утонет статика, которая
в рассказе стоит в прошедшем («Жена Егора была учительницей» обязана оставить
ребро жена->егор; урок заперт в test_kinship_edge_contract.py: ребро, утонувшее
в событии, — это 0 из 7 пар путей при живом экстракторе). Поэтому здесь ДВА
направленных требования в обе стороны: случай НЕ должен становиться законом,
а закон НЕ обязан тонуть из-за того, что о нём сказали в прошедшем времени.

ТРЕБОВАНИЯ НЕ ОПРЕДЕЛЯЮТ РЕМОНТ. Можно отправить связочную фразу в события,
можно писать её в рабочую память симулятора, можно держать в карантине до
решения Хозяина, — но постоянный мир обязан остаться чистым, а отказ обязан
быть назван в отчёте.
"""
import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.learner import Learner            # noqa: E402
from core.extract import SimpleExtractor     # noqa: E402

# ── ПАРА, ЗАДАННАЯ ХОЗЯИНОМ (её и запираем, а не удобную свою) ──
EVENT_SENTENCE = "Ботинок прочитал стих."      # случай  -> событие, НЕ связь
LAW_SENTENCE = "Ветер раздувает пламя."        # закон   -> связь, НЕ событие
PAST_PROPERTY = "Маша была жирная."            # дыра №1: случай стал законом
PRESENT_PROPERTY = "Маша толстая."             # закон, обязан остаться в мире
PAST_POSSESSIVE = "Жена Егора была учительницей."
COPULA_INSTRUMENT = "Вася был инженером."      # дыра №2, НЕ чинится здесь

# Поля записи наблюдения СЕГОДНЯ. Ни одно не означает «вымысел»: замок на
# схему, а не на флаг. Заведёт кто-то поле `fiction` — тест спросит, чем.
OBS_FIELDS = {"head", "relation", "tail", "confidence", "negated", "evidence",
              "source", "support", "first_seen", "lineage", "ground",
              "weight", "weight_law", "provenance", "staged_candidate"}

COPULA_WORDS = {"был", "была", "были", "было", "будет", "стал", "стала",
                "стали", "стало", "стать", "является", "являлся", "станет"}


def learn(texts, source="fate_lock"):
    """Урок без учителя и без ядра: тест не имеет права ни писать в живую
    Вселенную, ни звать LLM."""
    world = {"entities": {}, "observations": {}, "unknown_words": {}}
    lrn = Learner(load_world=lambda: dict(world),
                  save_world=lambda w: world.update(w),
                  teacher=lambda w: None, core=None, max_teacher_calls=0)
    reports = [lrn.learn_from_text(t, source=source, allow_teacher=False)
               for t in texts]
    return world, reports


def links_of(world):
    return set(world.get("observations", {}))


def events_of(text):
    return list(SimpleExtractor().extract(text).events)


def facts_of(text):
    return list(SimpleExtractor().extract(text).facts)


class TestGreenPair(unittest.TestCase):
    """Зелёная пара хозяина. Оба направления обязательны: без второго теста
    «починка» равна «запретить прошедшее время вообще»."""

    def test_event_sentence_is_event_and_not_law(self):
        self.assertTrue(events_of(EVENT_SENTENCE),
                        "прошедшее время не дало события: %r" % EVENT_SENTENCE)
        self.assertTrue(all(f.is_event for f in facts_of(EVENT_SENTENCE)),
                        "факт прошедшего времени помечен как закон: %s"
                        % facts_of(EVENT_SENTENCE))
        world, _ = learn([EVENT_SENTENCE])
        self.assertFalse(links_of(world),
                         "событие записано в мир как связь: %s"
                         % sorted(links_of(world)))

    def test_law_sentence_is_law_not_event(self):
        self.assertFalse(events_of(LAW_SENTENCE),
                         "настоящее время разобрано как событие")
        world, _ = learn([LAW_SENTENCE])
        self.assertTrue(links_of(world), "закон не попал в мир")

    def test_recorded_event_reaches_world_events(self):
        """Событие обязано дойти до секции events, иначе «не записали в
        онтологию» означает «выбросили»."""
        world, _ = learn([EVENT_SENTENCE])
        self.assertTrue(world.get("events"),
                        "событие не дошло до мира: %s" % sorted(world))


class TestPastStateIsNotEternalLaw(unittest.TestCase):
    """ДЫРА №1. «Маша была жирная» не имеет права стать вечным свойством."""

    def test_past_state_absent_from_static_world(self):
        world, _ = learn([PAST_PROPERTY])
        self.assertFalse([k for k in links_of(world)
                          if k.startswith("маша|")],
                         "«Маша была жирная» записана как вечная связь: %s"
                         % sorted(links_of(world)))

    def test_present_property_still_enters_world(self):
        """НЕЛЬЗЯ закрыть дыру, вырубив именные свойства: «Маша толстая» —
        именная часть в настоящем времени, ей место в мире (замер: даёт
        маша|IS_ROLE|толстой)."""
        world, _ = learn([PRESENT_PROPERTY])
        self.assertTrue([k for k in links_of(world) if k.startswith("маша|")],
                        "именная часть настоящего времени перестала доходить "
                        "до мира: %s" % sorted(links_of(world)))

    def test_past_relation_does_not_drown_with_state(self):
        """СТАТИКА В ПРОШЕДШЕМ НЕ ТОНЕТ (урок test_kinship_edge_contract).
        «Была учительницей» — случай, а связь «жена Егора» — отношение: оно
        обязано пережить разбор. Падеж связи здесь не мерится: чем является
        жена относительно Егора при пустом сказуемом (PART_OF или OBJECT_OF)
        — отдельная дыра, названа в докстринге файла."""
        world, _ = learn([PAST_POSSESSIVE])
        self.assertTrue([k for k in links_of(world)
                         if k.startswith("жена|") and k.endswith("|егор")],
                        "связь утонула вместе со сказуемым: %s"
                        % sorted(links_of(world)))

    def test_future_copula_is_not_an_event(self):
        """БУДУЩЕЕ — план, а не происшествие (тот же закон, что в
        `_collect_events`: «настоящее и будущее описывают закон или план»).
        Без требования именно ПРОШЕДШЕЙ формы связки «Вася будет
        инженером» уезжал бы в события, и план превращался бы в
        состоявшийся факт (замер M8 на этой фразе: вася|IS_ROLE|инженер с
        predicate=event)."""
        for sentence in ("Вася будет инженер.", "Вода будет горячая."):
            with self.subTest(sentence=sentence):
                sf = SimpleExtractor().extract(sentence)
                self.assertTrue(sf.facts, "фраза потеряна: %s" % sentence)
                self.assertFalse([f for f in sf.facts if f.is_event],
                                 "будущее время переведено в событие: %s"
                                 % [(f.head, f.relation, f.tail, f.predicate)
                                    for f in sf.facts])

    def test_only_subject_state_becomes_event(self):
        """ПРАВО НА «СОСТОЯНИЕ» ЕСТЬ ТОЛЬКО У СКАЗУЕМОГО подлежащего. Замер
        09.11 на «В комнате были стол старый и шкаф новый»: определитель
        «новый» относится к «шкаф», а не к подлежащему «стол». Без
        проверки головы чужое состояние уезжает из статики, хотя сказуемое
        было про стол."""
        sf = SimpleExtractor().extract(
            "В комнате был стол старый и шкаф новый.")
        by_key = {f.key(): f.predicate for f in sf.facts}
        self.assertEqual("fact", by_key.get(("шкаф", "HAS_PROPERTY", "новый")),
                         "чужое состояние переведено в событие: %s" % by_key)

    def test_state_event_is_not_forgotten(self):
        """Отброшенное из онтологии не обязано исчезать совсем: случай
        живёт в динамике. Проверка на то, что мир НЕ потерял фразу."""
        world, _ = learn([PAST_PROPERTY])
        self.assertTrue(world.get("events") or world.get("states")
                        or world.get("quarantine"),
                        "фраза исчезла из мира бесследно: %s" % sorted(world))


class TestRejectionIsNamed(unittest.TestCase):
    """ОТКАЗ ОБЯЗАН БЫТЬ НАЗВАН. Пустой приём неотличим от «в тексте нечего
    было брать» (урок про заметки разбора, memory.txt п.B)."""

    REASON = r"(не записан|не пош|отклон|событ|случа|вымысл|динамик|прошедш|"
    # Одна строка обязана назвать ВСЁ: и связку, и причину. Иначе отчёт
    # наберёт нужные слова из чужих заметок («Учитель не подключён») и
    # притворится объяснением (мутация M3: заметка стала «маша HAS_PROPERTY
    # жирный» — слова про связь есть, причины нет).
    REASON += r"состоян)"

    def test_rejection_is_named_in_report(self):
        _world, reports = learn([PAST_PROPERTY])
        rep = reports[0]
        lines = list(rep.notes) + list(getattr(rep, "parse_notes", []))
        named = [n for n in lines
                 if "была" in n and re.search(self.REASON, n, re.I)
                 and "маша" in n and "жирн" in n]
        self.assertTrue(named,
                        "связь отброшена, но ни одна строка отчёта не называет "
                        "её причину целиком (связка + факт + причина): %s"
                        % (lines,))

    def test_unknown_word_stays_in_queue(self):
        """Незнакомое слово НЕ исчезает вместе с отвергнутым фактом: урок
        про него ещё не выучен."""
        world, _ = learn(["Зябра была жирная."])
        self.assertIn("зябра", world.get("unknown_words", {}),
                      "отказ поедает unknown_words: %s"
                      % sorted(world.get("unknown_words", {})))

    def test_schema_has_no_fiction_field(self):
        """Ни одно поле записи наблюдения не означает «вымысел»: различение
        закона и сказки обязано жить в ПУТИ записи, а не в бите."""
        world, _ = learn([LAW_SENTENCE, PAST_PROPERTY])
        for key, rec in world.get("observations", {}).items():
            self.assertTrue(set(rec) <= OBS_FIELDS,
                            "запись наблюдения обзавелась полем без "
                            "объяснения: %s -> %s vs %s"
                            % (key, sorted(rec), sorted(OBS_FIELDS)))


class TestCopulaInstrumentHoleLeftAlive(unittest.TestCase):
    """ДЫРА №2 НЕ ЗАКРЫТА ЭТИМ КИРПИЧОМ, И ЭТО ЗАПЕРТО (тот же приём, что
    TestObliqueZoneLeftAlone в test_kinship_edge_contract: дыра, которая не
    названа, перестаёт быть дырой и становится «нормой»)."""

    def test_instrument_lie_is_not_called_fixed(self):
        got = {f.key() for f in facts_of(COPULA_INSTRUMENT)}
        self.assertIn(("вася", "INSTRUMENT", "инженер"), got,
                      "дыра №2 закрыта НЕ этим коммитом: если чинишь — "
                      "убирай тест и пиши кирпич отдельно. Сейчас: %s"
                      % sorted(got))


class TestPossessionBranchHoleLeftAlive(unittest.TestCase):
    """ДЫРА №4, НЕ ЗАКРЫТА ЭТИМ КИРПИЧОМ. «У Маши была жирная кошка» —
    possession-ветка уходит ДО шага судьбы, и там прошедшее время не видно
    вовсе. Замер 09.11 (ночь): `кошка|HAS_PROPERTY|жирный` остаётся ВЕЧНЫМ
    свойством и — хуже того — молчит: ни события, ни заметки об отказе.

    Заперта именно как МОЛЧАНИЕ, а не как исход: владение («была» =
    «имелось») — закон, и автоматически тащить его в события нельзя, это
    сломало бы зелёную пару. Но тихий отказ обязан быть назван, иначе
    следующий прогон примет «в тексте нечего было брать» за «время дошло».
    Чинится отдельным кирпичом вместе с possession-глаголом."""

    POSSESSION = "У Маши была жирная кошка."

    def test_possession_branch_is_silent_today(self):
        sf = SimpleExtractor().extract(self.POSSESSION)
        by_key = {f.key(): f.predicate for f in sf.facts}
        self.assertEqual("fact", by_key.get(("кошка", "HAS_PROPERTY", "жирный")),
                         "дыра №4 закрыта НЕ этим коммитом: переноси тест в "
                         "основной контракт. Сейчас: %s" % by_key)
        self.assertIn(("маша", "HAS_PART", "кошка"), by_key,
                      "владение потеряно: %s" % by_key)
        self.assertFalse(sf.events,
                         "possession-ветка заговорила НЕ этим коммитом: %s"
                         % sf.events)


class TestAmbiguityRefusesLoudly(unittest.TestCase):
    """ГДЕ РАЗБОР НЕ УВЕРЕН, ОТКАЗ ГРОМКИЙ. Две связочные формы в прошедшем
    — время сказуемого не выбирается; молчание было бы неотличимо от «связки
    не было», а исход разный."""

    AMBIGUOUS = "Вася был был инженер."

    def test_ambiguous_tense_leaves_link_and_says_why(self):
        sf = SimpleExtractor().extract(self.AMBIGUOUS)
        links = [f for f in sf.facts if f.relation == "IS_ROLE"]
        self.assertTrue(links, "двусвязочная фраза потеряла связь: %s"
                        % sf.facts)
        self.assertTrue(all(not f.is_event for f in links),
                        "неопределённое время перевело связь в состояние: %s"
                        % links)
        self.assertTrue([n for n in sf.notes if "не определено" in n],
                        "отказ не назван: %s" % sf.notes)


class TestInchoativeHoleLeftAlive(unittest.TestCase):
    """ДЫРА №3, НЕ ЗАКРЫТА ЭТИМ КИРПИЧОМ. «Маша СТАЛА жирная» — смена
    состояния, а оседает вечным свойством: «стать» — содержательный глагол
    (content=True, role=action), и фильтр связки его не берёт. Замер
    09.11: `маша|HAS_PROPERTY|жирный`, predicate=fact.

    НЕ ЧИНИТСЯ ЗДЕСЬ сознательно: инхоатив — своё решение (что делать с
    «стал читать», «становится холодно», сравнительными «выше», «старше»),
    и расширять на него фильтр связки одной строкой значит перевести в
    события половину рассказов. Измеренная цена: без guard'а content/role
    эта фраза уезжает в события (замер M7 батареи, 34 фразы разошлись ровно
    на одном)."""

    def test_inchoative_is_still_a_law(self):
        sf = SimpleExtractor().extract("Маша стала жирная.")
        got = [(f.head, f.relation, f.tail, f.predicate) for f in sf.facts]
        self.assertIn(("маша", "HAS_PROPERTY", "жирный", "fact"), got,
                      "инхоатив закрыт НЕ этим коммитом: если чинишь — "
                      "пиши отдельный кирпич и убери тест. Сейчас: %s" % got)


class TestFateCoins(unittest.TestCase):
    """МОНИТЫ (цена права на ошибку). Числа сняты ЗАМЕРОМ до правки; «0»
    пишется только там, где поиск ошибки реально был."""

    def test_corpus_has_no_past_copula(self):
        """Замер до правки: золотой корпус (35 + 49 фраз) НЕ содержит ни
        одной связочной фразы в прошедшем, поэтому метрику экстрактора эта
        починка не двигает. Тест следит, чтобы корпус не поехал молча."""
        cop = 0
        for name in ("corpus_dev.json", "corpus_heldout.json"):
            with open(os.path.join(ROOT, "experiments", name),
                      encoding="utf-8") as fh:
                cases = json.load(fh)["cases"]
            for c in cases:
                lems = {w.lower().strip(".,;:")
                        for w in c["text"].replace(",", " ").split()}
                if lems & COPULA_WORDS:
                    cop += 1
        self.assertEqual(0, cop,
                         "в корпусе появились связочные фразы: метрику "
                         "экстрактора надо переснять до правки")

    def test_nonsense_footprint_measured_is_five_links(self):
        """Замер 09.11 (experiments/measure_nonsense_footprint.py): связка из
        ТРЁХ выдуманных фраз оставляет в мире 5 связей. Число — цена
        «ничего не трогать»: каждая из них выглядит законной по всем полям.
        Изменение числа обязано быть объяснённым, а не молчаливым."""
        world, _ = learn(["Ботинок читает стихи.", "Ветер раздувает пламя.",
                          "Глаз не может оторваться от тебя."])
        self.assertEqual(5, len(links_of(world)),
                         "след вымысла в мире изменился: %s"
                         % sorted(links_of(world)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
