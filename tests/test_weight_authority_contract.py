#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Вес связи принадлежит ОСНОВАНИЮ, а не тому, кто громче высказался.

ШАГ Г (08.11, шаги 2.2/2.3 босса). Файл закрывает пять мутаций, которые
ВЫЖИЛИ в первом прогоне батареи (/tmp/mutate_weight.py): выжившая мутация —
это незакрытая дыра, а не неудачная мутация, и ниже написан тест на каждую.

ЧТО ИЗМЕРЕНО ДО ПРАВКИ (experiments/measure_weight_authority.py):
  * пять каналов с ОДНИМ основанием давали веса 161..220 — разброс 59 очков
    из 255 там, где различались только числа, которые каналы назвали о самих
    себе;
  * безымянная модель с самоотчётом 1.00 (вес 220) перекрыла ДВЕ названные
    книги (вес 171) — сила связи досталась самой громкой;
  * живой мир (data/memory/lipid_world.json.pre-clean): три связи с НУЛЁМ
    названных оснований весили 120..210;
  * `ground` не имел ВЛАСТИ над весом: testimony, owner и rule давали одно и
    то же число, то есть «слово Хозяина» и «мне показалось» были равны.

ЗАКОН, КОТОРЫЙ ЗАПРЕЮТ ЭТИ ТЕСТЫ:
  1. самоотчёт (confidence) не входит в вес НИ КАК;
  2. силу дают тип основания и число ОТДЕЛЬНЫХ названных оснований;
  3. «назвать нечем» — отдельная, БОЛЕЕ СЛАБАЯ ступень: у писателя есть
     материальный интерес назвать документ;
  4. одиночный и многократно повторенный чужой пересказ НЕ перешагивает
     порог диагноза «онтология без опыта» — ни при каком числе оснований;
  5. число, посчитанное сном из силы опыта, принадлежит опыту: свидетель не
     имеет права его пересчитать, читатель — перекрыть «максимумом»;
  6. приехавшее в payload основание не имеет силы даже через писателя.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lang.ru.lemma import install  # noqa: E402
install()

from core.records import (                          # noqa: E402
    EXPERIENCE_WEIGHT_LAW, GROUND_OBSERVATION, GROUND_OWNER,
    GROUND_WEIGHT_LAW,
    GROUND_TESTIMONY, add_witness, ground_of, independent_bases,
    observation_weight, record_weight, stamp_ground, weight_from_ground)
from brain.lipid2_brain import Lipid2Brain          # noqa: E402
# Порог диагноза ontology_without_experience и пол опыта. Держим их здесь
# ЯВНО: весь смысл теста в том, что вес обязан быть сопоставим с порогом, а
# молча разойтись с ним он не имеет права.
from brain.sleep import CONFLICT_WEIGHT_MIN, WEIGHT_FLOOR   # noqa: E402


def rec(**ки):
    """Минимальная запись наблюдения: только то, что читают писатели."""
    р = {"head": "дождь", "relation": "CAN_DO", "tail": "гасить",
         "confidence": 0.5, "negated": False, "support": 1,
         "source": "wiki", "evidence": "фраза", "lineage": "doc:дождь"}
    р.update(ки)
    return р


def pereskaz(n):
    """Пересказ: РОВНО n названных оснований, самоотчёт на максимум.

    Основания собираются в журнале, а не «на глаз»: иначе тест сверялся бы с
    собственной выдумкой, а не с поведением independent_bases.
    """
    lin = [f"doc:пересказ-{i}" for i in range(n)]
    журнал = [{"source": "wiki", "evidence": "т", "lineage": x} for x in lin]
    return rec(source="wiki", confidence=1.0,
               lineage=(lin[0] if lin else ""), provenance=журнал)


def brain_in(dir_):
    b = Lipid2Brain(memory_dir=dir_)
    b.ask_user_enabled = False
    b.use_llm_teacher = False
    b.use_wiki_teacher = False
    return b


def obs(b):
    return b._load_world()["observations"]


class TestSelfReportHasNoWeight(unittest.TestCase):
    """Мутация «самоотчёт вернулся в вес» обязана быть убита."""

    def test_1_confidence_changes_nothing(self):
        """Две записи, различающиеся ТОЛЬКО самоотчётом, весят ОДИНАКОВО.

        Это ядро шага Г и ровно та мутация, которую первый прогон батареи НЕ
        поймал: прежняя сверка сравнивала record_weight с weight_from_ground,
        то есть проверяла «число совпало с формулой», но не спрашивала, ОТ
        ЧЕГО формула. Здесь сравниваются ДВЕ записи между собой — подмена
        самоотчёта в формулу краснеет немедленно.
        """
        quiet = rec(confidence=0.05)
        loud = rec(confidence=1.00)
        self.assertEqual(record_weight(quiet), record_weight(loud),
                         "число, которое канал назвал о себе, изменило силу "
                         "связи: громкий канал снова сильнее обоснованного")
        self.assertEqual(observation_weight(quiet), observation_weight(loud))

    def test_2_loud_anonymous_loses_to_two_named_books(self):
        """Модель с self-report 1.00 НЕ имеет права перекрыть две книги.

        Замер старой формулы: 220 против 171 — сила досталась самой громкой.
        """
        loud = rec(source="llm", lineage="", confidence=1.00, provenance=[])
        books = rec(source="wiki", confidence=0.4,
                    lineage="doc:книга-1",
                    provenance=[{"source": "wiki", "evidence": "т",
                                 "lineage": "doc:книга-2"}])
        self.assertEqual(2, independent_bases(books),
                         "две названные книги обязаны быть двумя основаниями")
        self.assertEqual(0, independent_bases(loud))
        self.assertGreater(record_weight(books), record_weight(loud),
                           "шум перебил источник: ровно та дыра, из-за которой"
                           " формулу сняли")

    def test_3_owner_beats_any_amount_of_retelling(self):
        """Слово Хозяина сильнее пересказа ПРИ ЛЮБОМ числе его оснований."""
        хозяин = rec(source="user", lineage="owner", confidence=0.3,
                     provenance=[])
        for n in range(1, 13):
            self.assertGreater(record_weight(хозяин),
                               record_weight(pereskaz(n)),
                               f"при {n} основаниях пересказ переспорил "
                               f"Хозяина")


class TestNamedSourceIsCheaperThanNothing(unittest.TestCase):
    """Ступень «назвать нечем» обязана стоить ДОШЕВ названного основания."""

    def test_4_naming_a_document_is_always_paid(self):
        """Вес растёт от 0 к 1, 2, 3 основаниям — монотонно и строго.

        Убитая мутация: «безымянное = одно названное». Сравнять их значило бы
        отобрать у писателя единственный материальный интерес назвать
        документ, то есть обессмыслить весь кирпич происхождения.
        """
        for ground in (GROUND_TESTIMONY, GROUND_OBSERVATION, GROUND_OWNER):
            ряд = [weight_from_ground(ground, n) for n in range(0, 4)]
            self.assertEqual(ряд, sorted(ряд), f"{ground}: ряд не убывает")
            self.assertEqual(len(set(ряд)), len(ряд),
                             f"{ground}: назвать основание и не назвать дают "
                             f"одно число {ряд} — называть невыгодно")

    def test_5_writers_do_not_grant_a_lone_basis_they_cannot_name(self):
        """Ни один писатель не рисует вес «одно основание» безымянной фразе.

        Письменно: learner и brain обязаны брать число оснований из того же
        lineage, что записывают рядом. Замер до правки: безымянная фраза от
        канала "text" получала 110 вместо 55 — силу названной книги.
        """
        with tempfile.TemporaryDirectory() as dir_:
            b = brain_in(dir_)
            b._record_observation("дождь", {"actions": ["гасить"]},
                                  confidence=0.9, evidence="фраза",
                                  source="llm")
            r = obs(b)["дождь|CAN_DO|гасить"]
            self.assertEqual("", r.get("lineage"),
                             "писатель сам себе придумал происхождение")
            self.assertEqual(0, independent_bases(r))
            # СВЕРЯЕТСЯ ЗАПИСАННОЕ ЧИСЛО, а не вывод читателя. Первый вариант
            # теста звал record_weight, тот пересчитывает по коллекции и
            # ЧИНИЛ враньё писателя на месте — мутация «подставить 1»
            # выживала, а в файл между тем ложло число, которое кто-нибудь
            # однажды прочитает через .get("weight").
            self.assertEqual(weight_from_ground(ground_of("llm"), 0, False),
                             r["weight"],
                             "безымянной связи записана сила названной")
            self.assertEqual(weight_from_ground(ground_of("llm"), 0, False),
                             record_weight(r))

    def test_5b_learner_does_the_same(self):
        """Обучающий — второй писатель, и он обязан врать НЕ больше.

        Закон живёт, пока его соблюдают ОБА места записи: brain проверяли,
        learner не проверял никто, и именно он пишет основную массу наблюдений
        (кирпич 2 и кирпич 6 начинались одинаково — «правило продублировано и
        уже разошлось»).
        """
        import json
        from brain.learner import Learner
        with tempfile.TemporaryDirectory() as dir_:
            path = os.path.join(dir_, "w.json")

            def read():
                if os.path.exists(path):
                    return json.load(open(path, encoding="utf-8"))
                return {"entities": {}, "observations": {},
                        "unknown_words": {}}

            def write(w):
                json.dump(w, open(path, "w", encoding="utf-8"),
                          ensure_ascii=False)
            lr = Learner(load_world=read, save_world=write)
            lr.learn_from_text("Дождь гасит пожар.", source="text",
                               allow_teacher=False)
            r = read()["observations"]["дождь|CAN_DO|гасить"]
            self.assertEqual("", r.get("lineage"))
            self.assertEqual(weight_from_ground(ground_of("text"), 0, False),
                             r["weight"],
                             "обучающий выдал безымянной фразе силу названной")


class TestRetellingCannotReachTheDiagnosisFloor(unittest.TestCase):
    """Ложный замок, найденный мутированием: потолок пересказа был недостижим.

    TESTIMONY_LONE_MAX = 140 выглядел защитой, но BASE_CAP = 40 ограничивал
    рост раньше: 40 + 40 = 80, и строка `min(..., 140)` не защищала НИЧЕГО.
    Мутация «сними потолок, добавь 120» проходила МОМО неё и выводила
    testimony за порог диагноза — то есть проверка была заперта на число,
    которое нельзя получить. Здесь заперт ИНВАРИАНТ, который мутация ломает
    при любом способе подкупа.
    """

    def test_5c_alias_branch_is_under_the_same_law(self):
        """Третье место записи — ветка «вариант имени»; было цело лишь случайно.

        Открыто мутированием: записи SYNONYM_OF создаются в learner, но ни
        один тест их не СОЗДАВАЛ (тесты только ПОТРЕБЛЯЮТ SYNONYM_OF —
        thinker, alias, golden corpus). Мутация «писать запись мимо двери
        закона» в этой ветке ВЫЖИЛА, потому что проверять было нечего: без
        stamp_ground запись выходит вообще без `weight`, то есть сон и
        читатель её просто не видят.

        ПОМНЯ О ЗАКОНЕ ИМЁН (experiments/ENTITY_FIELDS.md): единственное место
        правды о синонимах — поле `aliases` на ЗАПИСИ СУЩНОСТИ; запись
        SYNONYM_OF рядом существует как пережиток append-only перечисления и
        ничего решать не должна. Тест поэтому требует от неё только одного:
        если она уже пишется, она обязана подчиняться тому же закону веса,
        что и все остальные записи, а не ходить вне его.
        """
        import json
        from brain.learner import Learner
        with tempfile.TemporaryDirectory() as dir_:
            path = os.path.join(dir_, "w.json")

            def read():
                if os.path.exists(path):
                    return json.load(open(path, encoding="utf-8"))
                return {"entities": {}, "observations": {},
                        "unknown_words": {}}

            def write(w):
                json.dump(w, open(path, "w", encoding="utf-8"),
                          ensure_ascii=False)
            lr = Learner(load_world=read, save_world=write)
            lr.learn_from_text("Бегемот, или гиппопотам, живёт в реке.",
                               source="text", allow_teacher=False)
            r = read()["observations"]["гиппопотам|SYNONYM_OF|бегемот"]
            self.assertEqual(GROUND_WEIGHT_LAW, r.get("weight_law"),
                             "запись варианта имени создана вне закона веса: "
                             "у неё нет ни закона, ни числа, и опыт её минует")
            self.assertEqual(
                weight_from_ground(ground_of("text"), 0, False), r["weight"],
                "безымянному варианту имени выдана сила названного документа")

    def test_6_lone_and_repeated_retelling_stay_under_the_floor(self):
        for n in range(0, 15):
            w = weight_from_ground(GROUND_TESTIMONY, n)
            self.assertLess(w, CONFLICT_WEIGHT_MIN,
                            f"чужой пересказ в {n} строк перешагнул порог "
                            f"диагноза ({w}): «онтология без опыта» начинает "
                            f"срабатывать на громкости")
            self.assertLess(record_weight(pereskaz(n)), CONFLICT_WEIGHT_MIN,
                            f"запись с {n} основаниями пересказа перешагнула "
                            f"порог диагноза через писателя")

    def test_7_diagnosis_needs_a_real_ground_not_volume(self):
        """Диагноз достижим, но только честным основанием (словом Хозяина).

        Заперты ОБЕ стороны: безымянная связь не обязана его вызывать (иначе
        инспектор орёт на всё), названная — обязана (иначе инспектор мёртв,
        ровно тот дефект, который чинили замером).
        """
        self.assertLess(weight_from_ground(GROUND_OBSERVATION, 0),
                        CONFLICT_WEIGHT_MIN)
        self.assertGreaterEqual(weight_from_ground(GROUND_OWNER, 1),
                                CONFLICT_WEIGHT_MIN)
        with tempfile.TemporaryDirectory() as dir_:
            b = brain_in(dir_)
            b.learn_from_text("Кит нырять глубоко.", allow_teacher=False)
            key = "кит|CAN_DO|нырять"
            self.assertEqual(
                [], [c for c in b.experience_conflicts()
                     if c["kind"] == "ontology_without_experience"],
                "инспектор видит силу там, где её никто не называл")
            world = b._load_world()
            world["observations"][key]["source"] = "user"
            world["observations"][key]["lineage"] = "owner"
            b._save_world(world)
            self.assertTrue([c for c in b.experience_conflicts()
                             if c["kind"] == "ontology_without_experience"],
                            "инспектор молчит даже когда говорит Хозяин")


class TestExperienceNumberBelongsToExperience(unittest.TestCase):
    """Число, посчитанное сном из силы опыта, — не собственность основания."""

    def test_8_witness_may_not_recompute_an_experience_weight(self):
        """НОВЫЙ сквозной тест: свидетель приходит к записи, чей вес — сон.

        Мутация «add_witness пересчитывает вес ВСЕГДА» прошла первый прогон
        батареи: закон был написан, а сквозного пути, где он работает, не
        было. Замер: сон понижает связь до 95 (опыт молчит), на следующий
        день приходит новое свидетельство и возвращает 180 по основанию —
        понижение исчезает МОЛЧА, а опыт не вернулся.
        """
        r = rec(source="user", lineage="owner",
                weight=weight_from_ground(GROUND_OWNER, 1),
                weight_law=EXPERIENCE_WEIGHT_LAW)
        self.assertLess(95, record_weight(r),
                        "проверка бессмысленна: основание тут не сильнее")
        r["weight"] = 95
        add_witness(r, source="wiki", evidence="другая фраза",
                    confidence=0.9, relation="CAN_DO", document="Дождь")
        self.assertEqual(95, r["weight"],
                         "новое свидетельство отменило понижение, положенное "
                         "опытом: вес перестал быть числом опыта")
        self.assertEqual(EXPERIENCE_WEIGHT_LAW, r["weight_law"],
                         "закон-источник переименован молча: читатель больше "
                         "не отличает опыт от основания")
        self.assertGreater(independent_bases(r), 0,
                           "основание не записано: засчёт потерян в данных")

    def test_9_reader_may_not_cover_an_experience_weight(self):
        """Читатель не имеет права «улучшить» число сна максимумом.

        Замер max(): сон понижал связь со словом Хозяина 180 -> 150,
        читатель возвращал 180, понижение НЕ ДОХОДИЛО до графа никогда. Та же
        ошибка, что лечилась allow_lower в replay, только с другой стороны.
        """
        r = rec(source="user", lineage="owner", confidence=0.4,
                weight=95, weight_law=EXPERIENCE_WEIGHT_LAW)
        self.assertGreater(weight_from_ground(GROUND_OWNER, 1), 95,
                           "проверка бессмысленна: основание тут не сильнее")
        self.assertEqual(95, record_weight(r),
                         "max(основание, опыт) вернул понижение назад")

    def test_10_bare_number_is_recomputed_not_trusted(self):
        """Голое число (без имени закона) — самоотчёт, права голоса нет."""
        r = rec(source="text", lineage="", confidence=1.0, weight=255)
        self.assertLess(record_weight(r), 100,
                        "числу без имени закона поверили: самоотчёт вернулся "
                        "через дверь «у записи уже лежит weight»")
        self.assertEqual(255, record_weight(dict(
            r, weight=255, weight_law=EXPERIENCE_WEIGHT_LAW)),
            "число опыта пересчитать на месте нечем — его обязаны беречь")

    def test_11_floor_is_not_a_license_to_inflate(self):
        """Пол опыта (80) не обязан превышать вес честного наблюдения."""
        self.assertLessEqual(WEIGHT_FLOOR,
                             weight_from_ground(GROUND_OBSERVATION, 1),
                             "опыт понижает связь ниже, чем стоит честное "
                             "наблюдение: пол стал верхней границей")


class TestClaimedGroundDiesAtTheWriter(unittest.TestCase):
    """Мёртвый замок, оживлённый этой правкой: stamp_ground не вызывался НИКЕМ.

    Функция существовала, была заперта тестом на record_ground, и ни один
    писатель её не звал — то есть защита «приехавшее ground не имеет силы»
    висела на честности каждого места записи. Теперь основание ставит ОДНА
    функция, и мутация «принять приехавшее» красна.
    """

    def test_12_payload_cannot_promote_itself_through_the_writer(self):
        for канал in ("wiki", "llm", "gemma", "agent_teacher", "text"):
            with self.subTest(канал=канал):
                r = rec(source=канал, ground="rule", lineage="",
                        provenance=[])
                add_witness(r, source=канал, evidence="фраза",
                            confidence=1.0, relation="CAN_DO",
                            document="Кит")
                self.assertNotEqual("rule", r.get("ground"),
                                    f"канал {канал} назначил себе аксиому")
                self.assertLess(record_weight(r), CONFLICT_WEIGHT_MIN,
                                f"канал {канал} наболтал себе вес диагноза")

    def test_12b_stamp_ground_is_the_live_door(self):
        """Замок обязан быть ЖИВЫМ: проверка прямо на stamp_ground.

        Первый вариант этого класса проверял только record_ground и писателей,
        и мутация «stamp_ground принимает приехавшее ground» ВЫЖИЛА: тесты
        охраняли функцию, которую никто не звал (исправлено — писатели теперь
        ставят основание через неё). Проверка ниже бьёт по самой двери, чтобы
        «защищено» перестало означать «написано».
        """
        r = stamp_ground({"head": "кит", "relation": "IS_A", "tail": "рыба",
                          "source": "wiki", "lineage": "doc:кит",
                          "ground": "rule", "confidence": 1.0})
        self.assertEqual(GROUND_TESTIMONY, r["ground"],
                         "приехавшее основание принято на веру")
        self.assertLess(r["weight"], CONFLICT_WEIGHT_MIN)
        # ВЕС И ОСНОВАНИЕ Ставятся ВМЕСТЕ: разойтись они не могут даже
        # руками, иначе запись «ground=owner, weight=20» осталась бы легальной.
        self.assertEqual(weight_from_ground(GROUND_TESTIMONY, 1, False),
                         r["weight"])
        self.assertEqual(GROUND_WEIGHT_LAW, r["weight_law"])
        # Число опыта — не собственность основания (тот же закон, что в
        # add_witness): перечитать его здесь значило бы стереть понижение сна.
        kept = stamp_ground({"head": "кит", "relation": "CAN_DO",
                             "tail": "нырять", "source": "user",
                             "lineage": "owner", "weight": 95,
                             "weight_law": EXPERIENCE_WEIGHT_LAW})
        self.assertEqual(95, kept["weight"],
                         "stamp_ground стёр понижение, положенное опытом")

    def test_13_brain_payload_cannot_promote_itself(self):
        """То же через мозг: пакет учителя со словом «правило» — чужие байты."""
        with tempfile.TemporaryDirectory() as dir_:
            b = brain_in(dir_)
            b._record_observation("кит", {"hierarchy": ["рыба"],
                                          "ground": "rule", "title": "Кит"},
                                  confidence=1.0, evidence="кит — рыба",
                                  source="llm")
            r = obs(b)["кит|IS_A|рыба"]
            self.assertEqual(GROUND_TESTIMONY, r.get("ground"))
            self.assertLess(record_weight(r), CONFLICT_WEIGHT_MIN)


class TestWeightAndSupportTellOneStory(unittest.TestCase):
    """Два числа об одном знании обязаны означать одно."""

    def test_14_support_equals_named_bases(self):
        """Поддержка = названные основания, а не «сколько раз засчитали».

        Замер расхождения (experiments/measure_support_vs_bases.py, случай A):
        безымянная фраза, затем её же называние документом давали support=2
        при independent_bases=1 — «видела 2 раза» на одном основании, и
        «предположительно (видела 2 раза)» показывало подтверждение, которого
        не было.
        """
        with tempfile.TemporaryDirectory() as dir_:
            b = brain_in(dir_)
            key = "дождь|CAN_DO|гасить"
            b._record_observation("дождь", {"actions": ["гасить"],
                                            "title": "Дождь"},
                                  confidence=0.8, evidence="фраза",
                                  source="wiki")
            r = obs(b)[key]
            self.assertEqual(1, independent_bases(r))
            self.assertEqual(1, int(r["support"]))
            # Пересказ ТОЙ ЖЕ статьи: число строк растёт, оснований нет.
            b._record_observation("дождь", {"actions": ["гасить"],
                                            "title": "Дождь"},
                                  confidence=0.8, evidence="та же фраза",
                                  source="llm")
            r = obs(b)[key]
            self.assertEqual(1, independent_bases(r),
                             "тот же документ назван другим каналом — и стал "
                             "вторым основанием")
            self.assertEqual(independent_bases(r), int(r["support"]),
                             "число подтверждений разошлось с числом оснований")
            # ДРУГОЙ названный документ — настоящее второе основание.
            b._record_observation("дождь", {"actions": ["гасить"],
                                            "title": "книга о воде"},
                                  confidence=0.8, evidence="та же мысль",
                                  source="wiki")
            r = obs(b)[key]
            self.assertEqual(2, independent_bases(r))
            self.assertEqual(independent_bases(r), int(r["support"]),
                             "пересказ всё ещё греет поддержку")

    def test_15_weight_is_recomputed_not_carried_along(self):
        """Приехавший `weight` не имеет силы: вес выводят из своих данных.

        Иначе пакет учителя принёс бы «255» и стал неоспоримым, минуя
        основание, — тот же класс лжи, что голое число в test_10.
        """
        with tempfile.TemporaryDirectory() as dir_:
            b = brain_in(dir_)
            b._record_observation("мяч", {"hierarchy": ["игрушка"],
                                          "weight": 255},
                                  confidence=1.0, evidence="фраза",
                                  source="llm")
            r = obs(b)["мяч|IS_A|игрушка"]
            self.assertNotEqual(255, r.get("weight"),
                                "число из пакета учителя стало весом связи")
            self.assertLess(record_weight(r), CONFLICT_WEIGHT_MIN)


class TestNumbersAreNotMagic(unittest.TestCase):
    """Ступени обязаны быть сопоставимы с порогом, иначе порог — магия."""

    def test_16_ladder_is_ordered(self):
        ряд = [weight_from_ground(GROUND_TESTIMONY, 1),
               weight_from_ground(GROUND_OBSERVATION, 1),
               weight_from_ground(GROUND_OWNER, 1)]
        self.assertEqual(ряд, sorted(ряд),
                         "пересказ сильнее наблюдения или наблюдение сильнее "
                         "слова Хозяина — порядок ступеней сломан")
        self.assertLess(ряд[0], CONFLICT_WEIGHT_MIN,
                        "одинокий пересказ обязан быть ниже порога диагноза")
        self.assertGreater(ряд[2], CONFLICT_WEIGHT_MIN,
                           "слово Хозяина обязано быть выше порога: иначе "
                           "инспектор недостижим законным путём")

    def test_17_negation_weakens_but_does_not_erase(self):
        pro = weight_from_ground(GROUND_OWNER, 1, False)
        contra = weight_from_ground(GROUND_OWNER, 1, True)
        self.assertLess(contra, pro, "отрицание не ослабляет связь")
        self.assertGreater(contra, 8,
                           "отрицание стёрто в ноль: «не» стало равным «ничего"
                           " не сказано»")

    def test_18_no_self_report_leaks_through_the_reader(self):
        """confidence не участвует в выводе НИ НА ОДНОМ пути чтения."""
        base = rec(confidence=0.5)
        for loud in (0.0, 0.3, 0.9, 1.0):
            self.assertEqual(record_weight(base),
                             record_weight(rec(confidence=loud)),
                             f"самоотчёт {loud} протёк в вес")
            self.assertEqual(observation_weight(base),
                             observation_weight(rec(confidence=loud)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
