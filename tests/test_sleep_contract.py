"""
КОНТРАКТ SLEEP CYCLE (очередь memory.txt, п.23-доработка, P1).

Сон — ЕДИНСТВЕННОЕ место, где опыт получает право править weight онтологии
(memory.txt). Значит тест обязан проверять не только «число выросло», но и
ГРАНИЦЫ права: что сон НЕ имеет права делать.

Числа взяты из experiments/PLASTICITY_CONTRACT.md (замер
plasticity_sweep.json) и повторены здесь константами — если их поменяют в
одном месте, тест покраснеет и заставит менять в двух.

Отдельный класс проверяет два дефекта, НАЙДЕННЫХ при реализации этой же
задачи (оба молчаливые, оба стоят потерянного знания):

  1. weight НЕ лежал в observations (ключей head/relation/tail/confidence/
     negated/evidence/source/support/first_seen), а experience_conflicts()
     читал obs["weight"] и вечно получал 0 — диагноз
     ontology_without_experience был НЕДОСТИЖИМ, хотя тесты на него были.
  2. replay_observations понижать вес не умел («вес никогда не понижаем»),
     поэтому понижение, сделанное сном, умирало в JSON: в файле 190, в графе
     после перезапуска 210.

Тесты НЕ трогают живую память и НЕ зовут LLM.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brain.sleep import (CONSOLIDATION_THRESHOLD, CONFLICT_STRENGTH_MAX,
                        CONFLICT_WEIGHT_DOWN, CONFLICT_WEIGHT_MIN, DECAY_RATE,
                        SOURCE, TICKS_PER_STRENGTH_POINT, WEIGHT_BASE,
                        WEIGHT_CAP, WEIGHT_FLOOR, WEIGHT_PER_STRENGTH,
                        SLEEP_TRIGGER_TICKS, consolidate, decay_transitions,
                        reconcile_saved, sleep, sleep_due, table_of_section)
from core.transitions import MAX_STRENGTH, TransitionTable
from lang.ru.text import experience as ru_experience  # IR A2.4: разбор приносят ядру

# ── helpers ───────────────────────────────────────────────────────────────


def obs_rec(head, tail, weight, *, negated=False, conf=0.9, support=1,
            relation="CAN_DO", source="text:t"):
    return {f"{head}|{relation}|{tail}": {
        "head": head, "relation": relation, "tail": tail,
        "confidence": conf, "negated": negated, "support": support,
        "weight": weight, "source": source, "evidence": ""}}


def world_of(*records):
    obs = {}
    for r in records:
        obs.update(r)
    return {"entities": {}, "observations": obs, "unknown_words": {}}


def table(pairs):
    t = TransitionTable()
    for head, tail, s in pairs:
        t.confirm(head, tail, s)
    return t


def section(pairs):
    """Секция transitions из списка (head, tail, сила)."""
    rows = {}
    for h, t, s in pairs:
        rows.setdefault(h, []).append((t, s))
    return {"version": 1, "pack": 2,
            "rows": [{"head": h,
                      "tails": [t for t, _ in sorted(v, key=lambda p: (-p[1], p[0]))],
                      "strengths": [s for _, s in sorted(v, key=lambda p: (-p[1], p[0]))]}
                     for h, v in sorted(rows.items())]}


class TestNumbersAreContract(unittest.TestCase):
    """Числа контракта зафиксированы: правка «на глаз» должна быть видна."""

    def test_contract_numbers(self):
        self.assertEqual(CONSOLIDATION_THRESHOLD, 8)
        self.assertEqual((WEIGHT_BASE, WEIGHT_PER_STRENGTH, WEIGHT_CAP),
                         (80, 8, 200))
        self.assertEqual((CONFLICT_WEIGHT_MIN, CONFLICT_STRENGTH_MAX,
                          CONFLICT_WEIGHT_DOWN, WEIGHT_FLOOR),
                         (150, 3, 30, 80))
        self.assertAlmostEqual(DECAY_RATE, 0.05)


class TestWhenToSleep(unittest.TestCase):
    """
    КОГДА СПАТЬ. До этой правки порог не был закреплён НИЧЕМ — и потому
    прожил неверным: он был выведен из замера «8 тиков за прогон», а реальный
    замер даёт 10 тиков за прогон и +1 к силе за прогон. Порог 64 при таких
    числах — это сон при силе 7, то есть ГАРАНТИРОВАННО пустой.

    Замер порога 64 (20 прогонов рассказа из 6 предложений):
        снов 2, осмысленных 1, создал 4 — ОДИН сон прошёл впустую,
        и впустую он не был: «затухло 8, стёрто 1» при created=0.
    Пустой сон СПИСЫВАЕТ опыт, из которого позже должно родиться знание.
    Порог 80: снов 2, осмысленных 2, создал 4, поднято 4, пустых 0.
    """

    def test_trigger_not_earlier_than_knowledge_can_exist(self):
        """
        Главный смысл теста: сон не может наступить РАНЬШЕ, чем хоть одна
        пара физически способна набрать силу порога. Иначе первый сон пустой
        по построению, а не по стечению обстоятельств.
        """
        self.assertGreaterEqual(
            SLEEP_TRIGGER_TICKS,
            CONSOLIDATION_THRESHOLD * TICKS_PER_STRENGTH_POINT,
            f"накоплено {SLEEP_TRIGGER_TICKS} тиков, а силу "
            f"{CONSOLIDATION_THRESHOLD} пара набирает только к "
            f"{CONSOLIDATION_THRESHOLD * TICKS_PER_STRENGTH_POINT} "
            f"(замер: +1 сила на {TICKS_PER_STRENGTH_POINT} тиков) — "
            "первый сон обязан упереться в created=0")

    def test_boundary_is_exact(self):
        self.assertFalse(sleep_due(SLEEP_TRIGGER_TICKS - 1))
        self.assertTrue(sleep_due(SLEEP_TRIGGER_TICKS))

    def test_counter_of_measured_story_matches_reality(self):
        """
        TICKS_PER_STRENGTH_POINT — измеренная величина, а не предположение.
        Если цикл изменит число тиков на предложение, этот тест обязан
        покраснеть: он сверяет константу с фактическим поведением цикла.
        """
        from core.cortex import PredictiveCortex
        from core.graph import GraphMemory
        story = ("Кошка крадётся к мыши. Кошка ловит мышь. Кошка ест мышь. "
                 "Пёс лает на кошку. Пёс гоняет кошку. Кошка видит мышь.")
        cortex = PredictiveCortex(GraphMemory(), teacher=None, segmenter=ru_experience)
        first = cortex.process_stream(story, verbose=False)["ticks"]
        second = cortex.process_stream(story, verbose=False)["ticks"]
        self.assertEqual(first, second,
                         "одно и то же предложение не даёт столько же тиков — "
                         "счёт тиков нестабилен, порог сна meaningless")
        # прирост силы самой частой пары за прогон = 1, значит тиков на очко
        # силы ровно столько, сколько тиков даёт прогон
        self.assertEqual(second, TICKS_PER_STRENGTH_POINT,
                         f"прогон рассказа даёт {second} тиков, а константа "
                         f"говорит {TICKS_PER_STRENGTH_POINT}: порог сна "
                         "рассчитан по устаревшему замеру")

    def test_env_override_is_read_at_import(self):
        """
        Порог обязан перекрываться окружением: без этого измерять его можно
        только правкой кода, а значит измерять перестают.
        """
        env = dict(os.environ, LIPID2_SLEEP_TICKS="13")
        p = subprocess.run([sys.executable, "-c",
                            "import sys; sys.path.insert(0, %r);"
                            "import brain.sleep as s;"
                            "print(s.SLEEP_TRIGGER_TICKS, s.sleep_due(12), "
                            "s.sleep_due(13))" % ROOT],
                           env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr[-600:])
        self.assertEqual(p.stdout.split()[-3:], ["13", "False", "True"],
                         "LIPID2_SLEEP_TICKS не подействовал: "
                         + p.stdout.strip())


class TestConsolidation(unittest.TestCase):
    """Правило 1: опыт силён, знания нет -> CAN_DO создаётся."""

    def test_creates_can_do_with_formula_weight(self):
        w = world_of()
        r = sleep(w, None, table([("птенец", "летать", 11)]))
        self.assertEqual(r.created, 1)
        rec = w["observations"]["птенец|CAN_DO|летать"]
        self.assertEqual(rec["weight"], WEIGHT_BASE + WEIGHT_PER_STRENGTH * 11)
        self.assertEqual(rec["source"], SOURCE)
        self.assertFalse(rec["negated"])
        self.assertIn("опыт", rec["evidence"],
                      "у знания обязан быть источник: откуда связь взялась")

    def test_below_threshold_creates_nothing(self):
        for s in range(1, CONSOLIDATION_THRESHOLD):
            w = world_of()
            r = sleep(w, None, table([("а", "б", s)]))
            self.assertEqual(r.created, 0, f"на силе {s} создавать нельзя")
            self.assertEqual(w["observations"], {})

    def test_created_weight_never_exceeds_cap(self):
        """Потолок — про то, что пишет СОН, а не про веса из текста."""
        w = world_of()
        r = sleep(w, None, table([("кит", "нырять", MAX_STRENGTH + 40)]))
        self.assertEqual(r.created, 1)
        self.assertEqual(w["observations"]["кит|CAN_DO|нырять"]["weight"],
                         WEIGHT_CAP,
                         "выше потолка — это «неоспоримо», а опыт неоспоримым "
                         "быть не может")

    def test_sleep_does_not_push_high_weight_over_cap(self):
        """Сон не поднимает то, что и так выше потолка (201+ = «неоспоримо»)."""
        w = world_of(obs_rec("кит", "нырять", 230))
        r = sleep(w, None, table([("кит", "нырять", 15)]))
        rec = w["observations"]["кит|CAN_DO|нырять"]["weight"]
        self.assertLessEqual(rec, 230, "чужое «неоспоримо» сон не усиливает")
        self.assertLessEqual(rec, max(WEIGHT_CAP, 230))

    def test_only_can_do_is_created(self):
        """Сон не создаёт IS_A/HAS_PROPERTY: опытом говорят только действия."""
        w = world_of()
        sleep(w, None, table([("кошка", "кошачьи", 15)]))
        for k in w["observations"]:
            self.assertIn("|CAN_DO|", k)

    def test_dry_run_changes_nothing(self):
        w = world_of(obs_rec("кошка", "лазать", 100, conf=0.5))
        t = table([("кошка", "лазать", 15), ("птенец", "летать", 12)])
        before = json.dumps(w, sort_keys=True)
        r = sleep(w, None, t, dry_run=True)
        self.assertEqual(json.dumps(w, sort_keys=True), before,
                         "пробный сон не имеет права править мир")
        self.assertEqual(t.get("кошка", "лазать"), 15,
                         "пробный сон не имеет права точить опыт")
        self.assertTrue(r.dry_run)


class TestRaise(unittest.TestCase):
    """Правило 2: подтверждение опыта поднимает вес, но не выше потолка."""

    def test_raise_step(self):
        w = world_of(obs_rec("кошка", "лазать", 100))
        r = sleep(w, None, table([("кошка", "лазать", 12)]))
        self.assertEqual(r.raised, 1)
        self.assertEqual(w["observations"]["кошка|CAN_DO|лазать"]["weight"],
                         100 + (12 - 7) * 10)

    def test_reason_is_recorded(self):
        w = world_of(obs_rec("кошка", "лазать", 100))
        sleep(w, None, table([("кошка", "лазать", 12)]))
        rec = w["observations"]["кошка|CAN_DO|лазать"]
        self.assertIn("weight_reason", rec,
                      "вес обязан быть подписан: по какому опыту изменился")

    def test_weak_experience_does_not_raise(self):
        w = world_of(obs_rec("кошка", "лазать", 100))
        r = sleep(w, None, table([("кошка", "лазать", 4)]))
        self.assertEqual(r.raised, 0)
        self.assertEqual(w["observations"]["кошка|CAN_DO|лазать"]["weight"], 100)


class TestReconcile(unittest.TestCase):
    """Правило 3: онтология утверждает, опыт молчит -> постепенное понижение."""

    def test_lowers_weight_by_step(self):
        w = world_of(obs_rec("кит", "нырять", 220))
        r = sleep(w, None, table([("кит", "нырять", 1)]))
        self.assertEqual(r.lowered, 1)
        self.assertEqual(w["observations"]["кит|CAN_DO|нырять"]["weight"],
                         220 - CONFLICT_WEIGHT_DOWN)

    def test_floor_is_not_broken(self):
        """Много снов подряд не превращают «может» в «не может»."""
        w = world_of(obs_rec("кит", "нырять", 220))
        for _ in range(30):
            sleep(w, None, table([("кит", "нырять", 0)]),
                  decay_rate=0.0)      # опыт не восстанавливаем: худший случай
            w["transitions"] = section([("кит", "нырять", 1)])
        rec = w["observations"]["кит|CAN_DO|нырять"]
        self.assertGreaterEqual(rec["weight"], WEIGHT_FLOOR,
                                "опыт не имеет права запрещать: отсутствие "
                                "наблюдений — не опровержение")

    def test_low_weight_no_conflict(self):
        """Слабая связь (ниже 150) — не конфликт: понижать нечего."""
        w = world_of(obs_rec("кит", "нырять", 140))
        r = sleep(w, None, table([("кит", "нырять", 1)]))
        self.assertEqual(r.lowered, 0)
        self.assertEqual(w["observations"]["кит|CAN_DO|нырять"]["weight"], 140)


class TestNegationIsUntouchable(unittest.TestCase):
    """Опыт НЕ имеет права превращать «не делает» в «делает»."""

    def test_negated_observation_is_not_flipped(self):
        w = world_of(obs_rec("рыба", "бегать", 200, negated=True))
        r = sleep(w, None, table([("рыба", "бегать", 15)]))
        rec = w["observations"]["рыба|CAN_DO|бегать"]
        self.assertTrue(rec["negated"], "отрицание снято быть не может")
        self.assertEqual(rec["weight"], 200, "вес отрицания — не частота")
        self.assertEqual(r.skipped_negated, 1,
                         "противоречие обязано быть посчитано, а не проглочено")

    def test_conflict_reported_in_text(self):
        w = world_of(obs_rec("рыба", "бегать", 200, negated=True))
        r = sleep(w, None, table([("рыба", "бегать", 15)]))
        self.assertTrue(r.notes, "противоречие обязано быть сказано словами")
        self.assertIn("ПРОТИВОРЕЧИТ", " ".join(r.notes))


class TestNegatedWeightIsHonored(unittest.TestCase):
    """Отрицанная связь не участвует в конфликте как «утверждает»."""

    def test_negated_high_weight_not_lowered(self):
        """
        Отрицание с высоким weight — это УВЕРЕННОЕ «не». Конфликт же про
        обратное: «онтология УТВЕРЖДАЕТ, опыт молчит». Если бы сон понижал
        отрицанные записи, он молча ослаблял бы запрет.
        """
        w = world_of(obs_rec("рыба", "бегать", 220, negated=True))
        r = sleep(w, None, table([("рыба", "бегать", 2)]))
        self.assertEqual(r.lowered, 0)
        self.assertEqual(w["observations"]["рыба|CAN_DO|бегать"]["weight"], 220)


class TestDecay(unittest.TestCase):
    """Затухание: шум обязан иметь шанс исчезнуть, сильные — остаться."""

    def test_strong_tails_stay_distinguishable(self):
        t = table([("а", "много", 15), ("а", "средне", 8), ("а", "мало", 3)])
        decay_transitions(t, DECAY_RATE)
        self.assertEqual(t.get("а", "много"), 14)
        self.assertEqual(t.get("а", "средне"), 7)
        self.assertGreater(t.get("а", "много"), t.get("а", "средне"))

    def test_rare_noise_can_die(self):
        t = table([("а", "шум", 1)])
        r = decay_transitions(t, DECAY_RATE)
        self.assertEqual(t.get("а", "шум"), 0)
        self.assertIn("шум", [c["tail"] for c in r.changes])
        self.assertEqual(t.known_tails("а"), [],
                         "compact() обязан убрать умершее имя из рядка")

    def test_zero_rate_is_noop(self):
        t = table([("а", "б", 9)])
        r = decay_transitions(t, 0.0)
        self.assertEqual((t.get("а", "б"), r.decayed), (9, 0))

    def test_bad_rate_raises(self):
        for bad in (-0.1, 1.0, 2.0):
            with self.assertRaises(ValueError):
                decay_transitions(table([("а", "б", 5)]), bad)

    def test_threshold_survives_one_sleep(self):
        """
        Порог 8 обязан оставаться порогом ПОСЛЕ сна: связь, дожившая до 8,
        станет знанием в ЭТОМ сне, а не «где-то потом», иначе порядок вызовов
        внутри сна менял бы результат.
        """
        w = world_of()
        t = table([("а", "б", 8)])
        sleep(w, None, t)
        self.assertIn("а|CAN_DO|б", w["observations"])
        self.assertEqual(w["observations"]["а|CAN_DO|б"]["weight"],
                         WEIGHT_BASE + WEIGHT_PER_STRENGTH * 8)


class TestSleepOrder(unittest.TestCase):
    """Порядок «сначала консолидация, потом decay» зафиксирован тестом."""

    def test_created_knowledge_keeps_exact_formula(self):
        """Если decay пойдёт первым, вес созданной связи изменится."""
        w = world_of()
        sleep(w, None, table([("а", "б", 10)]))
        self.assertEqual(w["observations"]["а|CAN_DO|б"]["weight"],
                         WEIGHT_BASE + WEIGHT_PER_STRENGTH * 10)

    def test_decay_happens_after(self):
        t = table([("а", "б", 10)])
        w = world_of()
        r = sleep(w, None, t)
        self.assertEqual(t.get("а", "б"), 9,
                         "сила затухает на глазах: 10 -> 9")
        self.assertEqual(r.decayed, 1)


class TestReconcileSaved(unittest.TestCase):
    """
    Запись сна: СВОЙ decay применим, ЧУЖОЙ опыт не откатан.

    Это ответ на собственный дефект реализации: обычный merge_json берёт
    максимум, поэтому decay отменяется собственным сохранением (в памяти 5,
    в файле 10). Наивная «замена» лечила decay, но молча ТЕРЯЛА чужой опыт.
    """

    def test_own_decay_survives_save(self):
        base = section([("кошка", "лазать", 10)])
        ours = section([("кошка", "лазать", 5)])       # после decay
        file_ = base                                    # никто не трогал
        sec, lost = reconcile_saved(base, ours, file_)
        self.assertEqual(lost, 0)
        self.assertEqual(table_of_section(sec).get("кошка", "лазать"), 5,
                         "decay не имеет права отменяться сохранением")

    def test_neighbor_experience_is_not_rewound(self):
        base = section([("кошка", "лазать", 10)])
        ours = section([("кошка", "лазать", 5)])
        file_ = section([("кошка", "лазать", 12)])      # сосед наработал
        sec, lost = reconcile_saved(base, ours, file_)
        self.assertEqual(lost, 1, "потеря обязана быть посчитана")
        self.assertEqual(table_of_section(sec).get("кошка", "лазать"), 12,
                         "чужой опыт откатывать decay-ом нельзя")

    def test_foreign_pair_is_kept(self):
        base = section([("кошка", "лазать", 10)])
        ours = section([("кошка", "лазать", 5)])
        file_ = section([("кошка", "лазать", 10), ("пёс", "плавать", 7)])
        sec, lost = reconcile_saved(base, ours, file_)
        t = table_of_section(sec)
        self.assertEqual(t.get("пёс", "плавать"), 7,
                         "увиден впервые — не значит «удалить»")
        self.assertEqual(t.get("кошка", "лазать"), 5)
        self.assertEqual(lost, 0)

    def test_dead_pairs_not_written_back(self):
        """Погасший переход не обязан лежать в файле нулём."""
        base = section([("а", "б", 1)])
        ours = {"version": 1, "pack": 2, "rows": []}
        sec, lost = reconcile_saved(base, ours, base)
        self.assertEqual(sec["rows"], [])

    def test_section_shape_is_readable(self):
        sec, _ = reconcile_saved({}, section([("б", "x", 3), ("а", "y", 4),
                                              ("а", "z", 9)]), {})
        self.assertEqual([r["head"] for r in sec["rows"]], ["а", "б"],
                         "head по алфавиту: байты файла не зависят от порядка")
        row_a = sec["rows"][0]
        self.assertEqual(row_a["strengths"], sorted(row_a["strengths"],
                                                   reverse=True))


class TestTableOfSection(unittest.TestCase):
    """
    Регистрация собственного дефекта этой же задачи.

    table_from_world ждёт СЛОВАРЬ МИРА и читает из него ключ "transitions".
    Если передать ему СЕКЦИЮ, он молча вернёт ПУСТУЮ таблицу («ещё не учились»
    — законное состояние, проверить нечем). Сон с такой таблицей ничего не
    видит и перезаписывает живой опыт пустотой: замерена потеря перехода
    силой 12.
    """

    def test_world_and_section_are_different_things(self):
        from core.transitions import table_from_world
        sec = section([("кошка", "лазать", 12)])
        self.assertEqual(table_from_world(sec).size(), 0,
                         "вот она, тихая дыра: секция, прочитанная как мир")
        self.assertEqual(table_of_section(sec).get("кошка", "лазать"), 12)
        self.assertEqual(table_from_world({"transitions": sec}).get(
            "кошка", "лазать"), 12)


class TestWeightLivesInData(unittest.TestCase):
    """Дефект №1: weight должен БЫТЬ в наблюдениях, иначе инспектор мёртв."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sleep_contract_")
        self.path = os.path.join(self.dir, "lipid_world.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _read(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, w):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(w, fh, ensure_ascii=False)

    def test_learner_writes_weight_into_observation(self):
        from brain.learner import Learner
        self._write({"entities": {}, "observations": {}, "unknown_words": {}})
        lr = Learner(load_world=self._read, save_world=self._write)
        lr.learn_from_text("Кошка боится воды.", source="t")
        obs = self._read()["observations"]
        can = [v for v in obs.values() if v["relation"] == "CAN_DO"]
        self.assertTrue(can)
        for v in can:
            self.assertIn("weight", v,
                          "без weight в данных читатель веса обречён читать 0")
            self.assertGreater(v["weight"], 0)

    def test_weight_recomputed_on_second_sighting(self):
        from brain.learner import Learner
        self._write({"entities": {}, "observations": {}, "unknown_words": {}})
        lr = Learner(load_world=self._read, save_world=self._write)
        lr.learn_from_text("Кошка боится воды.", source="t")
        first = self._read()["observations"]["кошка|CAN_DO|бояться"]["weight"]
        lr.learn_from_text("Кошка боится воды.", source="t")
        second = self._read()["observations"]["кошка|CAN_DO|бояться"]["weight"]
        self.assertGreater(second, first,
                           "второе свидетельство обязано быть видно числом")

    def test_conflict_inspector_is_alive(self):
        """
        ТО, ЧЕГО НЕ БЫЛО: experience_conflicts() читал obs["weight"], ключа
        weight не существовало — значит диагноз ontology_without_experience
        было НЕВОЗМОЖНО получить ни при каких данных.
        """
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=self.dir)
        brain.learn_from_text("Кит ныряет глубоко.", allow_teacher=False)
        conflicts = brain.experience_conflicts()
        self.assertTrue(any(c["kind"] == "ontology_without_experience"
                            for c in conflicts),
                        f"инспектор молчит: {conflicts}")
        self.assertGreaterEqual(
            [c for c in conflicts
             if c["kind"] == "ontology_without_experience"][0]["weight"],
            CONFLICT_WEIGHT_MIN)

    def test_inspector_reads_old_world_without_weight(self):
        """Старый файл без weight — НЕ «уверенность 0», а вывод по формуле."""
        from core.records import observation_weight
        old = {"head": "кит", "relation": "CAN_DO", "tail": "нырять",
               "confidence": 0.9, "support": 2, "negated": False}
        self.assertGreater(observation_weight(old), 100,
                           "старая запись не обязана выглядеть пустой")
        self.assertEqual(observation_weight({"weight": 77}), 77,
                         "записанное число авторитетнее вывода")
        self.assertEqual(observation_weight(None), 0)


class TestGraphProjectionIsExact(unittest.TestCase):
    """
    Дефект №2: «вес никогда не понижаем» делал понижение опыта невидимым.

    Замер: сон записал 190 в JSON, граф после перезапуска — 210. Наблюдения
    объявлены источником, из которого граф пересобирается; значит проекция
    обязана быть ТОЧНОЙ, а не одностронней.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sleep_graph_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_lowered_weight_reaches_graph_after_restart(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.learn_from_text("Кит ныряет глубоко.", allow_teacher=False)
        key = "кит|CAN_DO|нырять"
        w = b._load_world()
        w["observations"][key]["weight"] = 220
        w["observations"][key]["support"] = 2
        b._save_world(w)
        b.learner.replay_observations(w)

        b.transitions.confirm("кит", "нырять", 1)   # опыт почти не видел
        rep = b.sleep_now()
        self.assertEqual(rep["report"].lowered, 1, rep["text"])

        saved = json.load(open(b.world_filepath, encoding="utf-8"))
        self.assertEqual(saved["observations"][key]["weight"], 190)
        # ЖИВОЙ граф, а не только файл: это проверяет ветку replay внутри
        # sleep_now (brain/lipid2_brain.py: «наблюдения поменялись — граф
        # пересобирается»). МУТАЦИОННАЯ ПРОВЕРКА: при allow_lower=False файл
        # оставался 190, а живой граф — 220, и старый вариант теста этого НЕ
        # видел, потому что смотрел только на перезапуск (там граф строится с
        # нуля и понижать нечего). Замер: /tmp/amanda/mut_probe.py
        self.assertEqual(
            b.core.nodes["кит"].out_synapses_typed["нырять"]["weight"],
            190, "сон понизил вес в файле, а живой процесс продолжает работать"
                 " со старой уверенностью: ответ строится по графу, а не по JSON")
        restart = Lipid2Brain(memory_dir=self.dir)
        self.assertEqual(
            restart.core.nodes["кит"].out_synapses_typed["нырять"]["weight"],
            190, "понижение опыта не имеет права умирать в JSON")

    def test_created_knowledge_reaches_graph(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.transitions.confirm("птенец", "летать", 12)
        rep = b.sleep_now()
        self.assertEqual(rep["report"].created, 1, rep["text"])
        node = b.core.nodes.get("птенец")
        self.assertIsNotNone(node, "новое знание обязано стать СВЯЗЬЮ")
        self.assertIn("летать", node.out_synapses_typed)

    def test_replay_without_weight_still_never_lowers(self):
        """
        Снимает ли новая правка старый запрет? НЕТ: записи БЕЗ weight (живой
        мир старше поля) по-прежнему не имеют права обнулять выученные связи.
        """
        from brain.learner import Learner
        from core.graph import GraphMemory
        path = os.path.join(self.dir, "w.json")
        world = {"entities": {}, "unknown_words": {},
                 "observations": {"кит|CAN_DO|нырять": {
                     "head": "кит", "relation": "CAN_DO", "tail": "нырять",
                     "confidence": 0.9, "support": 2, "negated": False}}}

        def read():
            return json.loads(json.dumps(world))

        core = GraphMemory()
        lr = Learner(load_world=read, save_world=lambda w: None, core=core)
        lr.replay_observations(read())
        first = core.nodes["кит"].out_synapses_typed["нырять"]["weight"]
        lr.replay_observations(read())
        second = core.nodes["кит"].out_synapses_typed["нырять"]["weight"]
        self.assertEqual(first, second)


class TestSleepNowIntegration(unittest.TestCase):
    """sleep_now: права, транзакция, отчёт — и что dry_run ничего не пишет."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="sleep_now_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _brain(self):
        from brain.lipid2_brain import Lipid2Brain
        return Lipid2Brain(memory_dir=self.dir)

    def test_dry_run_does_not_touch_file(self):
        b = self._brain()
        b.learn_from_text("Птенец учится летать.", allow_teacher=False)
        b.transitions.confirm("птенец", "летать", 12)
        before = json.load(open(b.world_filepath, encoding="utf-8"))
        r = b.sleep_now(dry_run=True)
        after = json.load(open(b.world_filepath, encoding="utf-8"))
        self.assertFalse(r["saved"])
        self.assertEqual(before, after, "пробный сон не пишет на диск")
        self.assertIn("пробный", r["text"])

    def test_live_experience_is_not_lost_by_sleep(self):
        """
        Цикл копит силу В ПАМЯТИ ПРОЦЕССА и сохраняет её только на записи.
        Сон, прочитавший только файл, этого опыта не видит. Проверка, что
        объединение «живая таблица + файл» работает: сила не схлопнулась.
        """
        b = self._brain()
        b.learn_from_text("Птенец учится летать.", allow_teacher=False)
        b.transitions.confirm("птенец", "летать", 12)
        r = b.sleep_now()
        self.assertFalse(r.get("error"), r["text"])
        saved = json.load(open(b.world_filepath, encoding="utf-8"))
        strengths = {t: s for row in saved["transitions"]["rows"]
                     for t, s in zip(row["tails"], row["strengths"])}
        self.assertIn("летать", strengths,
                      "живой опыт не имеет права исчезнуть после сна")
        self.assertEqual(strengths["летать"], 11, "12 минус decay 5%")

    def test_report_is_human_readable(self):
        b = self._brain()
        b.learn_from_text("Кит ныряет глубоко.", allow_teacher=False)
        b.transitions.confirm("кит", "нырять", 1)
        r = b.sleep_now()
        self.assertIn("СОН", r["text"])
        for word in ("создано", "понижено", "затухло"):
            self.assertIn(word, r["text"])

    def test_sleep_is_idempotent_in_steady_state(self):
        """
        Два сна подряд в устойчивом состоянии не должны «улучшать» всё дальше:
        понизив вес до 190, второй сон понижает его ещё на 30 (это постепенное
        согласование, оно и задумано шагом), но вес НЕ уходит под пол и НЕ
        перестаёт быть числом. Проверяем монотонность и пол, а не «ноль».
        """
        b = self._brain()
        b.learn_from_text("Кит ныряет глубоко.", allow_teacher=False)
        b.transitions.confirm("кит", "нырять", 1)
        weights = []
        for _ in range(6):
            r = b.sleep_now()
            self.assertFalse(r.get("error"), r["text"])
            world = json.load(open(b.world_filepath, encoding="utf-8"))
            rec = world["observations"]["кит|CAN_DO|нырять"]
            weights.append(rec["weight"])
        self.assertEqual(weights, sorted(weights, reverse=True),
                         f"вес обязан падать монотонно: {weights}")
        self.assertGreaterEqual(weights[-1], WEIGHT_FLOOR)


if __name__ == "__main__":
    unittest.main(verbosity=2)
