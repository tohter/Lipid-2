"""
КОНТРАКТ (п.26): ВЕС ОПЫТА НЕ ВЫБИРАЕТ ОПРЕДЕЛЕНИЕ, а выбор сейчас — это
порядок записи в файле.

Откуда файл. Очередь спрашивала: влияет ли опыт (weight, сила перехода, сон) на
то, КАКОЕ определение выдаст мозг? Замер (experiments/p26_measure.py,
experiments/p26_risk.py, док experiments/EXPERIENCE_CHOICE_TASK.md) ответил:
НЕ ВЛИЯЕТ НИ ОДНИМ ЧИСЛОМ. Единственное чтение веса ребра во всём проекте —
core/graph.py::print_node_associations, то есть ПЕЧАТЬ. Ответ собирается в
_definition_answer по жёсткому приоритету definition > hierarchy[0] > properties.

Замеренные числа, которые здесь замурованы:

  родитель 'обувь' вес 80 (пол), 'кустарный предмет' вес 250 (потолок сна)
      ответ при нулевом опыте:  Валенок — это обувь (категории: обувь -> ...)
      ответ при весе 250:       Валенок — это обувь (категории: обувь -> ...)
      ОТВЕТ НЕ ИЗМЕНИЛСЯ
  вес 255 на категории против definition — проигрывает определение всегда.

НАЙДЕННЫЙ ДЕФЕКТ (он же — тест TestOrderOfRecordingChoosesTheAnswer; он ЗЕЛЁНЫЙ
СЕЙЧАС, потому что фиксирует текущее ДЕФЕКТНОЕ поведение, и ОБЯЗАН
покраснеть в момент починки, чтобы та не прошла молча):

  один и тот же набор фактов, записанный в разном порядке, даёт РАЗНЫЙ ответ:
      ['вещь', 'обувь']  ->  «Валенок — это вещь»
      ['обувь', 'вещь']  ->  «Валенок — это обувь»
  Порядок записи — побочный эффект того, кто ответил раньше (Вики или
  пользователь), и к миру отношения не имеет.

Ранжирование по весу опыта ЗДЕСЬ НЕ РАЗРЕШАЮТ, а запрещают числами: грубая
категория подтверждается опытом чаще именно потому, что она грубая (замер 9:
у 'вещь' подтверждений всегда больше, чем у 'обувь'). Правка обязана идти от
нового ЗАМЕРА по содержанию, а не от частоты.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONCEPT = "валенок"
NEAR, FAR = "обувь", "кустарный предмет"
COARSE, PRECISE = "вещь", "обувь"


def write_world(directory, hierarchy, weights=None, concept=CONCEPT):
    """Мир с одной сущностью и (опционально) наблюдениями заданного веса.

    concept меняет ТОЛЬКО имя сущности: слова разного рода нужны тесту
    согласования рода, а ответ обязан говорить про то слово, о котором
    спрашивают, а не про зашитое в файл «валенок».
    """
    obs = {}
    for cat, w in (weights or {}).items():
        obs[f"{concept}|IS_A|{cat}"] = {
            "head": concept, "relation": "IS_A", "tail": cat,
            "confidence": 0.99, "negated": False, "support": 40,
            "weight": w, "evidence": "замер п.26", "source": "experience",
            "first_seen": 1,
        }
    world = {
        "entities": {concept: {
            "concept": concept, "kind": "thing", "definition": "",
            "hierarchy": list(hierarchy), "properties": [], "actions": [],
            "provenance": [{"source": "user", "note": "замер п.26"}]}},
        "relations": [], "observations": obs, "unknown_words": {},
        "transitions": {},
    }
    with open(os.path.join(directory, "lipid_world.json"), "w",
              encoding="utf-8") as fh:
        json.dump(world, fh, ensure_ascii=False, indent=1)


class P26Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="l2p26_")
        os.environ["LIPID2_MEMORY_DIR"] = self.dir
        os.environ["LIPID2_STATUS"] = "0"

    def tearDown(self):
        os.environ.pop("LIPID2_MEMORY_DIR", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def answer(self, text=None):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        # Учитель выключен ВЕСЬ: ответ обязан зависеть от памяти, а не от сети.
        b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
        b.learn_experience = False
        return b.process_input(text or f"что такое {CONCEPT}")


class TestExperienceWeightDoesNotChoose(P26Base):
    """Ни пол, ни потолок сна не меняют ответ на «что такое X»."""

    def test_maximum_experience_weight_changes_nothing(self):
        write_world(self.dir, [NEAR, FAR])
        without = self.answer()
        write_world(self.dir, [NEAR, FAR], {NEAR: 80, FAR: 250})
        with_heavy = self.answer()
        self.assertEqual(
            without, with_heavy,
            "вес опыта 250 против 80 изменил ответ — значит выбор по весу "
            "ПОЯВИЛСЯ. Он не разрешён (см. док, замер 9: частота подтверждений "
            "коррелирует с грубостью). Правка обязана идти от замера по "
            "СОДЕРЖАНИЮ и обязана обновить этот тест осознанно.")

    def test_definition_beats_max_weight_category(self):
        """definition веса не имеет вовсе и выигрывает у веса 255 всегда."""
        world = {
            "entities": {CONCEPT: {
                "concept": CONCEPT, "kind": "thing",
                "definition": "старая русская обувь из войлока",
                "hierarchy": [NEAR], "properties": [], "actions": [],
                "provenance": [{"source": "user", "note": "замер п.26"}]}},
            "relations": [],
            "observations": {f"{CONCEPT}|IS_A|{NEAR}": {
                "head": CONCEPT, "relation": "IS_A", "tail": NEAR,
                "confidence": 0.99, "negated": False, "support": 99,
                "weight": 255, "evidence": "замер", "source": "experience",
                "first_seen": 1}},
            "unknown_words": {}, "transitions": {},
        }
        with open(os.path.join(self.dir, "lipid_world.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(world, fh, ensure_ascii=False, indent=1)
        text = self.answer()
        self.assertIn("старая русская обувь", text)
        self.assertNotIn(f"это {NEAR} ", text,
                         f"категория с весом 255 вытеснила определение: {text}")

    def test_every_category_is_still_visible(self):
        """
        Ни один вариант НЕ теряется — это хорошая часть замера, она обязана
        пережить любую правку порядка: менять порядок можно, терять мнение —
        нельзя.
        """
        write_world(self.dir, [NEAR, FAR], {NEAR: 80, FAR: 250})
        text = self.answer()
        self.assertIn(NEAR, text)
        self.assertIn(FAR, text,
                      f"родитель с весом 250 пропал из ответа: {text}")


class TestOrderOfRecordingChoosesTheAnswer(P26Base):
    """
    ЭТОТ ЗАМОК БЫЛ НАПИСАН «ЗЕЛЁНЫМ, НО ОБЯЗАН ПОКРАСНЕТЬ ПРИ ПОЧИНКЕ».
    Он покраснел 23.10, когда порядок категорий перестал быть порядком
    записи, и переписан — как и требовал его собственный текст.

    НОВОЕ ПРАВИЛО (и почему оно именно такое — три замера):
      1) p26b_wiki_noise.py + wiki_category_snapshot.json (22 слова):
         порядок prop=categories — АЛФАВИТНЫЙ, у 19 слов из 22 первая
         видимая категория = алфавитно первая;
      2) pA_rank.py + wiki_parents_snapshot.json: порядок из графа
         РОДИТЕЛЕЙ категорий совпал с золотом 1 раз из 5 — то есть не лучше
         алфавита. Гипотеза «ближайший род выводится из родителей категорий»
         ИЗМЕРЕНИЕМ ОПРОВЕРГНУТА и переиспользована не была;
      3) llm_hierarchy_snapshot.json: иерархия локальной УЧИТЕЛЯ порядок
         ДЕРЖИТ (16 слов из 17 заканчиваются корнем «объекты»), но первый
         элемент совпал с золотом лишь 2 раза из 7 — ранжировать по нему
         ответ нельзя.
      Золото во всех трёх — род из первой строки ру-Вики, выбранного
      РЕДАКТОРАМИ (wiki_lead_snapshot.json), а не моё мнение.

    СЛЕДСТВИЕ: категории Вики отвечают на «X ПРИНАДЛЕЖИТ множеству C», а не
    на «X является C». Ближайшего семейства в них нет НИКАКИМ порядком, и
    первая категория — алфавитная позиция, а не знание. Поэтому ответ
    ПЕРЕСТАЛ ВЫБИРАТЬ одну категорию и НАЧАЛ ПЕРЕЧИСЛЯТЬ известные множества.
    Порядок записи на ответ влиять перестал — что и проверяется ниже.
    """

    def test_record_order_no_longer_invents_the_genus(self):
        write_world(self.dir, [COARSE, PRECISE])
        first = self.answer()
        write_world(self.dir, [PRECISE, COARSE])
        second = self.answer()
        # 1. Ответ больше НЕ выбирает род позицией в списке.
        self.assertNotIn(f"это {COARSE}", first,
                         f"алфавитная категория снова выдаётся за род: {first}")
        self.assertNotIn(f"это {PRECISE}", second,
                         f"категория снова выдаётся за род по порядку записи: {second}")
        # 2. Оба порядка дают ОДИН ОТВЕТ — выбор ушёл из порядка записи.
        self.assertEqual(
            sorted_words(first), sorted_words(second),
            "ответ всё ещё зависит от ПОРЯДКА ЗАПИСИ (дефект не закрыт)")
        # 3. Но ничего не потеряно: обе категории названы.
        for cat in (COARSE, PRECISE):
            self.assertIn(cat, first, f"категория {cat!r} пропала из ответа")
            self.assertIn(cat, second, f"категория {cat!r} пропала из ответа")

    def test_answer_admits_it_is_not_a_definition(self):
        """Ответ обязан называть себя списком множеств, а не определением."""
        write_world(self.dir, [COARSE, PRECISE])
        text = self.answer()
        self.assertIn("Определения", text,
                      f"ответ притворяется определением: {text}")

    def test_single_category_still_answers_directly(self):
        """
        Одиночная категория — ДРУГОЕ знание: пользователь сказал «X — это Y»
        одним мнением, выбирать из одного нечего. Эта ветка замурована, чтобы
        правка множественной ветки не съела одиночную.

        РАНЬШЕ ЗДЕСЬ ЖИЛА ОГОВОРКА «за одиночную категорию из Вики мы тоже
        врум». ЗАМЕР п.A-остаток1 её снял: живой Вики-канал на 22 словах
        ОТДАЛ определение 22 раза из 22, и из тех семи слов, где категория
        осталась одна, «X — это <категория>» ответило НОЛЬ. Ветка «пустое
        определение + одна категория» из Вики НЕДОСТИЖИМА: _first_sentence
        берёт первое предложение статьи, а оно есть у любой нетекстовой
        страницы. Очередь «А-остаток1» мертва — об этом и следующий тест.
        """
        write_world(self.dir, [PRECISE])
        text = self.answer()
        self.assertIn(f"это {PRECISE}", text, text)


class TestWikiBranchCannotFallIntoCategoryAnswer(P26Base):
    """
    Очередь обещала починку: «одна категория от Вики отвечает как род — 7 слов
    из 22». ЗАМЕР постановку убил, а не починил, и это надо зафиксировать
    кодом, иначе через месяц кто-то снова напишет про эти семь слов.

    experiments/pR1_reach.py, три независимых проверки:
      1. Живой Вики-канал, те же семь слов (книга, река, город, экосистема,
         луна, валенок, рыбы): «X — это <категория>» — 0 из 7. Все семь
         ответили первым предложением статьи.
      2. Оффлайн по снимку 22 вступлений: definition непустой — 22 из 22.
      3. Все три канала (локалка / Вики / пользователь): ни один не дал
         «пустое definition + ровно одна категория».

    Значит тест не про ответ, а про Границу: путь, который сегодня мёртв,
    обязан остаться мёртвым, когда Вики поменяет формат выдачи.
    """

    def test_no_channel_can_produce_single_category_answer(self):
        """
        ТО ЖЕ измерение, распространённое на ВСЕ ЧЕТЫРЕ канала (п.A-остаток4
        закрыт осознанно, а не «на всякий случай»). Ветка «X — это <категория>»
        живёт в _definition_answer при ОДНОЙ категории и пустом определении.

        ЗАМЕР 26.10 (свежий, после п.A-остаток3 — ранжирование слияния могло
        схлопнуть список, проверено отдельно):
          * Вики     : definition непустой 22/22 (следующий тест);
          * локалка  : «пустое definition + ровно 1 категория» — 0 слов из 20
                       (test_teacher_prompt_contract);
          * слияние локалка<->Вики в обоих порядках — 0 слов из 20;
          * пользователь: 10 формулировок ответа, включая «флюгер» и «не
                       знаю» — 0: _learn_from_user кладёт в definition ВСЮ
                       фразу, поэтому пустой он бывает только при пустом
                       ответе, когда и записи нет;
          * урок (извлечение из текста): из трёх предложений записана ОДНА
                       сущность, и то ДЕЙСТВИЕ — оно уходит в
                       _describe_action, минуя ветку категории.
        """
        b = self.brain_no_channels()
        singles = []
        for answer in ("это животное", "животное; пушистый", "флюгер",
                       "это флюгер", "прибор", "не знаю", "млекопитающее",
                       "это млекопитающее; теплокровное"):
            data = b._learn_from_user("кошка", answer)
            if not data:
                continue
            ent = data["base_entity"]
            if len(ent.get("hierarchy") or []) == 1 and \
                    not str(ent.get("definition") or "").strip():
                singles.append(answer)
        self.assertEqual(
            singles, [],
            f"канал пользователя породил ветку «X — это Y» из ответов {singles}: "
            "категория — множество, а не род, и выдавать её за определение "
            "нельзя. Пункт А-остаток4 ОБЯЗАН быть перечислен новым ЗАМЕРОМ, "
            "а не правкой формулировки на глаз")

    def brain_no_channels(self):
        from brain.lipid2_brain import Lipid2Brain
        b = Lipid2Brain(memory_dir=self.dir)
        b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
        b.learn_experience = False
        return b

    def test_wiki_concept_always_carries_a_definition(self):
        """У сущности, пришедшей из Вики, определение НЕ пустое."""
        import teachers.teacher_wiki as tw
        leads = json.load(open(os.path.join(ROOT, "experiments",
                              "wiki_lead_snapshot.json"), encoding="utf-8"))
        teacher = tw.WikiTeacher.__new__(tw.WikiTeacher)
        empty = []
        for title, rec in leads.items():
            definition = (teacher._first_sentence(rec["lead"]) or "").strip()
            if not definition:
                empty.append(title)
        self.assertEqual(
            empty, [],
            f"у {len(empty)} статей Вики определение пустое: {empty}. Значит "
            "ветка «пустое определение + категория» снова ДОСТИЖИМА и ответ "
            "сможет выдать алфавитную категорию за род — ровно то, из-за чего "
            "в очкреди стоит пункт А-остаток1.")



class TestAnswerAgreesWithTheWordGender(P26Base):
    """
    ОТДЕЛЬНЫЙ ДЕФЕКТ, который дал тот же замер (experiments/pR1_reach.py).

    Фразу «известно, что он относится к:» я вставил в ответ накануне, и
    местоимение в ней ЗАШТОПАНО мужским родом. ЗАМЕР на шести словах разного
    рода (вода, туфля, окно, озеро, ёлка, стол): род не совпал в 5 случаях из
    6 — «Определения у вода нет; известно, что ОН относится к: жидкости».
    Единственное совпадение пришлось на «стол», то есть на слово мужского
    рода: формула работает только для одного рода из трёх.

    Ошибка того же класса, что уже была в проекте с «который»: текст ответа
    собран из догадки, а не из разбора. Правка — формулировка без рода.
    """

    def test_multi_category_answer_never_asserts_masculine_gender(self):
        for word in ("вода", "туфля", "окно", "озеро", "ёлка", "стол"):
            # Род проверяется на СВОЁМ слове: если мир о другом, мозг
            # отвечает отказом и проверка на род становится пустой.
            write_world(self.dir, [COARSE, PRECISE], concept=word)
            text = self.answer(f"что такое {word}")
            self.assertIn("Определения", text,
                          f"это не ответ о «{word}», проверка не состоялась: "
                          f"{text}")
            self.assertNotIn(
                "что он относится", text,
                f"ответ про «{word}» приписывает ему мужской род: {text}")

    def test_reworded_answer_still_names_both_sets(self):
        """Правка формулировки не имеет права съесть сами категории."""
        write_world(self.dir, [COARSE, PRECISE])
        text = self.answer()
        self.assertIn(COARSE, text, text)
        self.assertIn(PRECISE, text, text)
        self.assertIn("Определения", text, text)

def sorted_words(text):
    return sorted(w.strip(" .,;:") for w in text.split() if len(w) > 2)



class TestNoSilentWeightReader(unittest.TestCase):
    """
    Вес ребра во всём проекте обязан читаться ТОЛЬКО для печати и для диагноза
    расхождений. Если появится четвёртый читатель, выбирающий ответ, — этот
    тест обязан покраснеть первым.
    """

    def test_weight_is_read_only_in_known_places(self):
        import re
        readers = []
        for folder in ("core", "brain"):
            for fname in sorted(os.listdir(os.path.join(ROOT, folder))):
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(ROOT, folder, fname)
                with open(path, encoding="utf-8") as fh:
                    for no, line in enumerate(fh, 1):
                        if line.lstrip().startswith("#"):
                            continue
                        if re.search(r'syn\[.weight.\]|syn_info\[.weight.\]|'
                                      r'\[.weight.\]\s*[\)\]\n,<]', line):
                            readers.append(f"{folder}/{fname}:{no}")
        # core/graph.py:215 — печать; core/storage.py:108 — упаковка в бинарный
        # формат (запись, не выбор). Дальше вес читается только через
        # observation_weight (диагноз, replay, сон), а не из ребра графа.
        allowed_prefixes = ("core/graph.py", "core/storage.py")
        stray = [r for r in readers if not r.startswith(allowed_prefixes)]
        self.assertEqual([], stray,
                         "появилось новое чтение веса РЕБРА — именно так вес "
                         "незаметно превращается в выбор ответа (п.26): "
                         + ", ".join(stray))


if __name__ == "__main__":
    unittest.main(verbosity=2)
