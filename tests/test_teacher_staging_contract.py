# -*- coding: utf-8 -*-
"""Учитель учится в КАРАНТИН, а не во Вселенную (сво право хозяина).

ЧТО ЗАМЕРЕНО (07.11, живой Учитель через llama.cpp, вопрос про дождь и
огонь). Хозяин велел: «пока кортизол высокий — искать; ответ найден —
кортизол падает, и дофамин МОЖЕТ записать данные в базу». Шлюз
(HormoneGate.dopamine_ok) и писатель кандидата (Curiosity.promote ->
brain._write_knowledge) сделаны и работают. Но замер показал главное:

    вопрос «может ли дождь потушить огонь» при включённом живом Учителе
    ИЗМЕНИЛ lipid_world.json (хэш dfe15441 -> fd944cc2, сущностей 1 -> 4),
    а ОТВЕТА НЕ ДАЛ: «не нашёл, чем дождь может потушить огонь»,
    кортизол 200, целей 3.

Причина: Учитель (teachers/teacher1.py) и Вики-учитель пишут в файл сами,
по адресу, прошитому в конструкторе, — В ОБХОД шлюза. Шлюз охранял только
тот путь, который проходит через Curiosity.promote. Честная формулировка
того дня: «дофамин разрешает запись ТОМУ, КТО ПОПРОСИЛ РАЗРЕШЕНИЯ». Это не
выполненное слово хозяина, а его половина.

ЧТО ЗАКРЕПЛЯЕТСЯ ЗДЕСЬ. Пока поиск не завершён (кортизол высокий, ответа
нет), любой канал знания пишет в КАРАНТИННЫЙ файл — копию живой базы, чтобы
учитель видел, что уже известно, и не повторялся. Живая база за время поиска
НЕ МЕНЯЕТСЯ НИ НА ОДИН БАЙТ. В живую базу знание пускает только promote,
а тот стоит за dopamine_ok и за проверкой зацепки.

ЗАЧЕМ ПРОВЕРКА ПО ХЭШУ, а не по содержимому. Половина дыр рождалась именно
тем, что «записали по мелочи»: одна лишняя сущность, одно поле. Хэш живого
файла — единственный способ поймать запись, которую тест не ждал.

ЧТО УПАДЁТ, ЕСЛИ МЕХАНИКУ СЛОМАЮТ (мутации проверялись руками):
  * «пишем куда попало» (staging не создаётся) — умирает тест 1, база
    меняется до ответа;
  * «забрали из карантина всё, включая старое» — умирает тест 2: копия живой
    базы переехала бы в кандидаты целиком, и promote затащил бы в базу чужие
    знания под моей меткой;
  * «promote не смотрит на шлюз» — умирает тест 3;
  * «relief не открывает шлюз» — умирает тест 4 (знание навсегда осталось бы
    в карантине, то есть выученное пропало бы).
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from brain.curiosity import Curiosity, HormoneGate
from brain.simulator import StaticKnowledge


from core.records import merge_entity
# Тот же запрет, что и для производства: тест, сводящий формы слов молча,
# проверяет эвристику по буквам и думает, что проверяет язык.
from lang.ru.lemma import install as _install_lemma_service

_install_lemma_service()


def sha(path):
    """Хэш файла целиком. Любая запись — любая буква — меняет его."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


LIVE_KNOWLEDGE = {
    "entities": {"вода": {"concept": "вода", "hierarchy": ["жидкость"]}},
    "observations": {},
    "unknown_words": {},
}

# Кандидат, который Учитель принёс бы по неизвестному слову «роса».
TEACHER_ANSWER = {
    "base_entity": {"concept": "роса",
                    "hierarchy": ["вода"],           # зацепка за известное
                    "definition": "влага, оседающая на траве ночью",
                    "properties": [], "actions": []},
    "derived_entity": {},
}


def teacher_write(path, data):
    """Запись в стиле Учителя: merge_entity в файл по указанному адресу.

    Это НЕ импорт teachers/*: живой Учитель поднимает модель и ходит в
    llama.cpp, а проверяется здесь не модель, а ПУТЬ записи. Формат файла и
    правила слияния те же (core.records.merge_entity), поэтому подмена
    безопасна: меняя путь, мы меняем ровно то, что проверяем.
    """
    from core.records import merge_entity
    with open(path, encoding="utf-8") as f:
        world = json.load(f)
    ent = data["base_entity"]
    key = ent["concept"].lower().strip()
    world["entities"][key] = merge_entity(world["entities"].get(key), dict(ent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(world, f, ensure_ascii=False)
    return key


class TestSearchWritesOnlyToQuarantine(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.live = os.path.join(self.dir, "lipid_world.json")
        with open(self.live, "w", encoding="utf-8") as f:
            json.dump(LIVE_KNOWLEDGE, f, ensure_ascii=False)
        self.cur = Curiosity(self.dir)
        # Писатель — та же работа, что brain._write_knowledge: кандидат входит
        # в живую файл через merge_entity. Без писателя promote честно пишет
        # «некому» и возвращает 0, поэтому тесту обязан быть дан КАНАЛ ЗАПИСИ.
        self.cur.writer = self._writer
        self.gate = HormoneGate()
        self.goal = self.cur.note_unknown("роса", why="нужен ответ про дождь")
        self.gate.stress(255)                    # не знаю — ищу

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _writer(self, entities):
        """Канал записи во Вселенную (аналог brain._write_knowledge)."""
        with open(self.live, encoding="utf-8") as f:
            world = json.load(f)
        n = 0
        for key, rec in (entities or {}).items():
            key = str(key).lower().strip()
            if not key or not isinstance(rec, dict):
                continue
            prev = world["entities"].get(key)
            world["entities"][key] = merge_entity(prev, dict(rec))
            n += 1
        with open(self.live, "w", encoding="utf-8") as f:
            json.dump(world, f, ensure_ascii=False)
        return n

    def test_1_search_does_not_touch_live_base(self):
        """Поиск не имеет права менять живую базу ни на байт."""
        before = sha(self.live)
        staging = self.cur.begin_search()       # копия живой базы для чтения
        self.assertTrue(os.path.exists(staging),
                        "поиск обязан дать каналу файл для учёбы")
        with open(staging, encoding="utf-8") as f:
            self.assertIn("вода", f.read(),
                          "в карантине Учитель обязан видеть известное — "
                          "иначе он начнёт заново учить то, что уже есть")
        teacher_write(staging, TEACHER_ANSWER)
        self.assertEqual(before, sha(self.live),
                         "живая база изменена ДО ответа: дофамин ещё не "
                         "спрашивали, писать было нельзя")
        self.assertEqual(1, len(self.cur.open_goals()),
                         "цель поиска обязана пережить unsuccessful попытку")
        self.assertTrue(self.gate.must_search(),
                        "не нашли — тревога обязана остаться высокой")

    def test_2_intake_takes_only_new(self):
        """Из карантина забираются НОВЫЕ знания, а не копия всей базы."""
        staging = self.cur.begin_search()
        teacher_write(staging, TEACHER_ANSWER)
        taken = self.cur.collect_from_staging()
        self.assertEqual(["роса"], sorted(taken),
                          "в кандидаты попал не только ответ Учителя")
        self.assertEqual(1, len(self.cur.staged()),
                         "кандидатов должен быть ровно один")
        with open(self.live, encoding="utf-8") as f:
            self.assertEqual(json.dumps(LIVE_KNOWLEDGE, sort_keys=True,
                                        ensure_ascii=False),
                             json.dumps(json.load(f), sort_keys=True,
                                        ensure_ascii=False),
                             "collect_from_staging обязан читать, а не писать")

    def test_3_no_answer_still_no_write(self):
        """Знание есть, ответа нет — во Вселенную не идёт."""
        staging = self.cur.begin_search()
        teacher_write(staging, TEACHER_ANSWER)
        self.cur.collect_from_staging()
        before = sha(self.live)
        moved = self.cur.promote(
            lambda: StaticKnowledge(self._live_world()), self.gate,
            answer_found=False)
        self.assertEqual(0, moved,
                         "без ответа promote записал знание — дофамин "
                         "опять стал отчётом, а не разрешением")
        self.assertEqual(before, sha(self.live))
        self.assertEqual(1, len(self.cur.staged()),
                         "отклонённый кандидат обязан остаться в карантине")

    def test_4_answer_opens_the_gate(self):
        """Ответ найден — тревога падает, знание входит, карантин пустеет."""
        staging = self.cur.begin_search()
        teacher_write(staging, TEACHER_ANSWER)
        self.cur.collect_from_staging()
        self.gate.relief()
        moved = self.cur.promote(
            lambda: StaticKnowledge(self._live_world()), self.gate,
            answer_found=True)
        self.assertEqual(1, moved,
                         "найденный ответ обязан открыть дорогу знанию — "
                         "иначе выученное пропадает")
        with open(self.live, encoding="utf-8") as f:
            ents = json.load(f)["entities"]
        self.assertIn("роса", ents)
        self.assertEqual(0, len(self.cur.staged()))
        self.assertFalse(self.gate.must_search())

    def test_6_second_attempt_keeps_previous_gains(self):
        """Повторный поиск НЕ затирает то, что принесли в прошлой попытке.

        Провал поиска — обычное дело: один канал молчит, другой обрезал ответ.
        Если begin_search перельёт карточку копией живой базы, труд учителей
        исчезнет, а система сделает вид, что ничего не было.
        """
        staging = self.cur.begin_search()
        teacher_write(staging, TEACHER_ANSWER)
        self.assertEqual(["роса"], self.cur.collect_from_staging())
        # вторая попытка по тому же вопросу
        again = self.cur.begin_search()
        self.assertEqual(staging, again)
        with open(again, encoding="utf-8") as f:
            self.assertIn("роса", f.read(),
                          "повторный поиск затёр то, что уже выучено")
        # Собираем повторно: новых сущностей нет, и старых прибавить нельзя.
        self.assertEqual([], self.cur.collect_from_staging(),
                         "повторный собрал то, что уже стоит в кандидатах")
        self.assertEqual(1, len(self.cur.staged()),
                         "повторный сбор умножил кандидатов — promote перенесёт дубль")
        self.assertTrue(self.cur.clear_staging())
        self.assertFalse(os.path.exists(self.cur.staging_path()))

    def test_7_without_staging_the_hole_is_real(self):
        """Живое доказательство дыры: прямой путь Учителя меняет базу.

        Тест НЕ про желаемое поведение, а про то, почему оно нужно: если
        писать по старому адресу (как писали оба учителя до этой правки),
        база меняется без спроса. Пока этот тест зелёный, дыра существует и
        механизм staging не «потерялся» — он просто не задействован на этом
        пути.
        """
        before = sha(self.live)
        teacher_write(self.live, TEACHER_ANSWER)
        self.assertNotEqual(before, sha(self.live))

    def _live_world(self):
        with open(self.live, encoding="utf-8") as f:
            return json.load(f)


class TestObservationGoesToQuarantine(unittest.TestCase):
    """Свидетельство канала — тоже запись, и оно обязано слушаться шлюза.

    Вторая половина той же дыры, пойманная живым замером: после правки
    учителей число сущностей в живой базе стояло на месте (1), а файл всё
    равно менялся — росли observations (0 -> 3). Недоученный вопрос оставлял
    в базе «свидетельства», хотя дофамин молчал. Мозг обязан на время поиска
    перенаправлять ИХ по тому же адресу, что и сущности.
    """

    def setUp(self):
        from brain.lipid2_brain import Lipid2Brain
        self.dir = tempfile.mkdtemp()
        self.live = os.path.join(self.dir, "lipid_world.json")
        with open(self.live, "w", encoding="utf-8") as f:
            json.dump(LIVE_KNOWLEDGE, f, ensure_ascii=False)
        self.brain = Lipid2Brain(memory_dir=self.dir)
        self.cur = Curiosity(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _record(self):
        self.brain._record_observation("роса", {"hierarchy": ["вода"]},
                                       confidence=0.4, evidence="роса на траве",
                                       source="llm")

    def test_8_search_window_redirects_observations(self):
        before = sha(self.live)
        staging = self.cur.begin_search()
        self.brain._staging_target = staging
        try:
            self._record()
        finally:
            self.brain._staging_target = None
        self.assertEqual(before, sha(self.live),
                         "наблюдение записано в живую базу до ответа")
        with open(staging, encoding="utf-8") as f:
            self.assertIn("роса", json.dumps(json.load(f), ensure_ascii=False),
                          "свидетельство канала пропало: искать заново нечем")

    def test_9_outside_search_window_write_is_normal(self):
        """Вне окна поиска запись идёт как обычно — механизм не вечный замок."""
        before = sha(self.live)
        self._record()
        self.assertNotEqual(before, sha(self.live),
                            "перенаправление залипло: мозг перестал учиться вовсе")


if __name__ == "__main__":
    unittest.main()
