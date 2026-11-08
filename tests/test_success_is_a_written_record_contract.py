# -*- coding: utf-8 -*-
"""КОНТРАКТ: УСПЕХ КАНАЛА = ВОШЕДШАЯ ЗАПИСЬ, А НЕ НЕПУСТОЙ ОТВЕТ.

ЗАМЕРЕННАЯ ДЫРА (09.11, замер experiments/measure_use_vs_commit.py)
--------------------------------------------------------------------
Мозг объявлял канал успешным по НЕПУСТОСТИ возвращённого словаря
(`if learned_data:` в brain/lipid2_brain.py). Замерено руками на канале,
который ОТВЕТИЛ, но ничего не записал (живая Gemma делает это регулярно:
«ответ без вызова инструмента»):

    живая база:      сущностей 3 -> 3   (знание НЕ вошло)
    unknown_words:   было «дождь» -> пусто   (СЛОВО ВЫПАЛО ИЗ ОЧЕРЕДИ УРОКОВ)
    дофамин:         255
    channel_found:   llm += 1
    напечатано:      «Знание получено от локальной LLM»,
                     «Граф пополнен сущностью 'дождь'»

Итог: перестал считать слово неизвестным, не узнав его. Ровно то, что
запрещает закон проекта «ложная определённость хуже честной дыры»: дыра
замазана, очередь уроков слово потеряла, спросить его больше нечем.

ЧТО ЗАКРЕПЛЕНО
---------------
1. УСПЕХ ИЗМЕРЯЕТСЯ ФАКТОМ ЗАПИСИ. Канал считается успешным, только если
   после его работы сущность (или её форма) появилась в файле мира, в
   который канал пишет. Непустой словарь без записи = ОТКАЗ.
2. ОТКАЗ ИМЕЕТ ПРАВО ТОЛЬКО НА ТРЕВОГУ. unknown_words обязано остаться:
   слово не выучено, оно обязано остаться в очереди уроков.
3. НАЗВАТЬ ОТКАЗ УСПЕХОМ НЕЛЬЗЯ НИ В ОДНОМ СЛОВЕ. Ни «выучено», ни
   «вошло в граф», ни «Знание получено» не имеют права прозвучать, если
   записи нет.
4. НАСТОЯЩИЙ УСПЕХ НЕ ЛОМАЕТСЯ. Если запись вошла (канал владельца),
   unknown_words очищается и дофамин имеет право вырасти.

МУТАЦИИ, на которых контракт обязан покраснеть:
  * вернуть `if learned_data:` без проверки записи — падает test_1/2/3;
  * чистить unknown_words до проверки записи — падает test_2;
  * оставить «выучено» в тексте при отказе — падает test_3;
  * сделать проверку записи строже записи (например, требовать точного
    ключа без форм слова) — падает test_5 на падежной форме.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lang.ru.lemma import install  # noqa: E402
install()

from brain.lipid2_brain import Lipid2Brain  # noqa: E402

СЛОВО = "дождь"


def мир_с_очередью():
    """Мир, где «дождь» стоит в очереди уроков и нигде больше."""
    return {"entities": {"вода": {"name": "вода", "actions": ["гасить"]},
                         "огонь": {"name": "огонь"}},
            "observations": {
                "вода|CAN_DO|гасить": {"head": "вода", "relation": "CAN_DO",
                                       "tail": "гасить", "source": "user",
                                       "lineage": "owner", "support": 1,
                                       "confidence": 0.9}},
            "unknown_words": {СЛОВО: {"word": СЛОВО, "reason": "не знаю"}}}


class Пустобрёх:
    """Канал, который ОТВЕТИЛ и НИЧЕГО не записал.

    Ровно та форма, которую живая Gemma выдаёт каждый раз, когда модель
    рассуждает про онтологию, но не вызывает инструмент записи. Возврат
    непустой — вот что старый код считал успехом.
    """

    last_failure = None
    last_text = "Дождь — это осадки."

    def run_agent(self, *args, **kwargs):
        return {"base_entity": {"concept": СЛОВО, "definition": "осадки"},
                "matched_concept": СЛОВО}


class МозгВПробирке(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.say = []
        self.b = Lipid2Brain(memory_dir=self.dir)
        self.b._say = lambda m, **kw: self.say.append(str(m))
        self.b.use_wiki_teacher = False
        self.b.ask_user_enabled = False
        self.b.use_llm_teacher = True
        self.b.teacher_llm = Пустобрёх()
        self.b._save_world(мир_с_очередью())

    def tearDown(self):
        self._tmp.cleanup()

    def живой(self):
        return json.load(open(os.path.join(self.dir, "lipid_world.json"),
                              encoding="utf-8"))

    def спроси(self):
        self.b._trigger_curiosity_and_learn(СЛОВО)

    # 1. УСПЕХ = ЗАПИСЬ --------------------------------------------------
    def test_1_loud_answer_without_a_record_is_not_success(self):
        """Канал громко ответил и ничего не записал: это отказ.

        Мутация «оставить if learned_data» делает этот тест красным:
        мозг засчитает каналу успех, которого не было.
        """
        self.спроси()
        мир = self.живой()
        self.assertNotIn(СЛОВО, мир["entities"],
                         "канал ничего не записал, но сущность появилась: "
                         "успех нарисован из воздуха")
        self.assertEqual(0, self.b.channel_found["llm"],
                         "каналу засчитана находка без вошедшей записи")

    def test_2_the_word_must_stay_in_the_lesson_queue(self):
        """Невыученное слово ОБЯЗАНО остаться в unknown_words.

        Замеренный ущерб: слово пропало из очереди (было -> пусто), то есть
        дыра перестала быть видимой и больше никогда не будет задачена.
        """
        self.спроси()
        self.assertIn(СЛОВО, self.живой()["unknown_words"],
                      "слово выучено? нет. выпало из очереди уроков? да. "
                      "это замазанная дыра")

    def test_3_refusal_must_not_wear_success_words(self):
        """Ни одно «выучено/вошло/получено» не звучит без записи."""
        self.спроси()
        текст = " | ".join(self.say).lower()
        for запрещённое in ("выучено", "вошло в граф", "знание получено"):
            self.assertNotIn(запрещённое, текст,
                             f"при отказе прозвучало «{запрещённое}»")

    def test_4_dopamine_does_not_fire_on_an_empty_answer(self):
        """Дофамин — про входящее знание, а не про красивую речь канала."""
        self.спроси()
        self.assertLess(self.b.hormone_gate.dopamine, 200,
                        "дофамин поднят при нулевом приросте графа")

    # 2. НАСТОЯЩИЙ УСПЕХ НЕ ЛОМАЕТСЯ -------------------------------------
    def test_5_a_real_record_still_counts_as_success(self):
        """Если запись вошла — успех законен: очередь чистится, дофамин растёт.

        Это обратная сторона: кирпич не имеет права превратить поиск в
        вечный отказ.
        """
        self.b.teacher_llm = None
        self.b.use_llm_teacher = False
        self.b.ask_user_enabled = True
        self.b.ask_user_callback = lambda concept: f"{concept} — это осадки"
        self.b._trigger_curiosity_and_learn(СЛОВО)
        мир = self.живой()
        self.assertIn(СЛОВО, мир["entities"], "владелец ответил, а записи нет")
        self.assertNotIn(СЛОВО, мир["unknown_words"],
                         "слово выучено, но осталось в очереди неизвестных")

    def test_6_success_words_appear_on_a_real_record(self):
        """На настоящем успехе слово «выучено» обязано прозвучать."""
        self.b.teacher_llm = None
        self.b.use_llm_teacher = False
        self.b.ask_user_enabled = True
        self.b.ask_user_callback = lambda concept: f"{concept} — это осадки"
        self.b._trigger_curiosity_and_learn(СЛОВО)
        self.assertIn("выучено", " | ".join(self.say).lower())

    # 3. ФОРМА СЛОВА НЕ ИМЕЕТ ПРАВА СПРЯТАТЬ ЗНАНИЕ ----------------------
    def test_7_recorded_under_a_form_of_the_word_is_still_a_record(self):
        """Запись под падежной формой слова — всё равно запись.

        Если проверка записи сделана сравнением точного ключа, форма
        «дождя» даст ложный отказ, и мозг будет вечно искать уже известное.
        Мутация «сравнивать только key == concept» краснеет здесь.
        """
        directory = self.dir
        путь = os.path.join(directory, "lipid_world.json")

        class ПишетПодФормой:
            """Канал, который записал сущность в форму слова, а не в именительную."""

            last_failure = None

            def run_agent(self, *args, **kwargs):
                with open(путь, encoding="utf-8") as fh:
                    мир = json.load(fh)
                мир["entities"]["дождя"] = {"name": "дождя",
                                            "definition": "осадки"}
                with open(путь, "w", encoding="utf-8") as fh:
                    json.dump(мир, fh, ensure_ascii=False)
                return {"base_entity": {"concept": СЛОВО},
                        "matched_concept": СЛОВО}

        self.b.teacher_llm = ПишетПодФормой()
        self.b._trigger_curiosity_and_learn(СЛОВО)
        self.assertEqual(1, self.b.channel_found["llm"],
                         "запись вошла (под формой слова), а каналу отказали: "
                         "проверка записи слишком строга к ключу")

    def test_3b_all_three_channels_silent_is_not_success(self):
        """Вторая ветка отказа: каналы РАБОТАЛИ и промолчали все три.

        У отказа ДВЕ дороги с РАЗНЫМ текстом: «знание НЕ ДОБЫТО — <причины>»
        (были выключенные или сломавшиеся каналы) и «все каналы пусты» (все
        работали и вернули пусто). Первый тест идёт по дороге «ответ без
        записи» при выключенных 2 и 3; эта — по дороге трёх молчунов, где
        причин называть нечего. Мутант «назвать отказ выучено» выжил именно
        потому, что вторая дорога была не покрыта: тест знал половину отказа.

        Здесь намеренно включены ВСЕ три канала: канал 3 без подключённого
        обратного канала честно возвращает None (core.logchannel.request
        при отсутствии обработчика не блокируется), поэтому прогона по сети
        и ожидания человека тут нет.
        """
        class Молчун:
            last_failure = None

            def run_agent(self, *args, **kwargs):
                return None

            def fetch_concept(self, word):
                return None

            def write_to_world(self, data):
                return False

        self.b.teacher_llm = Молчун()
        self.b.teacher_wiki = Молчун()
        self.b.use_wiki_teacher = True
        self.b.ask_user_enabled = True
        self.b.ask_user_callback = lambda concept: None
        self.b._trigger_curiosity_and_learn(СЛОВО)
        текст = " | ".join(self.say).lower()
        for запрещённое in ("выучено", "вошло в граф", "знание получено"):
            self.assertNotIn(запрещённое, текст,
                             f"при трёх молчаливых каналах прозвучало "
                             f"«{запрещённое}»")
        self.assertIn(СЛОВО, self.живой()["unknown_words"],
                      "все каналы промолчали, а слово исчезло из очереди "
                      "уроков: дыра замазана")

    # 4. АДРЕС КАНАЛА ----------------------------------------------
    def test_8_a_record_in_the_search_window_counts(self):
        """Канал писал в ОКНО ПОИСКА (карантин) — запись там и надо мерить.

        Во время поиска по вопросу мозг перенаправляет учителей в копию
        живой базы (curiosity.begin_search), поэтому живая база при законном
        успехе пуста. Проверка «смотри всегда lipid_world.json» дала бы
        ВЕЧНЫЙ ОТКАЗ каналу, который честно записал: одна ложная
        определённость лечится другой. Замер: мутация «адрес = живая база»
        краснеет ровно здесь.
        """
        окно = os.path.join(self.dir, "staging_world.json")
        with open(окно, "w", encoding="utf-8") as fh:
            json.dump(мир_с_очередью(), fh, ensure_ascii=False)

        class ПишетВОкно:
            last_failure = None
            world_filepath = окно

            def run_agent(self, *args, **kwargs):
                with open(окно, encoding="utf-8") as fh:
                    мир = json.load(fh)
                мир["entities"][СЛОВО] = {"name": СЛОВО, "definition": "осадки"}
                with open(окно, "w", encoding="utf-8") as fh:
                    json.dump(мир, fh, ensure_ascii=False)
                return {"base_entity": {"concept": СЛОВО},
                        "matched_concept": СЛОВО}

        self.b.teacher_llm = ПишетВОкно()
        self.b._trigger_curiosity_and_learn(СЛОВО)
        self.assertEqual(1, self.b.channel_found["llm"],
                         "запись вошла в окно поиска, а каналу отказали: "
                         "успех меряется не по тому файлу")

if __name__ == "__main__":
    unittest.main(verbosity=2)
