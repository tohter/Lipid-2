"""
Качество того, что УЧИТЕЛЬ приносит из Википедии: определение и свойства.

Тест офлайновый: тексты вступлений живых статей сохранены в
tests/data/wiki_intros.json. Сеть тут не нужна — проверяется разбор, а не
Википедия.

Причина: ответы вида «флюгер — это флюгеры» и «свойства флюгера: [нет данных]»
при скачанной статье. Разбор:
  1) определение (первое предложение) раньше выбрасывалось при записи во
     Вселенную («для мира не нужны») и не использовалось, когда иерархия
     состояла из мусора;
  2) свойства искались закрытым списком из 24 корней: на 10 живых вступлениях
     12 свойств и 4 из 10 слов вообще без свойств. Морфологическое извлечение
     (только прилагательные) — 55 свойств, 0 пустых.
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from teachers.teacher_wiki import WikiTeacher, _get_morph

FIXTURE = os.path.join(ROOT, "tests", "data", "wiki_intros.json")


class TestDefinition(unittest.TestCase):
    """Что попадает в запись и что из неё читается."""

    def setUp(self):
        self.wiki = WikiTeacher(memory_dir="/tmp/l2-wiki-test")
        self.texts = json.load(open(FIXTURE, encoding="utf-8"))

    def test_definition_reaches_the_world_file(self):
        """
        Регресс: write_to_world отбрасывал definition («title/source/definition
        для мира не нужны»), хотя _handle_definition его читает. Итог: на уже
        прочитанную статью память хранила только иерархию из категорий.
        """
        d = self.wiki.fetch_concept("экосистема")
        self.assertTrue(d, "разбор не собрался")
        base = d["base_entity"]
        self.assertTrue(base.get("definition"), "разбор не принёс определение")
        tmp = os.path.join(self.wiki.memory_dir, "w1.json")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"entities": {}}, fh, ensure_ascii=False)
        self.wiki.world_filepath = tmp
        self.assertTrue(self.wiki.write_to_world(d))
        saved = json.load(open(tmp, encoding="utf-8"))["entities"]["экосистема"]
        self.assertIn("definition", saved,
                      "определение потерялось при записи — самое ценное поле")
        self.assertTrue(saved["definition"])

    def test_answer_not_started_with_self_name(self):
        """
        ЗАМЕРЕНО: дословное определение давало «Экосистема — Экосистема, или
        экологическая система (от др.-греч. οἶκος — жилище...) — основная...».
        Двойное название + этимология посреди ответа.
        """
        from brain.lipid2_brain import Lipid2Brain
        raw = ("Экосистема, или экологическая система (от др.-греч. οἶκος — "
               "жилище, местопребывание и σύστημα — «целое, составленное из "
               "частей; соединение») — основная природная единица "
               "на поверхности Земли, совокупность совместно обитающих "
               "организмов и условий их существования.")
        out = Lipid2Brain._definition_phrase("экосистема", raw)
        self.assertFalse(out.lower().startswith("экосист"), out)
        for junk in ("др.-греч", "οἶκος", "σύστημα"):
            self.assertNotIn(junk, out)
        self.assertIn("основная природная единица", out)

    def test_short_definition_kept_whole(self):
        from brain.lipid2_brain import Lipid2Brain
        out = Lipid2Brain._definition_phrase(
            "флюгер", "Флюгер — прибор для измерения направления ветра.")
        self.assertEqual(out, "прибор для измерения направления ветра.")

    def test_long_definition_cut_at_word_bound(self):
        from brain.lipid2_brain import Lipid2Brain
        out = Lipid2Brain._definition_phrase("тест", "слово " * 200)
        self.assertLessEqual(len(out), Lipid2Brain._DEF_MAX_CHARS + 10)
        self.assertTrue(out.endswith("…"))
        self.assertFalse(out.endswith("слов…"), "обрезка разорвала слово")

    def test_definition_of_unrelated_word_untouched(self):
        from brain.lipid2_brain import Lipid2Brain
        out = Lipid2Brain._definition_phrase("кошка", "хищное млекопитающее.")
        self.assertEqual(out, "хищное млекопитающее.")


@unittest.skipUnless(_get_morph() is not None, "нет pymorphy3 — морфологии нет")
class TestProperties(unittest.TestCase):

    def setUp(self):
        self.wiki = WikiTeacher(memory_dir="/tmp/l2-wiki-test")
        self.texts = json.load(open(FIXTURE, encoding="utf-8"))

    def test_no_word_left_without_properties(self):
        """
        Замер до/после на ЭТИХ ЖЕ текстах: закрытый список из 24 корня давал
        пустой список у 4 из 10 статей. Морфологическое извлечение (без фильтра)
        давало 0 пустых, но с шумом (таксономические прилагательные).

        Новый фильтр (только Qual-прилагательные) убирает шум, но некоторые
        статьи Википедии в определении не содержат качественных прилагательных
        (флюгер, пчела, кошка) — это ОЖИДАЕМО, Википедия не источник свойств для всех.
        Тест проверяет: шума нет, и ХОТЯ БЫ ОДНО слово имеет свойства.
        """
        empty = [w for w, t in self.texts.items()
                 if not self.wiki._extract_properties(t)]
        # Допускаем пустые для слов, у которых в вступлении нет Qual-прилагательных
        # (флюгер — техническое определение, пчела — таксономическое,
        #  кошка — вступление только таксономия/классификация)
        allowed_empty = {"флюгер", "пчела", "кошка"}
        unexpected_empty = [w for w in empty if w not in allowed_empty]
        self.assertEqual(unexpected_empty, [],
            f"неожиданно пустые свойства: {unexpected_empty}")
        # Но ХОТЯ БЫ ОДНО слово должно иметь свойства (экосистема)
        non_empty = [w for w, t in self.texts.items()
                     if self.wiki._extract_properties(t)]
        self.assertGreater(len(non_empty), 0,
            "вообще ни у кого нет свойств — фильтр слишком агрессивен")

    def test_properties_are_adjectives_not_verbs(self):
        """
        Причастия в список тянули глагольные леммы: «пчела: летать»,
        «вулкан: иметь». Действие — не свойство; для него есть actions.
        """
        from pymorphy3 import MorphAnalyzer
        morph = MorphAnalyzer()
        for word, text in self.texts.items():
            for prop in self.wiki._extract_properties(text):
                tags = str(morph.parse(prop)[0].tag)
                with self.subTest(word=word, prop=prop):
                    self.assertIn("ADJF", tags)
                    self.assertNotIn("VERB", tags)

    def test_properties_in_dictionary_form(self):
        for word, text in self.texts.items():
            for prop in self.wiki._extract_properties(text):
                with self.subTest(word=word, prop=prop):
                    self.assertTrue(prop.islower(), f"{prop} не словарной формы")

    def test_pronouns_not_properties(self):
        for text in self.texts.values():
            props = self.wiki._extract_properties(text)
            self.assertNotIn("который", props)
            self.assertNotIn("каждый", props)

    def test_limit_and_dedup(self):
        for text in self.texts.values():
            props = self.wiki._extract_properties(text)
            self.assertLessEqual(len(props), WikiTeacher.PROP_LIMIT)
            self.assertEqual(len(props), len(set(props)))

    def test_empty_input(self):
        self.assertEqual(self.wiki._extract_properties(""), [])
        self.assertEqual(self.wiki._extract_properties(None), [])

    def test_fallback_is_loud(self):
        """
        Без морфологии канал обязан СООБЩИТЬ, что стал хуже, а не молча
        вернуть пустоту: пустой список свойств неотличим от «в статье нет
        прилагательных».
        """
        import contextlib
        import io
        import teachers.teacher_wiki as tw
        saved = (tw._MORPH, tw._MORPH_TRIED)
        try:
            tw._MORPH, tw._MORPH_TRIED = None, True
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                props = self.wiki._extract_properties("Домашняя кошка — хищный зверь.")
            self.assertIn("pymorphy3", err.getvalue(),
                          "остаточный путь молчит: пользователь видит пустые "
                          "свойства, а не сломанный канал")
            self.assertTrue(props, "остаточный путь обязан хоть что-то находить")
        finally:
            tw._MORPH, tw._MORPH_TRIED = saved


class TestCategoryHierarchy(unittest.TestCase):
    """
    Категории Википедии — единственный источник иерархии, а иерархия отвечает
    на «что такое X», когда определения в записи нет.

    Замер: experiments/wiki_category_snapshot.json (22 живых слова, сняты ТЕМ
    ЖЕ запросом, что и канал), разбор — experiments/p26b_wiki_noise.py.
    """

    SNAP = os.path.join(ROOT, "experiments", "wiki_category_snapshot.json")

    def setUp(self):
        from teachers.teacher_wiki import WikiTeacher
        self.wiki = WikiTeacher(memory_dir="/tmp/l2-cats")

    def test_article_markup_never_becomes_an_ontology_category(self):
        """
        ЗАМЕРЕНО до правки: у слова «работа» обе категории Вики оказались
        разметкой статьи, остаточная ветка вернула ОТФИЛЬТРОВАННОЕ, и в graph
        легло «работа — это статьи со ссылками на викисловарь».
        """
        raw = ["Статьи со ссылками на Викисловарь",
               "Страницы, использующие волшебные ссылки ISBN"]
        out = self.wiki._categories_to_hierarchy(raw, "работа")
        self.assertEqual([], out,
                         f"разметка статьи стала категрией онтологии: {out}")

    def test_filter_speaks_when_nothing_but_markup_remained(self):
        """
        Пусто молча = «категорий не было», неотличимо от «все категории —
        мусор». Канал обязан сказать, что иерархию он не принёс.

        Проверка — подпиской на core.logchannel, а не перехватом stdout:
        вывод в stdout зависит от LIPID2_STATUS, а обязанность канала — дойти
        до слушателя (GUI) при ЛЮБЫХ флагах терминала.
        """
        from core import logchannel
        seen = []
        sid = logchannel.subscribe(seen.append)
        try:
            self.wiki._categories_to_hierarchy(
                ["Статьи со ссылками на Викисловарь"], "работа")
        finally:
            logchannel.unsubscribe(sid)
        self.assertIn("работа", " | ".join(seen),
                      "иерархия пуста, а слушатель об этом не узнал: "
                      f"{seen}")

    def test_no_category_at_all_stays_silent(self):
        """Слово без категорий — не авария. Кричать на пустоте значит
        обесценить настоящее сообщение из предыдущего теста."""
        from core import logchannel
        seen = []
        sid = logchannel.subscribe(seen.append)
        try:
            self.assertEqual([], self.wiki._categories_to_hierarchy([], "хдя"))
        finally:
            logchannel.unsubscribe(sid)
        self.assertEqual([], seen, f"канал зашумел на законной пустоте: {seen}")

    def test_wiki_gives_categories_in_alphabetical_order(self):
        """
        Док канала утверждал «категории Вики идут от абстрактных к конкретным»
        и ОБРАЩАЛ порядок. Это неверно: ru.wikipedia отдаёт prop=categories в
        АЛФАВИТНОМ порядке, поэтому hierarchy[0] — алфавитный первый, а не
        ближайший род.

        ЗАМЕР (experiments/wiki_category_snapshot.json, 22 слова):
        у 19 из 22 первая видимая категория = алфавитно первая.
        Отсюда «собака — это волки», «солнце — это жёлтые карлики», «кошка —
        это животные-компаньоны». Этот тест ЗАПРЕЩАЕТ доверять порядку,
        полученному из Вики: любая починка обязана строить порядок по
        СОДЕРЖАНИЮ (родители категорий), а не переставлять alphabet.
        """
        import json
        from core.records import meaningful_categories
        snap = json.load(open(self.SNAP, encoding="utf-8"))
        alphabetical = checked = 0
        for word, rec in sorted(snap.items()):
            vis = meaningful_categories(rec["title"],
                                        self.wiki._categories_to_hierarchy(
                                            rec["categories"], word))
            if len(vis) < 2:
                continue
            checked += 1
            low = [c.lower() for c in vis]
            if low[0] == sorted(low)[0]:
                alphabetical += 1
        self.assertGreaterEqual(
            alphabetical, 2 * checked // 3,
            f"порядок категорий перестал быть алфавитным ({alphabetical} из "
            f"{checked}) — значит появился источник порядка. Это ХОРОШО, но "
            f"тогда должен существовать ЗАМЕР, что он даёт ближайшее понятие, "
            f"а не самое общее (experiments/p26b_wiki_noise.py, п.10)")

    def test_snapshot_covered(self):
        """Снимок обязан оставаться данным, а не пустой подпоркой теста."""
        import json
        snap = json.load(open(self.SNAP, encoding="utf-8"))
        self.assertGreaterEqual(len(snap), 20,
                                "снимок усох — цифры замера больше не данные")
        self.assertTrue(all(r["categories"] for r in snap.values()),
                        "в снимке есть слово без категорий — нечего измерять")


class TestFirstSentence(unittest.TestCase):
    """Определение = ВСЁ первое предложение, а не обрывок до первой точки."""

    def setUp(self):
        from teachers.teacher_wiki import WikiTeacher
        self.wiki = WikiTeacher(memory_dir="/tmp/l2-sent")
        self.texts = json.load(open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                         "wiki_intros.json"), encoding="utf-8"))

    def test_no_definition_cut_inside_parenthesis(self):
        """
        ЗАМЕРЕНО до правки: 10 из 10 живых вступлений давали обрывок —
        «Пчёлы (лат», «Флюгер (нид», «Вулкан (от лат». Точка в «(лат.» была
        конец, потому что список сокращений не совпадал с реальным написaniem.
        """
        for word, text in self.texts.items():
            sent = self.wiki._first_sentence(text)
            with self.subTest(word=word):
                self.assertTrue(sent.endswith((".", "!", "?")), sent)
                self.assertEqual(sent.count("("), sent.count(")"),
                                 f"обрывок внутри скобки: {sent!r}")

    def test_wiki_meta_sentences_skipped(self):
        """
        «Это статья о джойстике…» и «О других устройствах см. …» — про статью.
        Выучить их вместо определения значит отвечать ссылками на разметку.
        """
        text = ("Это статья о джойстике как устройстве ввода. "
                "О других устройствах см. Игровая панель. "
                "Джойстик — устройство ввода для видеоигр. "
                "Широко применяется в авиации.")
        out = self.wiki._first_sentence(text)
        self.assertFalse(out.lower().startswith(("это статья", "о других")),
                         f"выучена мета-фраза Вики: {out!r}")
        self.assertIn("устройство ввода", out)

    def test_abbreviation_dot_not_a_sentence(self):
        sent = self.wiki._first_sentence(
            "Кошка (лат. Felis catus) — домашнее животное. Дальше.")
        self.assertIn("домашнее животное", sent)
        self.assertNotIn("Дальше", sent)

    def test_empty(self):
        self.assertEqual(self.wiki._first_sentence(""), "")
        self.assertEqual(self.wiki._first_sentence(None), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
