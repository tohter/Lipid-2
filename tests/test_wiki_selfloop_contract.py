"""ДВЕ ДЫРЫ КАНАЛА ВИКИ, ИЗМЕРЕННЫЕ 08.11 В ВЕЧЕР.

Замер ставил вопрос: можно ли сравнивать локальную модель с Википедией
(«а Вики подтвердит?»). Оказалось, что сравнивать нечего — канал оставляет в
онтологии мусор и теряет половину слов ещё до разбора. Указание босса: сначала
чинить это, ПОТОМ вводить ground, ПОТОМ сравнивать каналы.

─────────────────────────────────────────────────────────────────────────────
ДЫРА 1. КАТЕГОРИИ ВИКИ ПРОИЗВОДЯТ СВЯЗЬ СУЩНОСТИ С САМОЙ СОБОЙ.
В русской Википедии статья почти всегда состоит в категории со своим именем.
`_categories_to_hierarchy` клал категорию в `hierarchy`, а `StaticKnowledge`
читает `hierarchy` как IS_A. ЗАМЕРЕНО живьём (5 статей, experiments/
wiki_channel_probe.py и прямой прогон `_fetch_page`):

    огонь  -> IS_A «огонь»                 (ребро огонь -> огонь)
    солнце -> IS_A «солнце»
    вода   -> IS_A «вода»

Такое ребро вырождено (X IS_A X), всегда ложно как знание и вредно структурно:
в обходе графа это петля. Механически оно опознаётся ТОЧНО — лемма категории
совпадает с леммой концепта, семантика не нужна.

ЦЕНА, ПРИНЯТАЯ ОСОЗНАННО. У слова «стол» категория одна — «столы» (лемма
«стол»). После отсечения иерархия становится пустой, и ответ пойдёт по опре-
делению. Это ПРАВИЛЬНЫЙ проигрыш: «стол — это столы» есть ровно тот класс
подмены, на который уже жаловался код («собака — это волки», «солнце — это
жёлтые карлики»). Молчание при этом НЕ должно стать тихим — пустая иерархия
обязана быть объявленной, см. test_self_loop_loss_is_announced.

─────────────────────────────────────────────────────────────────────────────
ДЫРА 2. ПОСТАНОВЫЕ УДАРЕНИЯ ЛОМАЮТ РАЗБОР — И ЭТО БЫЛА МОЯ ОШИБКА, А НЕ
РУССКОЙ ВИКИПЕДИИ.
Статьи идут с U+0301 над гласными. Снятие стояло ТОЛЬКО в `_first_sentence`
(учителя/teacher_wiki.py), то есть у того, кто помнил про грабли. Любой дру-
гой читатель `*_fetch_page*` получал сырой текст. Я наступила на это дважды:
первый зонд Вики дал «экстрактор слеп, 0 из 2», и я собиралась доложить, что
канал не проверен. ЗАМЕРЕНО на живой статье «Курица» (5 ударных знаков):

    с ударениями:  «рица» --HAS_PROPERTY--> «шний»    (слово изувечено)
    без ударений:  «курица» --HAS_PROPERTY--> «домашний»

Чинится в одном месте: `_fetch_page` отдаёт текст БЕЗ постановых знаков, и
каждый следующий читатель Вики не обязан знать историю. Снятие из
`_first_sentence` убирается как дубль: два места снятия — это два места, где
одно из них забудут.

ЗАПРЕТ МУТАЦИИ. Вернуть самоцикл, снять фильтр, вернуть ударения или оставить
их снятие только в `_first_sentence` — все четыре мутации обязаны краснить эти
тесты. Тест офлайновый: сеть не трогается, `_fetch_page` получает подставной
ответ API.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lang.ru.lemma import install                  # noqa: E402
install()

from teachers import teacher_wiki                  # noqa: E402
from teachers.teacher_wiki import WikiTeacher      # noqa: E402

STRESS = "́"     # U+0301, постановое ударение


class TestSelfLoopEdge(unittest.TestCase):
    """Ребро «X IS_A X» не имеет права рождаться из категории-тёзки."""

    def setUp(self):
        self.room = tempfile.mkdtemp(prefix="wiki_contract_")
        self.t = WikiTeacher(memory_dir=self.room)

    def test_no_self_category(self):
        """Категория с именем самой сущности отбрасывается, остальные — нет.

        Ловит замеренные «огонь IS_A огонь», «солнце IS_A солнце»,
        «вода IS_A вода». Годные категории обязаны выжить: отсечь всё было бы
        так же плохо, как оставить петлю (потеря 11 настоящих родов).
        """
        got = self.t._categories_to_hierarchy(
            ["Огонь", "Огнеопасные вещества", "Пламя"], "огонь")
        self.assertNotIn("огонь", [g.lower() for g in got],
                         f"вернулась петля: {got}")
        self.assertIn("огнеопасные вещества", [g.lower() for g in got])
        self.assertIn("пламя", [g.lower() for g in got])

    def test_self_category_in_any_form_and_case(self):
        """Множественное число и падеж — та же петля: «столы» для «стол».

        Именно этот класс давал подмену вида «собака — это волки»: форма др.
        числа того же слова не является родителем.
        """
        self.assertEqual([], self.t._categories_to_hierarchy(["Столы"], "стол"))
        self.assertEqual([], self.t._categories_to_hierarchy(["Киты"], "кит"))

    def test_self_loop_loss_is_announced(self):
        """Пустая после фильтра иерархия обязана быть ОБЪЯВЛЕНА, а не молча пуста.

        Отсечь петлю «столы» для «стол» значит оставить слово без иерархии.
        Молчание здесь неотличимо от «категорий не было», а неотличимая дыра
        не чинится никогда.
        """
        said = []
        self.t._say = lambda msg, **kw: said.append(msg)
        self.assertEqual([], self.t._categories_to_hierarchy(["Столы"], "стол"))
        self.assertTrue(any("разметк" in s or "иерархия осталась пустой" in s
                            for s in said),
                        f"молчание вместо объявления: {said}")

    def test_other_words_categories_survive(self):
        """Фильтр не режет чужие категории: петля про СЕБЯ, а не про слова."""
        got = [g.lower() for g in self.t._categories_to_hierarchy(
            ["Домашние куры", "Сельскохозяйственные птицы"], "курица")]
        self.assertEqual(["домашние куры", "сельскохозяйственные птицы"], got)


class TestStressMarks(unittest.TestCase):
    """Сырые постановые ударения не должны доходить ни до одного читателя."""

    def _patch_api(self, title, extract, categories=()):
        """Подставляет ответ REST API Вики (офлайновый прогон `_fetch_page`)."""
        payload = {"query": {"pages": {"1": {
            "pageid": 1, "title": title, "extract": extract,
            "categories": [{"title": "Категория:" + c} for c in categories]}}}}
        body = json.dumps(payload).encode("utf-8")

        class _Resp:
            def read(self_inner):
                return body

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        def fake(_req, timeout=None):
            return _Resp()

        original = teacher_wiki.urllib.request.urlopen
        teacher_wiki.urllib.request.urlopen = fake
        self.addCleanup(setattr, teacher_wiki.urllib.request, "urlopen",
                        original)

    def test_fetch_page_strips_stress(self):
        """ЗАМЕРЕНО: живая статья «Курица» содержит 5 постановых ударений.

        `_fetch_page` обязан вернуть текст без U+0301 — иначе любой следующий
        читатель (не только `_first_sentence`) получит изувеченные слова.
        """
        self._patch_api("Ку́рица",
                        "Ку́рица или дома́шняя ку́рица — вид птицы.")
        title, extract, _cats = WikiTeacher(memory_dir=".")._fetch_page("курица")
        self.assertNotIn(STRESS, extract, "ударения дошли до читателя")
        self.assertNotIn(STRESS, title)
        self.assertIn("курица", extract)
        self.assertIn("домашняя", extract)

    def test_reader_sees_whole_words(self):
        """Тот же текст, что ломал разбор: слова обязаны читаться целыми.

        Это ровно мой замер с первым зондом: с U+0301 экстрактор выдавал
        «рица --HAS_PROPERTY--> шний» и я записала бы это как слепоту Вики.
        """
        from core.extract import SimpleExtractor
        self._patch_api("Курица",
                        "Ку́рица — домашняя птица, вид рода банки.")
        _title, extract, _c = WikiTeacher(memory_dir=".")._fetch_page("курица")
        facts = SimpleExtractor().extract(extract).facts
        heads = {f.head for f in facts}
        self.assertTrue(heads, "разбор не дал ничего")
        for f in facts:
            self.assertNotIn("рица", heads | {f.tail},
                             f"слово изувечено: {f.head} -> {f.tail}")

    def test_first_sentence_still_splits_clean_text(self):
        """Снятие ударений из `_first_sentence` не должно сломать разбиение.

        Убираю дубль — и обязана остаться проверка, что определение всё ещё
        берётся ЦЕЛИКОМ (точка внутри «лат.» не обрывает фразу).
        """
        t = WikiTeacher(memory_dir=".")
        got = t._first_sentence("Курица — птица (лат. Gallus gallus). "
                                "Второе предложение.")
        self.assertTrue(got.startswith("Курица"), got)
        self.assertNotIn("Второе предложение", got)


if __name__ == "__main__":
    unittest.main()
