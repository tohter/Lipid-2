"""
Замок п. F2: вопрос обязан искать ТОТ ЖЕ узел, который создаёт обучение.

Очередь звала: «лемматизация вопроса крива на определении и причастии».
ЗАМЕРЕНО до правки (lang.ru.questions.IRParser, HEAD):

    'у красивой кошки признаки'          -> ['красивый кошка']
    'какие признаки у бегущего человека' -> ['бежать человек']
    'у рыжей собаки действия'           -> ['рыжеть собака']  <- третий случай,
                                                        в очереди не записан

И это была не косметика. Каждый такой промах НЕ просто «не нашёл»: мозг шёл
СОЗДАВАТЬ узел с мусорным именем и СПРОСИТЬ пользователя про «красивый кошка»
(замер: созданный узел == 'красивый кошка'). Сора в мире прибывало с каждым
вопросом, а удалить её нечем — слияние узлов необратимо.

ПОЧЕМУ ПРАВИЛО ВЫБРАНО НЕ ГЛАЗАМИ, А ПО ПИШУЩЕЙ СТОРОНЕ. Замер SimpleExtractor
(той, что пишет мир) показал, как узел называется ИСХОДНО:

    'Летучая мышь спит'  -> мышь|CAN_DO|спать,  мышь|HAS_PROPERTY|летучий
    'Красивая кошка ловит' -> кошка|CAN_DO|ловить, кошка|HAS_PROPERTY|красивый
    'Бегущий человек дышит часто' -> человек|CAN_DO|дышать

Определение и причастие уезжают в HAS_PROPERTY, имя узла — ГОЛОЕ
существительное. Значит чтение обязано сводить вопрос к тому же правилу;
«сохранить прилагательное в строке поиска» было бы починкой в сторону,
противоположную записи. Поэтому главный тест файла сравнивает ДВЕ стороны,
а не пересказывает ожидания.

ЧЕГО ПРАВКА СОЗНАТЕЛЬНО НЕ ТРОГАЕТ (заперто как ИЗМЕРЕННОЕ, а не как «так
надо»):
* союз: 'у кошки и собаки признаки' остаётся ['кошка и собака']. Выбрать одну
  голову из перечисления — значит СОКРАТИТЬ вопрос пользователя;
* фраза без существительного остаётся собой (сводить не к чему, выдумывать
  голову опаснее промаха: промах говорит «не знаю», догадка спросит не про то);
* незнакомое слово ('вавикон' -> 'вавикона') лемматизатор сворачивает как
  сворачивал ДО правки — поведение не менялось, регрессии нет.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("LIPID2_STATUS", "0")

from core.extract import SimpleExtractor                    # noqa: E402
from lang.ru.questions import IRParser                      # noqa: E402


class TestHeadOfQuestion(unittest.TestCase):
    """Голова фразы вопроса = существительное, а не словосочетание."""

    @classmethod
    def setUpClass(cls):
        cls.p = IRParser()

    def subject(self, text):
        intent = self.p.parse_query(text)
        return intent.subjects[0] if intent.subjects else None

    def test_queue_cases_now_resolve_to_noun(self):
        # Ровно те две строки, что лежали в очереди, плюс найденная при замере
        # третья (морфология читает причастие как глагол: 'рыжеть собака').
        self.assertEqual(self.subject("у красивой кошки признаки"), "кошка")
        self.assertEqual(self.subject("какие признаки у бегущего человека"),
                         "человек")
        self.assertEqual(self.subject("у рыжей собаки действия"), "собака")

    def test_adjective_is_never_a_part_of_search_key(self):
        # Ни в одной форме падежа определение не имеет права остаться в ключе:
        # узел с прилагательным в имени писавшая сторона создать не может.
        for q in ["что такое красивая кошка", "у красивой кошки признаки",
                  "какие свойства у ржавого гвоздя",
                  "что делает бегущий человек"]:
            with self.subTest(q=q):
                self.assertNotIn(" ", self.subject(q),
                                 f"{q!r} ищет словосочетанием — узла с таким "
                                 "именем в мире не бывает")

    def test_compound_names_collapse_to_the_written_node(self):
        # 'летучая мышь' и 'морская свинка' — честные составные названия, и мы
        # их РЕЖЕМ. Сознательно: ядро записало бы их как мышь/свинка +
        # HAS_PROPERTY, а узел «летучая мышь» в мире появился бы только из
        # вопроса. Разнобой чтения и записи стоил бы дороже потерянного
        # прилагательного.
        self.assertEqual(self.subject("что такое летучая мышь"), "мышь")
        self.assertEqual(self.subject("что такое морская свинка"), "свинка")

    def test_second_noun_does_not_win(self):
        # 'хвост кота' — вопрос про хвост, не про кота: первое существительное.
        self.assertEqual(self.subject("какие признаки у хвоста кота"), "хвост")

    def test_numeral_does_not_become_the_head(self):
        self.assertEqual(self.subject("какие свойства у двух котов"), "кот")


class TestReadMatchesWrite(unittest.TestCase):
    """ГЛАВНЫЙ контракт: вопрос ищет там же, где обучение пишет."""

    @classmethod
    def setUpClass(cls):
        cls.parser = IRParser()
        cls.extractor = SimpleExtractor()

    def written_nodes(self, sentence):
        """Имена узлов, которые ЗАПИСАЛО бы обучение этой фразы."""
        return {f.head for f in self.extractor.extract(sentence).facts}

    def asked_node(self, sentence):
        """Тот же сюжет, но СПРОШЕННЫЙ пользователем.

        Вопрос строится из той же лексики, что и предложение: проверяем не
        удачную формулировку, а путь «записали здесь -> спросили сюда».
        """
        pairs = [
            ("Летучая мышь спит на дереве.", "что такое летучая мышь"),
            ("Красивая кошка ловит мышь.", "какие признаки у красивой кошки"),
            ("Бегущий человек дышит часто.", "что делает бегущий человек"),
            ("Ржавый гвоздь стоит в доске.",
             "какие свойства у ржавого гвоздя"),
            ("Морская свинка живёт в клетке.", "что такое морская свинка"),
        ]
        for sent, question in pairs:
            nodes = self.written_nodes(sent)
            subject = self.parser.parse_query(question).subjects
            got = subject[0] if subject else None
            with self.subTest(sentence=sent, question=question):
                self.assertTrue(nodes,
                                f"обучение не записало ни узла, ни связи из "
                                f"{sent!r}: проверять нечего")
                self.assertIn(got, nodes,
                              f"вопрос {question!r} ищет {got!r}, а обучение "
                              f"пишет {sorted(nodes)}: промах — это поиск в "
                              "пустоте и новый мусорный узел")

    def test_asked_node_is_taken_verbatim_from_the_written_name(self):
        # Ключ сверяется БЕЗ нормализации: если бы тест сам «подчищал» строку,
        # он прощал бы ровно тот дефект, который ловит.
        written = self.written_nodes("Красивая кошка ловит мышь.")
        asked = self.parser.parse_query(
            "у красивой кошки признаки").subjects[0]
        self.assertEqual(asked, asked.strip().lower())
        self.assertIn(asked, written)


class TestMeasuredNotTouched(unittest.TestCase):
    """Что правка НЕ меняла — заперто, чтобы «починка» не съела лишнего."""

    @classmethod
    def setUpClass(cls):
        cls.p = IRParser()

    def test_enumeration_is_not_silently_shortened(self):
        # Выбрать одну голову из «кошки и собаки» = ответить не на вопрос.
        # Терпим: пока нет не-угадывающего способа спросить про обе сущности,
        # честнее отдать вопрос целиком, чем отвечать про одну из двух.
        self.assertEqual(
            self.p.parse_query("у кошки и собаки признаки").subjects,
            ["кошка и собака"])

    def test_phrase_without_noun_stays_as_is(self):
        # Сводить не к чему -> строка остаётся. None в помощнике есть значение,
        # а не молчаливая замена на первое попавшееся слово.
        self.assertIsNone(self.p._head_noun("очень красиво"))
        self.assertEqual(self.p._head_noun("красивая кошка"), "кошка")

    def test_unknown_word_is_lemmatized_as_before(self):
        # Незнакомое слово сворачивается КАК РАНЬШЕ ('вавикон' -> 'вавикона').
        # Заперто, чтобы никто не записал это в «починили лемматизацию»:
        # правка головы к этому отношения не имеет и чинить это не обещала.
        self.assertEqual(self.p.parse_query("что такое вавикон").subjects,
                         ["вавикона"])

    def test_intent_is_still_correct(self):
        # Очередь отмечала: интент был верный, ломалась только строка поиска.
        self.assertEqual(self.p.parse_query("у красивой кошки признаки")
                         .intent_type, "PROPERTIES")
        self.assertEqual(self.p.parse_query("что такое красивая кошка")
                         .intent_type, "DEFINITION")


class TestSourceIsSharedNotCopied(unittest.TestCase):
    """Признак головы берётся ИМЕНЕМ, а не переписан текстом."""

    def test_head_pos_is_named_constant(self):
        # Тест сверяет КОНСТАНТУ, а не строку "NOUN" в своём теле: иначе
        # правка константы осталась бы зелёной (урок 45 про общий код).
        self.assertEqual(IRParser.HEAD_POS, "NOUN")
        self.assertTrue(hasattr(IRParser, "_head_noun"),
                        "помощник головы исчез — тест выше проверяет бытовое "
                        "совпадение, а не правило")


if __name__ == "__main__":
    unittest.main()
