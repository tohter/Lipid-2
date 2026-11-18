"""Attachment contracts from the expert task; fixtures are data, not metric exceptions."""
import unittest
from core.extract import SimpleExtractor


class TestPrepositionAttachment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = SimpleExtractor()

    def test_both_heads(self):
        cases = [
            ('Дом с крышей.', 'дом', 'WITH', 'крыша'),
            ('Старик с удочкой сидел у реки.', 'старик', 'WITH', 'удочка'),
            ('Чай с лимоном остыл.', 'чай', 'WITH', 'лимон'),
            ('Девочка с косичками смеётся.', 'девочка', 'WITH', 'косичка'),
            ('Кот гуляет с собакой.', 'гулять', 'WITH', 'собака'),
            ('Книга о городе лежит на столе.', 'книга', 'ABOUT', 'город'),
            ('Кот говорит о собаке.', 'говорить', 'ABOUT', 'собака'),
            ('Книга об озере лежит на столе.', 'книга', 'ABOUT', 'озеро'),
            ('Рассказ про город лежит на столе.', 'рассказ', 'ABOUT', 'город'),
            ('Кот говорит про собаку.', 'говорить', 'ABOUT', 'собака'),
            ('Письмо к брату лежит на столе.', 'письмо', 'RECIPIENT', 'брат'),
            ('Письмо к сестре лежит на столе.', 'письмо', 'RECIPIENT', 'сестра'),
            ('Кот идёт к брату.', 'идти', 'RECIPIENT', 'брат'),
            ('Звонок по телефону разбудил мать.', 'звонок', 'PATH', 'телефон'),
            ('Занятие по расписанию началось.', 'занятие', 'PATH', 'расписание'),
            ('Кот идёт по дороге.', 'идти', 'PATH', 'дорога'),
            ('Кот спит в кресле.', 'спать', 'LOCATED_IN', 'кресло'),
            ('Мышь забралась в нору.', 'забраться', 'LOCATED_IN', 'нора'),
        ]
        for sentence, head, relation, tail in cases:
            with self.subTest(sentence=sentence):
                result = self.extractor.extract(sentence)
                actual = [(f.head, f.relation, f.tail) for f in result.facts
                          if f.relation == relation and f.tail == tail]
                self.assertEqual(actual, [(head, relation, tail)])

    def test_ambiguous_object_attachment_refuses(self):
        sentence = 'Мать читает письмо к брату.'
        result = self.extractor.extract(sentence)
        actual = [f for f in result.facts if f.relation == 'RECIPIENT']
        self.assertEqual(actual, [])
        dependent = sentence.rstrip('.').split()[-1].lower()
        self.assertTrue(any(dependent in note for note in result.notes))
