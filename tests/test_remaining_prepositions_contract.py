"""Remaining prepositions: named refusals preserve facts about the main clause."""
import unittest
from core.extract import SimpleExtractor


class TestRemainingPrepositions(unittest.TestCase):
    def test_for_recipient(self):
        cases = [('Подарок для сына.', 'подарок', 'сын'),
                 ('Письмо для сестры.', 'письмо', 'сестра'),
                 ('Кот принёс подарок для брата.', 'принести', 'брат')]
        for sentence, head, tail in cases:
            with self.subTest(sentence=sentence):
                result = SimpleExtractor().extract(sentence)
                actual = [(f.head, f.relation, f.tail) for f in result.facts
                          if f.tail == tail]
                self.assertEqual(actual, [(head, 'RECIPIENT', tail)])
        for sentence in ['Подарок для дома.', 'Лекарство для сна.']:
            result = SimpleExtractor().extract(sentence)
            self.assertEqual(result.facts, [])
            self.assertTrue(result.notes)

    def test_without_refuses(self):
        extractor = SimpleExtractor()
        cases = [('Чай без сахара.', 'сахар'),
                 ('Кот спит без одеяла.', 'одеяло'),
                 ('Дом без крыши.', 'крыша')]
        for sentence, dependent in cases:
            with self.subTest(sentence=sentence):
                result = extractor.extract(sentence)
                self.assertFalse(any(f.tail == dependent for f in result.facts))
                self.assertTrue(result.notes)
