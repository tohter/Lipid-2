"""Remaining prepositions: named refusals preserve facts about the main clause."""
import unittest
from core.extract import SimpleExtractor


class TestRemainingPrepositions(unittest.TestCase):
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
