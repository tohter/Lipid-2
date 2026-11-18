"""Remaining prepositions: named refusals preserve facts about the main clause."""
import unittest
from core.extract import SimpleExtractor


class TestRemainingPrepositions(unittest.TestCase):
    def test_genitive_source_or_time(self):
        cases = [('Письмо от брата.', 'письмо', 'PATH', 'брат'),
                 ('Кот вышел из дома.', 'выйти', 'PATH', 'дом'),
                 ('Кот вышел изо рва.', 'выйти', 'PATH', 'ров'),
                 ('Туман держится от рассвета.', 'держаться', 'CIRCUMSTANCE', 'рассвет')]
        for sentence, head, relation, tail in cases:
            with self.subTest(sentence=sentence):
                facts = SimpleExtractor().extract(sentence).facts
                self.assertEqual([(f.head, f.relation, f.tail) for f in facts
                                  if f.tail == tail], [(head, relation, tail)])

    def test_untyped_ambiguous_pair_refuses(self):
        cases = [('Мальчик выглянул из-за угла.', 'угол'),
                 ('Пожар гаснет из-за дождя.', 'дождь'),
                 ('Пожар из-за искры.', 'искра')]
        for sentence, dependent in cases:
            with self.subTest(sentence=sentence):
                result = SimpleExtractor().extract(sentence)
                self.assertFalse(any(dependent in (f.head, f.tail) for f in result.facts))
                word = sentence.rstrip('.').split()[-1]
                self.assertTrue(any(n.startswith(word + ':') for n in result.notes))

    def test_through_time_or_route(self):
        extractor = SimpleExtractor()
        cases = [('Поезд едет через час.', 'ехать', 'час', 'CIRCUMSTANCE'),
                 ('Поезд едет через мост.', 'ехать', 'мост', None)]
        for sentence, head, tail, relation in cases:
            with self.subTest(sentence=sentence):
                result = extractor.extract(sentence)
                self.assertEqual([(f.head, f.relation, f.tail) for f in result.facts
                                  if f.tail == tail], [(head, relation, tail)] if relation else [])
                if not relation:
                    self.assertTrue(any(n.startswith(tail + ':') for n in result.notes))
                    extractor.set_entity_kind(place={tail})
                    result = extractor.extract(sentence)
                    self.assertEqual([(f.head, f.relation, f.tail) for f in result.facts
                                      if f.tail == tail], [(head, 'PATH', tail)])

    def test_temporal_boundaries(self):
        cases = [('Дождь шёл до вечера.', 'идти', 'вечер'),
                 ('Кот спит перед рассветом.', 'спать', 'рассвет'),
                 ('Кот спит после заката.', 'спать', 'закат')]
        for sentence, head, tail in cases:
            with self.subTest(sentence=sentence):
                result = SimpleExtractor().extract(sentence)
                self.assertEqual([(f.head, f.relation, f.tail) for f in result.facts
                                  if f.tail == tail], [(head, 'CIRCUMSTANCE', tail)])

    def test_boundary_requires_evidence(self):
        cases = [('Кот спит перед домом.', 'спать', 'дом', 'LOCATED_IN'),
                 ('Турист дошёл до реки.', 'дойти', 'река', 'PATH'),
                 ('Кот спит перед сном.', '', 'сон', ''),
                 ('Рабочие ужинают после работы.', '', 'работа', '')]
        for sentence, head, tail, relation in cases:
            with self.subTest(sentence=sentence):
                extractor = SimpleExtractor()
                result = extractor.extract(sentence)
                self.assertFalse(any(f.tail == tail for f in result.facts))
                word = sentence.rstrip('.').split()[-1]
                self.assertTrue(any(n.startswith(word + ':') for n in result.notes))
                if relation:
                    extractor.set_entity_kind(place={tail})
                    result = extractor.extract(sentence)
                    self.assertEqual([(f.head, f.relation, f.tail) for f in result.facts
                                      if f.tail == tail], [(head, relation, tail)])

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
