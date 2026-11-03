"""
КОНТРАКТ П1b: СКАЛЯР ПРОИСХОЖДЕНИЯ ОБЯЗАН ОТРАЖАТЬ ПОСЛЕДНЕЕ ДОКАЗАТЕЛЬСТВО.

Вопрос, на который отвечает поле: «это я прочитала или мне рассказали». П1a
замкнула канал ДО РЕБРА ядра; здесь речь идёт о скаляре на записи сущности,
который читают пять мест кода и НЕ читает ни один тест (замер grep:
читателей 5 — brain/lipid2_brain.py:277, :796, :866, brain/learner.py:687,
:1243 — тестов 0).

ЗАМЕРЕНО ДО ПРАВКИ (запись на живом коде, не по памяти):

1) merge_entity ОСТАВЛЯЕТ СКАЛЯР ОТ ПРЕДЫДУЩЕГО ПИСАТЕЛЯ. Правило слияния
   «новое добавляет, но не стирает» распространяется и на learned_from, а
   provenance при этом накапливается хвостом:
       prev learned_from='llm' prov=['llm']
       new  learned_from='user' prov=['user']
       ----> learned_from='llm', provenance=['llm','user'], ПОСЛЕДНЕЕ='user'
   Человек ответил поверх ответа модели — а поле говорит «модель». Это не
   умолчание, это УСТАРЕВШЕЕ значение: оно было верным и перестало.

2) МЕТОД ГЛУХ К ДОКАЗАТЕЛЬСТВУ. brain/lipid2_brain.py:866:
       raw_src = str(data.get("source") or data.get("learned_from") or "llm")
   provenance в расчёт НЕ берётся. Достижимый замер: запись с
   provenance=[wiki] и без скаляра даёт канал 'teacher:llm'. То есть
   знание, пришедшее из Википедии или от человека, приписывается локальной
   модели, хотя единственное достоверное место (список доказательств) рядом.

ЗАЧЕМ ЭТОМУ ОТДЕЛЬНЫЙ ФАЙЛ, а не приложение к П1a. П1a — про ДОСТАВКУ известного
канала. Здесь канала нет или он устарел: чинить надо ТОЧКУ ЗАПИСИ скаляра, а
не пять точек чтения (решение босса: пять правок в читателях при нуле тестов —
это пять шансов поменять поведение незаметно).

ИНВАРИАНТ (требование босса): learned_from == source ПОСЛЕДНЕГО элемента
provenance. Доказательств нет — скаляр пуст, а не «text».

Граница файла: ЧЕСТНОСТЬ метки происхождения. Онтологию не трогаем. Отдельно
отмечаю и НЕ чиню здесь: неизвестное ИМЯ канала («wattpad») сейчас молча
превращается в llm — это другой вопрос, он не про «устарел», а про «не знаем»,
и правка его шире одной точки.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.records import merge_entity            # noqa: E402

# merge_entity сводит формы слов (алиасы, иерархия) и без службы лемм
# молча работает эвристикой — то есть тест проверял бы не язык, а догадку
# морфологии. Ставим службу ЯВНО: иначе результат файла зависел бы от
# порядка импортов в общем прогоне.
from lang.ru.lemma import install as _install_lemma_service   # noqa: E402

_install_lemma_service()


def last_src(rec):
    prov = rec.get("provenance") or []
    return (str(prov[-1].get("source") or "") if prov else None)


class TestScalarMatchesLastEvidence(unittest.TestCase):
    """Инвариант: скаляр = последнее доказательство."""

    def test_human_answer_over_model_answer(self):
        """ЗАМЕРЕННЫЙ СЛУЧАЙ: человек ответил поверх ответа модели.

        До правки learned_from оставался 'llm'. Знание о том, что слово
        объяснил человек (самый надёжный канал), пропадало из поля, которое
        читают веса и порядок слияния иерархии.
        """
        prev = {"concept": "невестка", "learned_from": "llm",
                "definition": "жена сына",
                "provenance": [{"source": "llm", "evidence": "ответ модели",
                                "at": 1}]}
        new = {"concept": "невестка", "learned_from": "user",
               "provenance": [{"source": "user", "evidence": "объяснил человек",
                               "at": 2}]}
        merged = merge_entity(prev, new)
        self.assertEqual(merged.get("learned_from"), "user",
                         "скаляр отражает ПРЕДЫДУЩЕГО писателя: поле "
                         "«откуда это» устарело молча")
        self.assertEqual(merged.get("learned_from"), last_src(merged),
                         "скаляр разошёлся с последним доказательством")

    def test_three_writers_in_a_row(self):
        """Инвариант держится на цепочке, а не на одном слиянии."""
        rec = {"concept": "флюгер", "learned_from": "llm",
               "provenance": [{"source": "llm", "evidence": "а", "at": 1}]}
        for src, at in (("wiki", 2), ("text", 3), ("user", 4)):
            rec = merge_entity(rec, {"concept": "флюгер", "learned_from": src,
                                     "provenance": [{"source": src,
                                                     "evidence": src,
                                                     "at": at}]})
            self.assertEqual(rec.get("learned_from"), last_src(rec),
                             f"после канала {src} скаляр отстал")
        self.assertEqual(rec.get("learned_from"), "user")

    def test_evidence_list_keeps_history(self):
        """Скаляр честный — НЕ значит «историю выбросили». Провенанс обязан
        остаться накопленным: скаляр отвечает «кто последний», список — «кто
        вообще»."""""
        prev = {"concept": "сноха", "learned_from": "llm",
                "provenance": [{"source": "llm", "evidence": "а", "at": 1}]}
        merged = merge_entity(prev, {"concept": "сноха", "learned_from": "wiki",
                                     "provenance": [{"source": "wiki",
                                                     "evidence": "b",
                                                     "at": 2}]})
        srcs = [p.get("source") for p in merged.get("provenance") or []]
        self.assertEqual(srcs, ["llm", "wiki"],
                         "починка скаляра не имеет права выбросить историю")
        self.assertEqual(merged.get("learned_from"), "wiki")


class TestNothingInvented(unittest.TestCase):
    """Нет доказательств — нет и скаляра. Ни «text», ни «llm» по умолчанию."""

    def test_empty_provenance_leaves_scalar_empty(self):
        merged = merge_entity({}, {"concept": "роза",
                                   "definition": "цветок",
                                   "provenance": []})
        self.assertFalse(merged.get("learned_from"),
                         f"скаляр выдуман при пустых доказательствах: "
                         f"{merged.get('learned_from')!r}")

    def test_known_scalar_not_erased_when_new_has_no_evidence(self):
        """Защита данных: НЕ обнулять то, что уже известно, только потому,
        что у нового доказательства пустые provenance. Иначе правка скаляра
        стёрла бы происхождение у старых записей мира (в файле таких много).
        """
        prev = {"concept": "кошка", "learned_from": "text"}
        merged = merge_entity(prev, {"concept": "кошка", "properties": ["рыжая"],
                                     "provenance": []})
        self.assertEqual(merged.get("learned_from"), "text",
                         "известный канал стёрт из-за пустоты нового — это "
                         "потеря данных, а не честность")


class TestChannelNotInventedForUnknownSource(unittest.TestCase):
    """Мозг не имеет права называть канал там, где метки нет, и обязан
    посмотреть в доказательство, прежде чем выдумывать.

    ЗАМЕРЕНО достижимым входом, а не воображаемым. Первый вариант этого
    теста был написан на записи БЕЗ source, БЕЗ learned_from и БЕЗ
    provenance — и это недостижимо: base_entity приходит из _read_back
    (teachers/teacher1.py), то есть из уже записанной сущности мира, а
    запись мира всегда несёт хотя бы provenance. Тест на недостижимом
    входе — это зелёный замок на несуществующем поведении.

    Живой замер (brain/lipid2_brain.py:866 берёт ТОЛЬКО source или
    learned_from и глух к provenance):
        provenance=[wiki], скаляра нет -> канал 'teacher:llm'
        provenance=[user], скаляра нет -> канал 'teacher:llm'
        provenance=[text], скаляра нет -> канал 'teacher:llm'
    То есть запись, пришедшая из Википедии или от человека, приписывается
    локальной модели — потому что метод не посмотрел в единственное место,
    где происхождение заведомо есть.
    """

    def setUp(self):
        from brain.lipid2_brain import Lipid2Brain
        self.brain = Lipid2Brain(memory_dir="/tmp/p1b_contract")

    def test_channel_read_from_evidence_when_scalar_absent(self):
        for mark in ("wiki", "user", "text"):
            got = self.brain._channel_provenance(
                {"concept": "флюгер",
                 "provenance": [{"source": mark, "evidence": "дока", "at": 1}]})[2]
            self.assertEqual(got, f"teacher:{mark}",
                             f"доказательство говорит {mark!r}, а прибор "
                             f"рапортует {got!r}")

    def test_truly_unmarked_record_gets_no_channel(self):
        conf, evidence, src = self.brain._channel_provenance(
            {"concept": "флюгер", "definition": "прибор"})
        self.assertFalse(src,
                         f"каналу без единой метки приписан {src!r}: ложная "
                         f"атрибуция дешевле честного «не знаю»")

    def test_marked_channels_still_normalize(self):
        """Честность НЕ отменяет нормализацию: каналы, которые названы,
        обязаны узнаваться как раньше (иначе правка сломает вес и порядок).
        """
        for mark, want in (("text", "teacher:text"), ("wiki", "teacher:wiki"),
                           ("wikipedia", "teacher:wiki"),
                           ("user", "teacher:user"),
                           ("teacher:wiki", "teacher:wiki")):
            got = self.brain._channel_provenance({"source": mark})[2]
            self.assertEqual(got, want, f"метка {mark} перестала узнаваться")

    def test_scalar_beats_evidence_when_they_disagree(self):
        """Скаляр — указание писателя, доказательство — свидетель. При споре
        верим указанию: именно скаляр ставит вызывающий, знающий свой канал.
        Заперто, чтобы правка не устроила тихую перетяжку приоритетов.
        """
        got = self.brain._channel_provenance(
            {"learned_from": "user",
             "provenance": [{"source": "wiki", "evidence": "", "at": 1}]})[2]
        self.assertEqual(got, "teacher:user")


if __name__ == "__main__":
    unittest.main(verbosity=2)
