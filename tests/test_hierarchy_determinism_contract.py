"""
КОНТРАКТ (п.A-остаток3): ПОРОЖДЕНИЕ ИЕРАРХИИ НЕ ЗАВИСИТ ОТ ПОРЯДКА КАНАЛОВ.

ЗАМЕchadoЙ ДЕФЕКТ. merge_entity складывает `hierarchy` списком «сначала то,
что пришло первым», а из `hierarchy[0]` строится ОДНО ребро IS_A в графе ядра
(brain/lipid2_brain.py::_integrate_entity_into_core). Значит ПОРЯДОК, В
КОТОРОМ каналы успели ответить на одно и то же слово, выбирает РОДИТЕЛЯ узла
в ядре — то есть то, чем система считает предмет.

Замер (реальные записи двух каналов, experiments/*_snapshot.json):

    Вики-запись: hierarchy = ['ветроуказатели', 'экстерьер', 'объекты']
    ЛОКАЛКА:     hierarchy = ['метеорологические приборы', 'инструменты', 'объекты']

    Вики->локалка : файл[0] = 'ветроуказатели'          -> IS_A -> 'ветроуказатели'
    локалка->Вики : файл[0] = 'метеорологические приборы' -> IS_A -> '...приборы'

    ОДНА И ТА ЖЕ СУЩНОСТЬ, РАЗНЫЙ РОДИТЕЛЬ В ЯДРЕ.

ПОЧЕМУ РАНЖИРОВАНИЕ ПО СОДЕРЖАНИЮ (а не «первому пришедшему») — три замера,
те же снимки, из которых нельзя выбрать удобное:

  1. Иерархия ЛОКАЛКИ ранжирована: из 18 слов с цепочкой от двух категорий
     15 заканчиваются верхним корнем «*объект*» — то внутри неё есть
     направление «от ближнего к дальнему».
  2. Категории Вики АЛФАВИТНЫ: из 22 слов со списком от двух категорий 14
     строго алфавитны после отсева служебных; в редакционный род первая
     категория Вики не попала НИ РАЗУ (0/7), локалка — 2/7. Порядок Вики —
     позиция в словаре, а не близость.
  3. Ответ пользователя — прямое называние рода («Пчела — летаемое
     насекомое»), уверенность канала 1.0 (CHANNEL_CONFIDENCE), выше него
     мнения нет.

ПОЭТОМУ: порядок внутри иерархии задаёт КАНАЛ по рангу
(user > llm > text > wiki), а не очередь прихода ответов. При равных рангах
остаётся прежнее правило «ранее записанное главнее» — оно стабильно во
времени (запись дополняется, а не переставляется).

Что для этого ПОНАДОБИЛОСЬ: канал записи должен быть ИЗВЕСТЕН. Замер:
локальный Учитель не писал `provenance` ВООБЩЕ (ни одного места в
teachers/teacher1.py), тогда как Вики пишет «wiki», пользователь — «user»,
урок — имя источника. Без метки ранжировать не по чему, и правка была бы
маговством. Учителю метка добавлена (source="llm").
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.records import merge_entity

WIKI = {"concept": "флюгер", "kind": "thing",
        "definition": "прибор для указания направления ветра",
        "hierarchy": ["ветроуказатели", "экстерьер", "объекты"],
        "provenance": [{"source": "wiki", "evidence": "Флюгер", "at": None}]}
LLM = {"concept": "флюгер", "kind": "thing",
       "hierarchy": ["метеорологические приборы", "инструменты", "объекты"],
       "provenance": [{"source": "llm", "evidence": "разбор", "at": None}]}
USER = {"concept": "флюгер", "hierarchy": ["прибор указателя ветра"],
        "provenance": [{"source": "user", "evidence": "это прибор", "at": None}]}


def cp(rec):
    return json.loads(json.dumps(rec))


class TestChannelRankNotCallOrder(unittest.TestCase):
    """Один и тот же набор фактов, записанный в любом порядке, даёт один [0]."""

    def test_llm_before_wiki_whichever_came_first(self):
        """
        Регресс п.A-остаток3. Иерархия локалки ранжирована (15/18 цепочек
        кончаются верхним корнем), порядок Вики алфавитен (14/22) и в род не
        попал ни разу (0/7) — значит ближним должен считаться локалочный
        HEAD, независимо от того, какой канал ответил первым.
        """
        a = merge_entity(cp(LLM), cp(WIKI))["hierarchy"]
        b = merge_entity(cp(WIKI), cp(LLM))["hierarchy"]
        self.assertEqual(a[0], "метеорологические приборы",
                         f"родом стал алфавитный хвост Вики: {a}")
        self.assertEqual(b[0], "метеорологические приборы",
                         f"[0] зависит от порядка вызова каналов: {b[0]!r} != {a[0]!r}")
        self.assertEqual(a, b, "состав/порядок иерархии зависит от порядка каналов")
        # и ничего не потеряно:
        for cat in WIKI["hierarchy"] + LLM["hierarchy"]:
            self.assertIn(cat, a, f"категория {cat} пропала при слиянии")

    def test_user_beats_every_channel(self):
        """
        Пользователь называет род словами — это единственное мнение с
        уверенностью 1.0. Ранг выше локалки: «Пчела — летаемое насекомое»
        не имеет права встать вторым после догадки модели.
        """
        a = merge_entity(cp(LLM), cp(USER))["hierarchy"]
        b = merge_entity(cp(USER), cp(LLM))["hierarchy"]
        self.assertEqual(a[0], "прибор указателя ветра")
        self.assertEqual(b[0], "прибор указателя ветра")

    def test_same_channel_keeps_earlier_record(self):
        """
        Равные ранги — прежнее правило: главное ранее записанное. Иначе
        повторный ответ модели переставлял бы родителя узла при каждом
        вопросе, и знание «плыло» бы между сеансами.
        """
        first = cp(LLM)
        second = cp(LLM)
        second["hierarchy"] = ["указатели ветра", "объекты"]
        merged = merge_entity(first, second)["hierarchy"]
        self.assertEqual(merged[0], "метеорологические приборы")

    def test_records_without_channel_marker_keep_old_behavior(self):
        """
        Совместимость: записей без provenance в мире много (иерархии из
        текста, старые файлы, самописные тесты). Без метки канала ранжировать
        нечем — обязан работать прежний порядок «prev первым», иначе эта
        правка молча переставит половину существующей Вселенной.
        """
        plain_a = {"concept": "ф", "hierarchy": ["приборы"]}
        plain_b = {"concept": "ф", "hierarchy": ["предметы"]}
        self.assertEqual(merge_entity(cp(plain_a), cp(plain_b))["hierarchy"][0],
                         "приборы")
        self.assertEqual(merge_entity(cp(plain_b), cp(plain_a))["hierarchy"][0],
                         "предметы")


class TestTeacherMarksItsChannel(unittest.TestCase):
    """
    Без метки канала ранжирование невозможно. Замер: teachers/teacher1.py
    НЕ писал provenance ни в одном месте, хотя teacher_wiki пишет «wiki»,
    а пользователь — «user».
    """

    def _write(self, content):
        from teachers.teacher1 import AgentTeacherV2
        d = tempfile.mkdtemp(prefix="l2a3_")
        try:
            t = AgentTeacherV2(memory_dir=d)
            res = t.write_json_file(content=json.dumps(content,
                                                       ensure_ascii=False))
            world = json.load(open(os.path.join(d, "lipid_world.json"),
                                   encoding="utf-8"))
            return res, world
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_llm_write_stamps_provenance(self):
        res, world = self._write({"base_entity": {
            "concept": "флюгер", "hierarchy": ["приборы", "объекты"]},
            "derived_entity": {}})
        self.assertFalse(str(res).startswith("[Tool Error]"), res)
        ent = world["entities"]["флюгер"]
        srcs = [p.get("source") for p in ent.get("provenance") or []]
        self.assertIn("llm", srcs,
                      f"запись Учителя без метки канала: provenance={srcs} — "
                      "её иерархию нельзя поставить выше алфавитной Вики")

    def test_stamp_survives_merge_with_wiki(self):
        """
        Метка обязана доживать до слияния: если merge_entity оставит в
        слитой записи только Вики-происхождение, следующий канал уже не
        узнает, чей порядок главенствует.
        """
        res, world = self._write({"base_entity": {
            "concept": "флюгер",
            "hierarchy": ["метеорологические приборы", "объекты"]},
            "derived_entity": {}})
        ent = world["entities"]["флюгер"]
        merged = merge_entity(ent, cp(WIKI))
        self.assertEqual(merged["hierarchy"][0], "метеорологические приборы",
                         "метка канала есть, но слияние её игнорирует")
        srcs = [p.get("source") for p in merged.get("provenance") or []]
        self.assertEqual(srcs, sorted(set(srcs), key=srcs.index),
                         "происхождение при слиянии перемешалось")


class TestParentEdgeInCore(unittest.TestCase):
    """
    ДОКАЗАТЕЛЬСТВО, что дефект доходил до ядра, а не жил в файле.
    Иначе правка была бы косметикой: «пусть в JSON порядок плывёт».
    """

    def _world(self, record):
        d = tempfile.mkdtemp(prefix="l2a3g_")
        ent = {"concept": "флюгер", "kind": "thing", "definition": "",
               "properties": [], "actions": [], **record}
        with open(os.path.join(d, "lipid_world.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"entities": {"флюгер": ent}, "relations": [],
                       "observations": {}, "unknown_words": {},
                       "transitions": {}}, fh, ensure_ascii=False)
        return d

    def test_is_a_edge_follows_channel_rank_not_file_order(self):
        """
        СЦЕНА, А НЕ ВЫДУМКА: один и тот же флюгер, оба канала ответили,
        файл собран merge_entity в двух порядках. Родитель узла в ядре
        обязан быть ОДИН — иначе порядок сети выбрал онтологию.
        """
        from core.graph import RelationType
        from brain.lipid2_brain import Lipid2Brain
        parents = []
        for a, b in ((cp(LLM), cp(WIKI)), (cp(WIKI), cp(LLM))):
            merged = merge_entity(a, b)
            d = self._world(merged)
            try:
                br = Lipid2Brain(memory_dir=d)
                node = br.core.nodes["флюгер"]
                parents += [t for t, v in node.out_synapses_typed.items()
                            if v["rel_type"] == RelationType.IS_A]
            finally:
                shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(parents, ["метеорологические приборы"] * 2,
                         f"родитель в ядре зависит от того, какой канал "
                         f"записался первым: {parents}")


if __name__ == "__main__":
    unittest.main()
