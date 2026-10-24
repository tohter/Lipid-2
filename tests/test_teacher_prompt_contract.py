"""
КОНТРАКТ ПРОМПТА ЛОКАЛЬНОГО УЧИТЕЛЯ (п.A-остаток2).

Очередь утверждала: «Учитель не соблюдает контракт hierarchy[0] — 29%».
Замер эту цифру и постановку НЕ подтвердил. Три факта, проверенные по
снимку experiments/llm_hierarchy_snapshot.json и живому замеру 25.10:

  1. Ранжирование УЧИТЕЛЯ работает. Из 17 слов с неразложимыми категориями
     16 заканчиваются корнем «объекты» — цепочка от ближнего к дальнему есть
     (у Вики-канала порядок АЛФАВИТНЫЙ, 19 из 22 — см.
     test_experience_choice_contract; у локалки алфавита нет).
  2. Первая позиция НЕ совпадает с редакционным родом: 2 из 7 по прежнему
     замеру, 4 из 11 (36%) по свежему живому замеру на тех 11 словах, где
     есть честный эталон. Но ОТВЕТ это не отравляет: при нескольких
     категориях _definition_answer перечисляет ВСЕ и строку «X — это Y»
     не производит — проверено ниже.
  3. Ветка «пустое definition + ровно одна категория» — единственный вид
     записи, где позиция [0] диктует слова ответа, — для локального канала
     НЕДОСТУЖИМА: в снимке 20 слов таких 0.

Значит «чинить промпт ради 29%» нельзя: правка в пустоту запрещена. Чинится
то, что нашлось ПОД дорогой очереди — дефект 2:

  Шаг 3 алгоритма («ПРОВЕРЬ ИНВАРИАНТЫ») проверял инварианты ПРИМЕРА
  («идёт ли иерархия от 'кошачьи' к 'животные'?», «есть ли 'самец' у
  кота?») на ЛЮБОМ концепте. На запросе «флюгер» или «Земля» такая
  самопроверка заведомо ложна, а шаг 4 велит при нарушении ПЕРЕПИСАТЬ:
  буквально следующая модель обязана стирать правильную иерархию и
  ставить кошачью. Живой замер 25.10: сегодняшняя Gemma мину обходит
  (кошачьих иерархий в ответах нет) — дефект латентный, правка бесплатна.

ЧЕСТНЫЙ ИТОГ ПРАВКИ (4 прогона живой Gemma по 11 словам, 2 на базу и 2 на
правки, /tmp/a2/{before,before2,after,after2}.json):

      попадание hierarchy[0] в редакционный род: 4/11 — БАЗА прогон 1 и 2,
      ПРАВКА прогон 1 и 2. То есть правка НЕ подняла качество иерархии.

Остальные метрики снять «до/после» НЕЛЬЗЯ, и это тоже результат: средняя
длина цепочки у ОДНОЙ И ТОЙ ЖЕ базой версии скачет 4.4 -> 7.7 между прогонами,
«цепочка кончается верхним корнем» — 7/11 -> 3/11. Разброс модели больше
любого эффекта правки, поэтому заявлять «хвосты выросли» или «корень портился»
по одной паре прогонов — подтасовка. Метрика «попадание в род» устойчива,
она и записана; она и есть ответ очереди: контракт иерархии локалка держит,
род не угадывает, и промптом это не лечится.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SNAPSHOT = os.path.join(ROOT, "experiments", "llm_hierarchy_snapshot.json")


class TestPromptInvariant(unittest.TestCase):
    """Текст system_prompt: контракт есть, ложной самопроверки нет."""

    def _prompt(self):
        """
        ТОТ текст, который реально уходит в модель, а не исходник функции:
        в исходнике стоят комментарии разработчика, модель их не читает, и
        проверка по исходнику путает «упоминание в комментарии» с
        «требованием в промпте». Транспорт подменён, сеть не трогается.
        """
        import urllib.request
        from teachers import teacher1
        captured = {}

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"}}]}).encode()

        def fake_urlopen(req, timeout=None):
            captured.setdefault("payload", json.loads(req.data.decode()))
            return Resp()

        real = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        d = tempfile.mkdtemp(prefix="l2prompt_")
        try:
            teacher1.AgentTeacherV2(memory_dir=d).run_agent(
                base_concept="флюгер", derived_concept="флюгер")
        finally:
            urllib.request.urlopen = real
            shutil.rmtree(d, ignore_errors=True)
        payload = captured.get("payload")
        if payload is None:
            self.skipTest("агент не сформировал запрос — промпт не измерен")
        return payload["messages"][0]["content"]

    def test_genus_invariant_is_stated(self):
        """Контракт hierarchy[0]=ближайшее семейство обязан быть в тексте."""
        prompt = self._prompt()
        self.assertIn("hierarchy[0]", prompt)
        self.assertIn("БЛИЖАЙШЕЕ семейство/класс", prompt)

    def test_root_invariant_is_stated(self):
        prompt = self._prompt()
        self.assertIn("hierarchy[-1]", prompt)
        self.assertIn("ВЕРХНИЙ АБСТРАКТНЫЙ КОРЕНЬ", prompt)

    def test_selfcheck_is_not_example_specific(self):
        """
        Шаг-самопроверка не вправе требовать инвариантов ПРИМЕРА на чужих
        концептах. Регресс: «Идет ли иерархия от 'кошачьи' к 'животные'?».
        'кошачьи' имеет право жить в примере структуры и в примере порядка,
        но не в шаге, который выполняется для любого слова.
        """
        prompt = self._prompt()
        start = prompt.find("АЛГОРИТМ РАБОТЫ")
        self.assertGreater(start, 0, "шаг алгоритма не найден в тексте промпта")
        algo = prompt[start:]
        self.assertNotIn(
            "'кошачьи'", algo,
            "самопроверка требует инварианты конкретного примера — на любом "
            "чужом концепте она ложна и провоцирует модель перезаписать всё")
        self.assertNotIn("'самец'", algo)
        self.assertIn("hierarchy[0]", algo,
                      "самопроверка обязана проверять род, а не примеры")


class TestSnapshotOffline(unittest.TestCase):
    """Замер по снимку: ветки, на которые опирается вывод очереди."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(SNAPSHOT):
            raise unittest.SkipTest(f"нет снимка {SNAPSHOT}")
        with open(SNAPSHOT, encoding="utf-8") as fh:
            cls.snap = json.load(fh)

    def test_single_category_branch_unreachable_for_llm(self):
        """
        Иерархия без definition и ровно из одной НЕ-САМОСЫЛКИ — единственный
        вид записи, где _definition_answer выдаёт «X — это <первая>». Для
        локального канала таких записей в снимке 0 из 20: позиция в списке
        на слова ответа не влияет. Если снимок обновится и такие слова
        появятся — тест обязан покраснеть: ветка ожила и требует нового
        ЗАМЕРА, а не молчаливой правки промпта.
        """
        from core.records import meaningful_categories
        singles = {w: r.get("hierarchy") for w, r in self.snap.items()
                   if not str(r.get("definition") or "").strip()
                   and len(meaningful_categories(w, r.get("hierarchy") or [])) == 1}
        self.assertEqual(singles, {},
                         f"у локалки появились слова с одной категорией {singles} — "
                         "ветка «X — это hierarchy[0]» для неё ожила")

    def test_genus_miss_does_not_poison_answer(self):
        """
        Ответ с несколькими категориями НЕ ВЫБИРАЕТ hierarchy[0] — иначе
        36%-ное попадание в род прямо перетекло бы в слова ответа. Это то,
        что защищает вывод: промпт ради 36% чинить нельзя.
        """
        from brain.lipid2_brain import Lipid2Brain
        from core.records import meaningful_categories
        word = next(w for w, r in self.snap.items()
                    if len(meaningful_categories(w, r.get("hierarchy") or [])) >= 2)
        h = meaningful_categories(word, self.snap[word]["hierarchy"])
        d = tempfile.mkdtemp(prefix="l2a2_")
        try:
            with open(os.path.join(d, "lipid_world.json"), "w", encoding="utf-8") as fh:
                json.dump({"entities": {word: {
                    "concept": word, "kind": "thing", "definition": "",
                    "hierarchy": list(h), "properties": [], "actions": [],
                    "provenance": [{"source": "llm", "note": "замер п.A-остаток2"}]}},
                    "relations": [], "observations": {}, "unknown_words": {},
                    "transitions": {}}, fh, ensure_ascii=False)
            b = Lipid2Brain(memory_dir=d)
            b.use_llm_teacher = b.use_wiki_teacher = b.ask_user_enabled = False
            b.learn_experience = False
            text = b.process_input(f"что такое {word}")
            self.assertIn("Определения", text,
                          "ответ с несколькими категориями обязан называть себя "
                          "списком, иначе позиция [0] снова выдаётся за род")
            self.assertNotIn(f"— это {h[0]}.", text)
            for cat in h:
                self.assertIn(cat, text, f"категория {cat} пропала из ответа")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
