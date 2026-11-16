# -*- coding: utf-8 -*-
"""
ДЫРА 5 «ПРОБЕЛ УЧИТЕЛЯ»: определения учителя не идут через extract_text.

Запись учителя умеет три структурированных поля (hierarchy -> IS_A,
properties -> HAS_PROPERTY, actions -> CAN_DO) и одно текстовое — definition
(«Невестка — жена сына»). Дыра в двух местах:

1) Запись, у которой есть ТОЛЬКО definition (а hierarchy/properties пусты),
   отбрасывалась как «degenerate» — знание терялось целиком. Самый больной
   случай: родственные определения («жена сына») не имеют иерархии.
2) Даже когда definition присутствовало, оно не разбиралось в рёбра графа:
   «невестка — жена сына» не давало рёбер, думалка могла только процитировать
   строку, но не рассуждать.

Закрытие — по вердикту Жозефины + решениям Хозяина 0.1/0.3:

- Определение проходит через тот же extract_text.
- Принимаются ТОЛЬКО ОПРЕДЕЛЯЮЩИЕ связи: IS_A/IS_ROLE (голова = концепт)
  и RELATION_TO (родственные отношения). Шум отбрасывается: OBJECT_OF,
  LOCATED_IN, ABOUT, PART_OF, HAS_PROPERTY-из-определения — не входят, иначе
  «экосистема ... на поверхности Земли» дало бы экосистема OBJECT_OF земля.
- HAS_PROPERTY/IS_A приходят структурированно от учителя (properties,
  hierarchy), определение их НЕ дублирует.
"""
import json
import os
import shutil
import tempfile
import unittest


class TestTeacherDefinitionContract(unittest.TestCase):
    """Красный контракт дыры 5: определения учителя разбираются в граф."""

    def setUp(self):
        from brain.learner import LearningReport, Learner
        from core.graph import GraphMemory
        self.tmp = tempfile.mkdtemp(prefix="l2h5_")
        self.path = os.path.join(self.tmp, "w.json")
        self._write({"entities": {}, "observations": {}, "unknown_words": {}})
        self.core = GraphMemory()
        self.lr = Learner(load_world=self._read, save_world=self._write, core=self.core)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _write(self, w):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(w, f, ensure_ascii=False)

    def _store(self, lemma, data, source="t"):
        from brain.learner import LearningReport
        rep = LearningReport()
        self.lr._store_entity({}, lemma, data, source, evidence=data.get("definition", ""),
                              rep=rep)
        return rep

    def _edges(self, node_name):
        n = self.core.nodes.get(node_name)
        if n is None:
            return []
        return [(e["rel_type"].name if hasattr(e["rel_type"], "name") else e["rel_type"],
                 e["target"]) for e in n.out_synapses_typed.raw_edges()]

    # ── 1. Запись ТОЛЬКО с definition не отбрасывается ──────────────────
    def test_definition_only_record_is_not_degenerate(self):
        """Запись учителя с одним definition проходит usable и создаёт сущность."""
        def fake_teacher(lemma):
            if lemma == "невестка":
                return {"concept": "невестка", "kind": "entity",
                        "definition": "Невестка — жена сына.",
                        "hierarchy": [], "properties": [], "actions": []}
            return None
        self.lr.teacher = fake_teacher
        rep = self.lr.learn_from_text("Невестка пришла домой.", source="t")
        # entities_created считает и глаголы-действия; проверяем СУЩЕСТВЕННОЕ —
        # что запись только с definition НЕ отброшена, а узел создан.
        self.assertIn("невестка", self.core.nodes,
                      "узел невестка не создан: запись только с definition "
                      "отброшена как degenerate")

    # ── 2. Определяющие связи попадают в граф ───────────────────────────
    def test_relation_edges_from_definition(self):
        """«невестка — жена сына» даёт рёбра IS_ROLE и RELATION_TO."""
        self._store("невестка", {
            "concept": "невестка", "kind": "entity",
            "definition": "Невестка — жена сына.",
            "hierarchy": [], "properties": [], "actions": [],
        })
        self.assertIn(("IS_ROLE", "жена"), self._edges("невестка"),
                      "нет ребра невестка IS_ROLE жена из определения")
        self.assertIn(("RELATION_TO", "сын"), self._edges("жена"),
                      "нет ребра жена RELATION_TO сын из определения")

    def test_role_edge_from_definition(self):
        """«слон — животное с хоботом» даёт слон IS_ROLE животное."""
        self._store("слон", {
            "concept": "слон", "kind": "entity",
            "definition": "Слон — это животное с хоботом.",
            "hierarchy": ["млекопитающие"], "properties": [], "actions": [],
        })
        self.assertIn(("IS_ROLE", "животное"), self._edges("слон"),
                      "нет ребра слон IS_ROLE животное из определения")

    # ── 3. Шум из определения НЕ попадает в граф ────────────────────────
    def test_no_noise_from_definition(self):
        """«экосистема — ... на поверхности Земли» НЕ даёт OBJECT_OF земля."""
        self._store("экосистема", {
            "concept": "экосистема", "kind": "entity",
            "definition": "Экосистема — это основная природная единица на поверхности Земли.",
            "hierarchy": [], "properties": [], "actions": [],
        })
        edges = self._edges("экосистема")
        self.assertNotIn(("OBJECT_OF", "земля"), edges,
                         "шум «экосистема OBJECT_OF земля» попал из определения")
        self.assertNotIn(("LOCATED_IN", "поверхность"), edges,
                         "шум «экосистема LOCATED_IN поверхность» попал")

    def test_no_object_of_for_genus(self):
        """«кошка — домашнее животное» НЕ даёт кошка OBJECT_OF животное."""
        self._store("кошка", {
            "concept": "кошка", "kind": "entity",
            "definition": "Кошка — это домашнее животное.",
            "hierarchy": ["животные"], "properties": ["домашнее"], "actions": [],
        })
        edges = self._edges("кошка")
        self.assertNotIn(("OBJECT_OF", "животное"), edges,
                         "экстрактор дал «кошка OBJECT_OF животное», шум")
        self.assertNotIn(("HAS_PROPERTY", "домашний"), edges,
                         "HAS_PROPERTY из определения дублирует properties учителя")

    def test_latin_parenthesis_makes_no_noise(self):
        """Скобка «(лат. Felis catus)» не создаёт ребро OBJECT_OF лат."""
        self._store("кошка", {
            "concept": "кошка", "kind": "entity",
            "definition": "Кошка (лат. Felis catus) — домашнее животное.",
            "hierarchy": ["животные"], "properties": [], "actions": [],
        })
        edges = self._edges("кошка")
        self.assertNotIn(("OBJECT_OF", "лат"), edges,
                         "латинская скобка превратилась в узел 'лат'")


if __name__ == "__main__":
    unittest.main(verbosity=2)