"""
УЧИТЕЛЬ ОБЯЗАН УМЕТЬ СКАЗАТЬ «КТО КОМУ ПРИХОДИТСЯ», а не только «КТО ЧЕМУ».

ЗАЧЕМ ЭТОТ ФАЙЛ. Хозяин поставил цель словами (30.01):

  «нам важно внести в базу данных, что невестка — это жена сына, а не
   млекопитающее-эукариот. Надо, чтобы учитель как-то умел соображать, в
   каких случаях нужна классификация, а в каких нет. И видимо запрос к модели
  надо отрегулировать».

ЗАМЕР, ИЗ-ЗА КОТОРОГО ПИШЕТСЯ ЭТОТ ЗАМОК (одна и та же модель, gemma-4-12B-
it-Q8_0 @8080, 30.01):

  свободный вопрос «кто такая невестка?»        33.5 с -> «Невестка — это жена
      сына» (+ синоним «сноха»). ВЕРНО.
  НАШ канал run_agent (prod-путь)             20.1/22.6 с -> properties:
      ["жена брата", "женщина", "член семьи"], иерархия
      ["родственники по браку","родственники","люди","существа","объекты"],
      СЛОВО «сын» В ОТВЕТЕ ОТСУТСТВУЕТ ВОВСЕ.
  наш же канал + reasoning_effort=low          100.7 с -> ни «сын», ни «жена»,
      ни «брат»: реляционного знания не стало вовсе (правка настроенки не
      лечит, а стоит x5 времени).

Значит модель ЗНАЕТ ответ, а наш канал его НЕ ВЫНИМАЕТ. Причина по дереву:
`get_tools_schema()` описывает поле content одной фразой («JSON-строка с
сущностями base_entity и derived_entity»), а пример в system_prompt перечисляет
только concept/hierarchy/properties/actions. СЛОТА ДЛЯ ОТНОШЕНИЯ В ЗАПРОСЕ НЕТ.
Модель отвечает в размеченных полях — и «жена сына» сваливается в properties
списком прилагательных, где «жена брата» выглядит столь же уместно. Приёмник
при этом есть: write_json_file пишет dict(ent) как есть, а brain читает
definition (brain/lipid2_brain.py:276, :877, :1362). Передатчика нет, приёмник
есть — это дыра канала, а не глупость модели.

ЧЕГО ЭТОТ ТЕСТ НЕ УТВЕРЖДАЕТ. Он НЕ заявляет, что правка промпта поднимет
качество ответов: прецедент есть — test_teacher_prompt_contract честно пишет,
что прежняя правка промпта НЕ подняла попадание в род (4/11 и до, и после).
Поэтому здесь заперт ТОЛЬКО текст запроса (его можно измерить без сети), а
качество ответа меряет живой прогон tools/ablation_teacher.py, и его числа
обязаны быть записаны в его собственный docstring, а не в веру.

ЗАМОК ОДНОСТОРОЙНИЙ НАМЕРЕННО. Дешёвый способ позеленить — вписать в промпт
готовый ответ «невестка = жена сына». Это ровно то, что хозяин запрещал:
«придётся программировать для каждого конкретного текста». Поэтому проверки
смотрят на СТРУКТУРУ запроса (есть ли слот и правило выбора типа знания) и
ни на одно конкретное слово из нашей загадки.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestTeacherAsksForRelation(unittest.TestCase):
    """Запрос обязан спрашивать отношение, а не только класс и признаки."""

    def _payload(self):
        """ТОТ запрос, что уходит в модель: system + описание инструментов.

        Подмена транспорта, а не исходника: проверка по исходнику путает
        «упоминание в комментарии разработчика» с «требованием в промпте»
        (тот же приём, что в test_teacher_prompt_contract).
        """
        import urllib.request
        from teachers import teacher1
        captured = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"choices": [{"finish_reason": "stop",
                               "message": {"role": "assistant",
                                          "content": "ok"}}]}).encode()

        def fake_urlopen(req, timeout=None):
            captured.setdefault("payload", json.loads(req.data.decode()))
            return Resp()

        real = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        d = tempfile.mkdtemp(prefix="l2rel_")
        try:
            teacher1.AgentTeacherV2(memory_dir=d).run_agent(
                base_concept="невестка", derived_concept="невестка")
        finally:
            urllib.request.urlopen = real
            shutil.rmtree(d, ignore_errors=True)
        payload = captured.get("payload")
        if payload is None:
            self.skipTest("агент не сформировал запрос — измерять нечего")
        return payload

    def test_prompt_has_slot_for_relation(self):
        """В тексте запроса обязан быть слот определения через отношение.

        Слово `definition` — не украшение: без него модель некуда кладёт
        «жена сына», и знание уходит в properties, где оно необратимо
        теряет вторую часть связи (замер: «сын» в ответе канала отсутствует).
        """
        p = self._payload()
        sys_txt = p["messages"][0]["content"]
        schema = json.dumps(p.get("tools", []), ensure_ascii=False)
        blob = (sys_txt + " " + schema).lower()
        self.assertIn("definition", blob,
                      "в запросе нет слота, куда модель положила бы, чем "
                      "концепт является по отношению к другому — знание о "
                      "родстве некуда записать, и оно тонет в properties")

    def test_prompt_names_the_relation_participants(self):
        """Одного «definition» мало: нужен слот «к кому относится».

        Определение «жена сына» — это связь ДУХ участников. Если модель
        обязана назвать участников отдельным списком, фраза остаётся
        структурой, а не строкой, из которой потом нельзя вытащить «сын».
        """
        p = self._payload()
        blob = (p["messages"][0]["content"] + " "
                + json.dumps(p.get("tools", []), ensure_ascii=False)).lower()
        self.assertTrue(
            any(k in blob for k in ("relates_to", "related_to", "references")),
            "нет поля для участников отношения: «жена сына» вернётся текстом "
            "свойства, и вывести из него связь будет нечем")

    def test_prompt_tells_when_classification_is_useless(self):
        """Правило выбора ТИПА знания, а не запрет классификации.

        Хозяин: «чтобы учитель умел соображать, в каких случаях нужна
        классификация, а в каких нет». Для слов родства верхние звенья
        («люди», «существа», «объекты») ответа не дают: на «Кто такая
        невестка?» мир и без учителя отвечает «относится к: родственники,
        люди, существа, объекты». Проверка НЕ требует убрать hierarchy —
        она требует, чтобы промпт различал, где главное класс, а где связь.
        """
        sys_txt = self._payload()["messages"][0]["content"].lower()
        marks = ("родств", "роль", "отношени", "приходится", "связь")
        self.assertTrue(
            sum(m in sys_txt for m in marks) >= 2,
            "в промпте нет правила, когда главное — класс, а когда — связь "
            "(замер ответа мира: «невестка относится к: родственники, люди, "
            "существа, объекты» — класс есть, знания нет)")

    def test_hierarchy_is_not_forbidden(self):
        """ОБОРОТНАЯ СТОРОНА ЗАМКА. Правка не имеет права выкинуть иерархию:
        «дождь IS_A вода» и транзитивный вывод по IS_A заперты другими
        тестами (test_role_category_contract, test_schemes_contract)."""
        sys_txt = self._payload()["messages"][0]["content"]
        self.assertIn("hierarchy", sys_txt,
                      "правка убрала контракт иерархии — сломается законное "
                      "категориальное наследование")
        self.assertIn("hierarchy[0]", sys_txt)


class TestSchemaDescriptionIsNotEmpty(unittest.TestCase):
    """Описание инструмента обязано описывать СХЕМУ, а не только название."""

    def test_content_description_lists_fields(self):
        """Раньше description поля content был «JSON-строка с сущностями
        base_entity и derived_entity»: имена полей модель не знала и отвечала по
        своему вкусу, а приёмник (write_json_file) молча получал то,
        что модель посчитала нужным."""
        from teachers.teacher1 import AgentTeacherV2
        import tempfile as _tf
        d = _tf.mkdtemp(prefix="l2schema_")
        try:
            tools = AgentTeacherV2(memory_dir=d).get_tools_schema()
        finally:
            shutil.rmtree(d, ignore_errors=True)
        write = [t for t in tools
                 if t["function"]["name"] == "write_json_file"][0]
        desc = write["function"]["parameters"]["properties"]["content"]["description"]
        self.assertGreater(len(desc), 60,
                           "описание content слишком короткое, чтобы задавать "
                           "форму ответа")
        for field in ("hierarchy", "properties", "definition"):
            self.assertIn(field, desc,
                          f"поле {field} не описано в схеме инструмента — "
                          "модель не знает, что его заполнять")


if __name__ == "__main__":
    unittest.main(verbosity=2)
