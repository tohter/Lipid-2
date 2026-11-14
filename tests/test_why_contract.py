"""
КОНТРАКТ: «почему» — честный уровень 1 (первый кирпич настоящей думалки).

Откуда. Оба советчика (жозефина/chatgpt и босс/qwen, вслепую, одна формулировка,
13.03) сошлись на первом шаге: пока система не отвечает на «почему» цепочкой
СОБЫТИЙ рассказа, абстрактную мотивацию («предательство») строить не на чем.
Замер живьём: паттерна WHY в lang/ru/questions.py не было, вопрос падал в
UNKNOWN «обработчик в разработке». События в world['events'] несут action/order/
evidence, но поле cause из текста не извлекается.

Форма замка. Здесь держится РАСКЛАДКА ответа (уровень 1), а не текст как красота:

  1. parse_why() достаёт роли (субъект/действие/объект) леммами;
  2. render_why() честно делит: факт / причинная цепочка / «причина не названа»;
  3. обработчик WHY строит цепочку из событий того же агента в том же «моменте»
     (одинаковый order), ничего в базу не пишет;
  4. неверная посылка (агент не тот, события нет) — честный отказ, а не додумка.

Мутации, которые обязан поймать этот файл:
  * паттерн WHY слетел из lang/ru/questions.py -> вопрос уходит в UNKNOWN;
  * parse_why() разбирает не роль-парсером, а шаблонами -> одна форма ломает всё;
  * render_why() забыл честный отказ (found=False) -> выдумка вместо «нет»;
  * _handle_why() строит цепочку через ПРЕДЫДУЩИЙ момент -> «контекст» выдаётся
    за «причину»;
  * _handle_why() берёт событие без сверки агента -> «кошка рассказала секрет»
    станет причиной, хотя это сделала канарейка.
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("LIPID2_STATUS", "0")

from lang.ru.why import parse_why, render_why   # noqa: E402

CAT_CANARY = ("Кошка дружно жила с канарейкой и весело с ней играла. "
              "Кошка разрабатывала искусственный интеллект, и поделилась "
              "секретами разработки с канарейкой, просила никому не "
              "рассказывать. В один прекрасный день кошка видит, как канарейка "
              "сидит у открытой форточки и на всю улицу рассказывает её секреты. "
              "Кошка разозлилась и сожрала канарейку.")


class TestParseWhy(unittest.TestCase):
    """Роль-парсер вопроса о причине: леммы, а не шаблоны."""

    def test_roles_of_eat(self):
        r = parse_why("Почему кошка сожрала канарейку")
        self.assertEqual((r["subject"], r["action"], r["object"]),
                         ("кошка", "сожрать", "канарейка"),
                         "нужно именно роль-разбор, а не совпадение по шаблону")

    def test_roles_of_get_angry(self):
        r = parse_why("почему кошка разозлилась")
        self.assertEqual((r["subject"], r["action"], r["object"]),
                         ("кошка", "разозлиться", ""),
                         "объекта нет — разозлилась на ком/чём не названо")

    def test_roles_of_canary_told(self):
        r = parse_why("почему канарейка рассказала секреты")
        self.assertEqual(r["subject"], "канарейка",
                         "агент события обязан распознаться точно: "
                         "секреты рассказала канарейка, а не кошка")

    def test_non_why_returns_empty(self):
        r = parse_why("что такое кошка")
        self.assertEqual(r["action"], "",
                         "без маркера причины parse_why не имеет права решать "
                         "за другие обработчики")

    def test_why_marks_but_not_late(self):
        r = parse_why("почему")
        self.assertEqual(r["marker"], "почему",
                         "маркер обязан находиться даже без глагола после")


class TestRenderWhy(unittest.TestCase):
    """Честная раскладка: факт / цепочка / «причина не названа» / отказ."""

    def test_refusal_on_wrong_agent(self):
        out = render_why("кошка", "рассказать", "секрет", [], found=False)
        self.assertIn("нет", out,
                      "отказ обязан говорить «нет», а не выдумывать причину")
        self.assertIn("рассказать", out,
                      "глагол вопроса обязан участвовать в пути")

    def test_chain_renders_with_arrow(self):
        out = render_why("кошка", "сожрать", "канарейка",
                         ["разозлиться", "сожрать"],
                         evidence="Кошка разозлилась и сожрала канарейку")
        self.assertIn("разозлиться", out,
                      "цепочка обязана содержать «разозлиться» — первый кирпич")
        self.assertIn("сожрать", out,
                      "глагол вопроса обязан участвовать в пути")
        self.assertIn("→", out,
                      "направление цепочки обязано быть видно")

    def test_cause_when_named(self):
        out = render_why("кошка", "сожрать", "канарейка", [], cause="канарейка "
                         "рассказала секреты")
        self.assertIn("Причина прямо названа", out,
                      "прямо названная причина обязана быть отмечена как факт")


class TestWhyIntentReachesHandler(unittest.TestCase):
    """Сквозной контракт: WHY-интент доходит до обработчика уровня 1."""

    def _fresh_brain(self):
        from brain.lipid2_brain import Lipid2Brain
        tmp = tempfile.mkdtemp(prefix="lip_why_")
        b = Lipid2Brain(memory_dir=tmp)
        b.use_wiki_teacher = False
        b.use_llm_teacher = False
        return b, tmp

    def test_eat_chain(self):
        b, _ = self._fresh_brain()
        b.learn_from_text(CAT_CANARY, source="cat", allow_teacher=False)
        out = b.process_input("почему кошка сожрала канарейку")
        self.assertIn("разозлиться", out,
                      "уровень 1 обязан дать цепочку с «разозлиться»")
        self.assertIn("сожрать", out,
                      "глагол вопроса обязан участвовать в пути")
        self.assertIn("→", out,
                      "направление причинной цепочки обязано быть видно")

    def test_get_angry_honest_not_named(self):
        b, _ = self._fresh_brain()
        b.learn_from_text(CAT_CANARY, source="cat", allow_teacher=False)
        out = b.process_input("почему кошка разозлилась")
        self.assertIn("причина в тексте не названа", out,
                      "причины в событиях нет — честный «не названа», "
                      "а не додумка про «разрабатывать -> поделиться»")

    def test_canary_told_not_in_events(self):
        b, _ = self._fresh_brain()
        b.learn_from_text(CAT_CANARY, source="cat", allow_teacher=False)
        out = b.process_input("почему канарейка рассказала секреты")
        self.assertIn("нет", out,
                      "события «рассказать» с агентом «канарейка» в событиях "
                      "нет — отказ честнее додумки")

    def test_cat_told_wrong_agent(self):
        b, _ = self._fresh_brain()
        b.learn_from_text(CAT_CANARY, source="cat", allow_teacher=False)
        out = b.process_input("почему кошка рассказала секрет")
        self.assertIn("нет", out,
                      "кошка в рассказе секреты не рассказывала — "
                      "неверная посылка обязана быть поймана")

    def test_why_does_not_write_to_db(self):
        b, tmp = self._fresh_brain()
        b.learn_from_text(CAT_CANARY, source="cat", allow_teacher=False)
        before = sorted(b._load_world().keys())
        b.process_input("почему кошка сожрала канарейку")
        after = sorted(b._load_world().keys())
        self.assertEqual(after, before,
                         "уровень 1 ничего в базу не пишет: только чтение")


if __name__ == "__main__":
    unittest.main()