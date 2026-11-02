"""
КОНТРАКТ П2: ОБРАТНЫЙ ЗАПРОС. «КТО ССЫЛАЕТСЯ НА ЭТО СЛОВО».

ЗАЧЕМ ЭТО ВООБЩЕ НУЖНО. Загадка хозяина: «Кем друг другу приходятся
невестка Зинаиды и жена Егора?» — «это одно лицо». Кирпичи для ответа в
мире ЕСТЬ (замер experiments/P2_BEFORE.txt, учителя выключены, LLM не зван):

    невестка IS_ROLE жена      жена PART_OF сын
    егор     IS_ROLE сын       сын  PART_OF зинаида

Пути при этом НЕТ. Причина одна: связи читаются только ОТ головы.
StaticKnowledge строит _by_head: {голова: [(отношение, хвост)]} и умеет
спросить «куда ссылается X». Спросить «КТО ссылается на X» нечем — метода
нет ни в StaticKnowledge, ни в GraphMemory (core/graph.py, 327 строк:
единственный обход — print_node_associations, а это ПЕЧАТЬ). Узел хранит
только out_synapses_typed; обратного индекса в мире не существует.

ЗАМЕР, КОТОРЫЙ ОПРЕДЕЛИЛ ПРАВКУ (тот же мир, тот же обход, отличие одно —
читается ли ребро в обе стороны):

    невестка -> егор   только вперёд:  НЕТ ПУТИ
    невестка -> егор   в обе стороны:  невестка -IS_ROLE-> жена
                                       -PART_OF-> сын <-IS_ROLE- егор

Значит дыра НЕ в знании, а в приборе. Это различение стоило отдельного
замера: без него правку выбрали бы наугад — «доучить слова» или «починить
разбор», и промахнулись бы.

ОБРАТНЫЙ ХОД ОПАСЕН, И ЦЕНА ИЗМЕРЕНА ДО ПРАВКИ. Мир = определения +
рассказ (20 рёбер). Сколько пар «сущность ⟷ сущность» рождает одно лишь
правило «сошлись в одном узле»:
    все отношения               45 пар
    IS_ROLE + PART_OF + IS_A    15 пар
    только PART_OF               3 пары
Настоящих тождеств среди них единицы, остальные ЛОЖЬ:
    деверь ⟷ шурин   через «брат»  (оба братья — но не одно лицо)
    егор   ⟷ зинаида через «сын»   (мать и сын)
    брат   ⟷ сноха   через «жена»
Поэтому этот файл запирает ДВА требования сразу, а не одно «путь нашёлся»:

    (А) обратный запрос существует и отдаёт РОВНО РЁБРА, ничего не выдумывая;
    (Б) «сошлись в узле» НЕ является и не становится тождеством.

(Б) — не украшение. Механизм, который «находит общий узел» и объявляет
встречу идентичностью, собрал бы все тесты на «путь найден» и одновременно
залил мир ложными родствами. Ложная связь хуже пропущенной (правило
проекта), значит запрет обязан появиться РАНЬШЕ починки, а не после.

ЗАМЕРКА СОБСТВЕННОГО ТЕКСТА. Эту докстрингу проверяет
tests/test_static_hygiene.py: она краснеет на любом чужом письме и на
латинице, врезанной в русское слово, — включая цитаты самих дефектов.
Отсутствие находки у выбранного инструмента НЕ есть чистота (урок 70).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LIPID2_STATUS", "0")

from brain.simulator import StaticKnowledge  # noqa: E402


def world_of(edges, negated=()):
    """Мир из кортежей (head, relation, tail) в формате lipid_world.json.
    negated — те же кортежи, но помеченные отрицанием: их обязан НЕ видеть
    ни прямой, ни обратный запрос."""
    obs = {}
    for i, (h, r, t) in enumerate(edges):
        obs[f"k{i}"] = {"head": h, "relation": r, "tail": t}
    for i, (h, r, t) in enumerate(negated):
        obs[f"n{i}"] = {"head": h, "relation": r, "tail": t, "negated": True}
    return {"observations": obs, "entities": {}}


# ОПРЕДЕЛЕНИЯ ИЗ ЗАГАДКИ — ровно те, что дал рассказ, без подсказки ответа.
# «Жена Егора — это невестка Зинаиды» сюда НЕ включена: первая проба подсунула
# именно её, путь построился мгновенно, и меряла бы я свой текст, а не систему.
RIDDLE = [
    ("невестка", "IS_ROLE", "жена"),
    ("жена", "PART_OF", "сын"),
    ("егор", "IS_ROLE", "сын"),
    ("сын", "PART_OF", "зинаида"),
]

# ЛОЖНЫЕ ВСТРЕЧИ из того же замера. Право на жизнь у них есть: это настоящие
# рёбра. Красным должно быть любое утверждение, что их концы — одно лицо.
FALSE_MEETS = [
    ("деверь", "IS_ROLE", "брат"),
    ("шурин", "IS_ROLE", "брат"),
]


class TestReverseQueryMissing(unittest.TestCase):
    """КРАСНОЕ ЯДРО. Умеет ли прибор ответить «КТО называет это слово своим
    определением/хозяином». Сейчас — нет, и тесты это фиксируют вызовом."""

    def test_can_ask_who_points_at_a_node(self):
        sk = StaticKnowledge(world_of(RIDDLE))
        # «егор IS_ROLE сын» значит: у слова «сын» есть тот, кто на него
        # ссылается. Спросить об этом нечем — метода нет вовсе.
        self.assertTrue(
            hasattr(sk, "referenced_by"),
            "у StaticKnowledge нет ни одного способа узнать, кто ссылается "
            "на узел: обход только вперёд, цепочка загадки нерешаема")

    def test_reverse_step_of_the_riddle_is_answerable(self):
        sk = StaticKnowledge(world_of(RIDDLE))
        fn = sk.referenced_by
        got = {h for h, r in fn("сын", "IS_ROLE")}
        self.assertIn("егор", got,
                      f"обратный шаг «кто сын?» не отдаёт Егора: {got}")

    def test_relation_type_is_respected_backwards(self):
        """Обратный запрос обязан различать ОТНОШЕНИЕ, а не всякую ссылку.
        «жена PART_OF сын» и «егор IS_ROLE сын» — разные причины встретиться,
        смешать их = потерять смысл связи."""
        sk = StaticKnowledge(world_of(RIDDLE))
        fn = sk.referenced_by
        role = {h for h, r in fn("сын", "IS_ROLE")}
        part = {h for h, r in fn("сын", "PART_OF")}
        self.assertEqual(role, {"егор"}, f"в ролях кто попало: {role}")
        self.assertEqual(part, {"жена"}, f"в принадлежности кто попало: {part}")


class TestReverseMirrorsForward(unittest.TestCase):
    """СВОЙСТВО, А НЕ СПИСОК. Прямой и обратный запрос обязаны видеть ОДНИ И
    ТЕ ЖЕ рёбра — иначе починка вводит второй источник правды, и расхождение
    между ними ничем не измерить."""

    def _all(self, sk):
        fwd = set()
        for rel in ("IS_A", "IS_ROLE", "PART_OF", "HAS_PROPERTY", "OBJECT_OF",
                    "CAN_DO", "CAUSES", "LOCATED_IN"):
            for t in sk._tails("невестка", rel):
                fwd.add(("невестка", rel, t))
            for t in sk._tails("жена", rel):
                fwd.add(("жена", rel, t))
            for t in sk._tails("сын", rel):
                fwd.add(("сын", rel, t))
        return fwd

    def test_every_forward_edge_is_readable_backwards(self):
        sk = StaticKnowledge(world_of(RIDDLE))
        fn = sk.referenced_by
        back = set()
        for node in ("невестка", "жена", "сын", "егор", "зинаида"):
            for rel in ("IS_A", "IS_ROLE", "PART_OF", "HAS_PROPERTY",
                        "OBJECT_OF", "CAN_DO", "CAUSES", "LOCATED_IN"):
                for h, r in fn(node, rel):
                    back.add((h, r, node))
        fwd = self._all(sk)
        self.assertEqual(fwd - back, set(),
                         f"прямой запрос видит рёбра, которых нет у обратного: "
                         f"{fwd - back}")

    def test_reverse_invents_nothing(self):
        """Обратный ход не имеет права порождать рёбра. Проверка в одну
        сторону: всё, что вернул обратный запрос, обязано стоять в мире."""
        world = world_of(RIDDLE)
        real = {(o["head"], o["relation"], o["tail"])
                for o in world["observations"].values()}
        sk = StaticKnowledge(world)
        fn = sk.referenced_by
        invented = []
        for node in ("невестка", "жена", "сын", "егор", "зинаида", "чужак"):
            for rel in ("IS_A", "IS_ROLE", "PART_OF", "OBJECT_OF", "CAN_DO"):
                try:
                    hits = fn(node, rel)
                except KeyError:
                    continue          # отказ по незнакомому узлу — это не связь
                for h, r in hits:
                    if (h, r, node) not in real:
                        invented.append((h, r, node))
        self.assertEqual(invented, [],
                         f"обратный запрос сочинил связи: {invented}")


class TestRefusalIsNotSilent(unittest.TestCase):
    """«НИКТО» и «НЕТ ТАКОГО СЛОВА» — РАЗНЫЕ ОТВЕТЫ. Молча вернуть пустоту
    туда, где узла вообще нет, значит однажды спутать «у невестки нет хозяина»
    с «невестки не существует» (правило: отказ обязан быть явным)."""

    def _ask(self, fn, node, rel):
        """ФОРМА отказа не важна, важна ОТЛИЧИМОСТЬ: возвращает «вернуло»
        со списком либо «отказ» с именем исключения. Замок намеренно НЕ
        диктует класс исключения — он диктует, что два ответа различимы."""
        try:
            return ("вернуло", list(fn(node, rel)))
        except Exception as exc:                       # noqa: BLE001
            return ("отказ", type(exc).__name__)

    def test_absent_node_refuses_explicitly(self):
        sk = StaticKnowledge(world_of(RIDDLE))
        fn = sk.referenced_by
        known_empty = self._ask(fn, "зинаида", "IS_ROLE")   # узел есть, ссылок нет
        nowhere = self._ask(fn, "пельмень", "IS_ROLE")      # узла нет в мире
        self.assertEqual(known_empty[1], [],
                         f"у Зинаиды кто-то нашёлся: {known_empty}")
        self.assertNotEqual(
            nowhere, known_empty,
            "отказ неотличим от «ссылок нет»: и там и там пустой ответ — "
            "тихий ноль вместо отказа («невестки не существует» однажды "
            "означало бы «у невестки нет хозяина»)")

    def test_negated_edge_stays_out_of_reverse_index(self):
        """«Невестка НЕ жена сына» не имеет права читаться обратным ходом как
        настоящая связь: обратный индекс обязан уважать отрицание ровно так
        же, как прямой (прямой — уважает: negated отбрасывается в __init__).

        Слова в мире ЕСТЬ (так и делает обучение: отрицанное предложение всё
        равно называет участников). Иначе тест мерил бы «узла нет», а не
        «отрицание учтено» — я так и написала сначала, и падение это
        обличило: при пустом списке узлов метод кричал «нет такого слова»,
        что к отрицанию отношения не имело."""
        world = world_of([], negated=[("невестка", "IS_ROLE", "жена")])
        world["entities"] = {"жена": {}, "невестка": {}}
        sk = StaticKnowledge(world)
        self.assertTrue(sk.knows("жена"), "слова мира потерялись — тест "
                                          "спрашивает не про отрицание")
        self.assertEqual(list(sk.referenced_by("жена", "IS_ROLE")), [],
                         "отрицанная связь читается обратным ходом — "
                         "отрицание потерялось на одном из двух ходов")
        self.assertEqual(sk._tails("невестка", "IS_ROLE"), [],
                         "тот же дефект на прямом ходе — значит он не в "
                         "обратном индексе, а в самом чтении мира")


class TestMeetingIsNotIdentity(unittest.TestCase):
    """ЗАПРЕТ НА ПОДМЕНУ. Здесь ничего не чинится и ничего не чинят — этот
    класс о том, ЧЕГО починка иметь НЕ право.

    Замер: правило «сошлись в одном узле» рождает 15 реляционных пар, из них
    настоящая — одна-две (докстринг модуля, experiments/P2_BEFORE.txt). Если
    звено 4 построят «в лоб» по общему узлу, тесты «путь найден» зелены, а мир
    залит ложными родствами. Поэтому запрет пишется ДО починки.

    Из двух проверок ОДНА уж зелёная — и это измерено, а не догадка: прямой
    запрос владельцев возвращает СПИСОК, то есть всех хозяев. Вторая красна,
    потому что требует обратного шага. После починки обе обязаны позеленеть и
    больше не краснеть: они не требуют нового знания, они запрещают подменять
    встречу тождеством."""

    def test_shared_role_node_is_not_one_person(self):
        sk = StaticKnowledge(world_of(RIDDLE + FALSE_MEETS))
        # у брата два носителя роли. Это НЕ значит, что деверь и шурин —
        # один человек: оба могут быть правы и быть разными людьми.
        fn = sk.referenced_by
        brothers = {h for h, r in fn("брат", "IS_ROLE")}
        self.assertEqual(brothers, {"деверь", "шурин"},
                         f"носители роли «брат» потерялись: {brothers}")
        self.assertNotEqual(len(brothers), 1,
                            "носителей роли свалили в одного — это и есть "
                            "подмена тождества встречей")

    def test_two_owners_of_one_slot_are_not_equal(self):
        """«жена PART_OF сын» и «жена PART_OF муж»: у Жены два хозяина.
        Ребёнок и муж — разные люди, хотя оба держат жену. Любая futura
        «идентичность по общему хвосту» покраснеет здесь."""
        sk = StaticKnowledge(world_of([
            ("жена", "PART_OF", "сын"), ("жена", "PART_OF", "муж")]))
        self.assertEqual(sorted(t for t in sk._tails("жена", "PART_OF")),
                         ["муж", "сын"],
                         "роль обязана хранить ВСЕХ владельцев, а не выбрать "
                         "одного: выбор = молчаливое тождество")


class TestWorldOfTheStory(unittest.TestCase):
    """Тот же замер, что и в experiments/P2_BEFORE.txt, но живьём через
    экстрактор: обратный ход обязан работать не на игрушечном мире."""

    def test_reverse_from_definition_reaches_the_person(self):
        from core.extract import SimpleExtractor
        ex = SimpleExtractor()
        edges, seen = [], set()
        for sf in ex.extract_text("Невестка — жена сына. Егор — сын Зинаиды."):
            for f in sf.facts:
                k = (f.head, f.relation, f.tail)
                if k not in seen:
                    seen.add(k)
                    edges.append(k)
        world = world_of([(h, r, t) for h, r, t in edges])
        sk = StaticKnowledge(world)
        fn = sk.referenced_by
        got = sorted({h for h, r in fn("сын", "IS_ROLE")})
        self.assertEqual(got, ["егор"],
                         f"обратный шаг не найден на живом разборе "
                         f"(рёбра {edges}): {got}")


if __name__ == "__main__":
    unittest.main()
