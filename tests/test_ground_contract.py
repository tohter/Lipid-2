"""КОНТРАКТ: у знания есть ОСНОВАНИЕ, и ни один канал не выдаёт его себе сам.

ЗА ЧЕМ ЭТОТ ФАЙЛ.
08.11 живьём измерено: уверенность, с которой канал произнёс фразу,
сохраняется в записи знания (core/records.py::add_witness пишет
max(старое, новое) в поле confidence, brain/learner.py::_commit пересчитывает
от него weight). Источник этого числа — таблица
brain/lipid2_brain.py::CHANNEL_CONFIDENCE:

    {"user": 1.0, "wiki": 0.85, "llm": 0.75, "text": 0.6}

то есть уверенность, ПРИПИСАННАЯ каналу по имени канала, а не заработанная
проверкой. Для CAN_DO вес = 90 + 130*confidence, поэтому «разбор что-то
извлёк» (0.6) даёт weight 168 из 255, а «языковая модель сказала» (0.75) —
187, СРАЗУ, до всякого подтверждения. В этой системе слово «уверенность»
означает «насколько громко сказал источник».

ВТОРОЕ ИЗМЕРЕНИЕ, ИЗ-ЗА КОТОРОГО ПОЛЕ СТАЛО НУЖНО СРОЧНО.
Канал Gemma на 12 проверенных утверждениях ответил правильно 12/12 — и на
всех 12 поставил себе уверенность 1.00. Про бросок кубика и погоду он честно
сказал 50/90/17, то есть шкалу держать умеет, но к СОБСТВЕННЫМ знаниям
применяет её как галочку «знаю / не знаю». 1.00 от модели и 1.00 от Хозяина
в нынешней схеме неотличимы, хотя за первым — пересказ интернета, а за
вторым — слово того, для кого база пишется.

ПОЧЕМУ ЗДЕСЬ НЕТ ЧИСЛА ВЕСА И НЕТ ПРАВА ЗАПИСИ.
Это намеренно НЕ весь ground-механизм, а только его основание. Вес и право
записи — отдельный кирпич, и по нему два советчика сошлись в том, что
сначала надо перестать врать в хранении: 1) босс требует поле `ground` с
четырьмя основаниями; 2) Жозефина (второй советчик, experiments/
ADVISOR_GROUND_JO.txt) доказала, что ОДНО поле лжёт, потому что смешивает
«кто сказал», «как получено» и «почему верим», и потребовала хранить
КОЛЛЕКЦИЮ свидетельств. Её аргумент проверен живьём: четыре пересказа одной
статьи Вики (модель пересказала, затем перефраз, затем пересказ перефраза)
подняли support 1 -> 5 и weight 160 -> 190, хотя независимое основание ОДНО.
Разрешение, принятое здесь и не требующее нового решения Хозяина:
основание ЗАПИСИ — производная от коллекции provenance, которая уже
существует; поле `ground` не первоисточник, а прочтение этой коллекции.
Так выполнение требования босса не предает запрет Жозефины.

ЧТО ЗАПРЕЩЕНО ЭТИМ КОНТРАКТОМ.
1. Основание ВЫВОДИТСЯ ИЗ ИСТОЧНИКА. Канал, назвавший своё слово «аксиомой»
   или «правилом», остаётся свидетельством и права на запись не получает.
2. `rule` вообще недостижим из имени канала: его может выставить только
   проверяющий механизм, у которого есть тест. Ни одна строка «source» к
   нему не ведёт — это и есть запрет «учитель сам себе аксиома».
3. Присланное каналом поле `ground` ПЕРЕЗАПИСЫВАЕТСЯ при записи. Значение
   «rule», приехавшее в payload учителя, не имеет силы.
4. Много пересказов != лучшее основание. support греется, основание остаётся.
5. Незнакомый канал не получает высокого статуса по умолчанию: неизвестно —
   значит чужое свидетельство.
6. Перенос из карантина (staged_candidate) — ПУТЬ, а не канал: он не обязан
   понижать основание, заработанное до карантина.

МУТАЦИИ, ОТ КОТОРЫХ ЭТО ДОРОГО.
- вывести ground из поля, которое пишет учитель: канал начнёт сам решать, что
  он аксиома (проверяется тестом 6 и мутацией);
- сделать max по ground между записями без вывода из источников: пересказ,
  приехавший вторым, «повысит» себя сам;
- бросить unknown в owner: незнакомый канал станет хозяином по умолчанию.
"""
import os
import shutil
import tempfile
import unittest

from lang.ru.lemma import install

install()

from core.records import (GROUNDS, GROUND_OBSERVATION, GROUND_OWNER,          # noqa: E402
                          GROUND_RULE, GROUND_TESTIMONY, add_witness,
                          ground_of, record_ground)
from core.worldfile import read_world, write_world                            # noqa: E402


class TestGroundIsDerived(unittest.TestCase):
    """Основание выводится из ИСТОЧНИКА, а не из заявления канала."""

    def test_1_kinds_of_ground_are_kinds_not_numbers(self):
        """ground — имя типа основания, и список типов закрыт."""
        self.assertEqual({"rule", "owner", "observation", "testimony"},
                         set(GROUNDS),
                         "состав оснований изменился: добавлять тип можно "
                         "только вместе с правилом вывода и тестом на него")

    def test_2_no_source_of_a_channel_becomes_a_rule(self):
        """Ни wiki, ни llm, ни «учитель» не дают права назваться правилом."""
        for source in ("wiki", "llm", "gemma", "text", "agent_teacher",
                       "teacher:wiki", "unknown_channel", "", None,
                       "rule", "axiom"):
            g = ground_of(source)
            self.assertNotEqual(GROUND_RULE, g,
                                f"источник {source!r} получил статус правила "
                                f"— а правило выносит только проверка")
            self.assertIn(g, GROUNDS,
                          f"источник {source!r} остался без основания: "
                          f"неизвестный канал обязан попасть в проверяемое")

    def test_3_owner_and_text_are_distinguished(self):
        """Слово Хозяина и вычитанное из текста — РАЗНЫЕ основания."""
        self.assertEqual(GROUND_OWNER, ground_of("user"))
        self.assertEqual(GROUND_OWNER, ground_of("owner"))
        self.assertEqual(GROUND_OBSERVATION, ground_of("text"))
        self.assertEqual(GROUND_OBSERVATION, ground_of("story"))

    def test_4_second_hand_is_testimony(self):
        """Чужой пересказ — свидетельство, каким бы уверенным он ни был."""
        for source in ("wiki", "llm", "gemma", "agent_teacher",
                       "teacher:wiki", "unknown_channel", "вася"):
            self.assertEqual(GROUND_TESTIMONY, ground_of(source),
                             f"{source!r} — чужой текст или незнакомый канал; "
                             f"он не может быть ничем, кроме свидетельства")

    def test_5_word_axiom_from_a_channel_is_ignored(self):
        """Канал, сам себя назвавший аксиомой, остаётся свидетельством."""
        self.assertEqual(GROUND_TESTIMONY, ground_of("llm", claimed="rule"),
                         "слово «правило» из уст канала стало правилом — это "
                         "ровно та дыра, из-за которой поле и ввели")
        self.assertEqual(GROUND_OWNER, ground_of("user", claimed="rule"),
                         "Хозяин не понижается собственным сомнением: его "
                         "основание — он сам, а не его число")


class TestGroundIsRecorded(unittest.TestCase):
    """Основание ЗАПИСИ — прочтение коллекции свидетельств."""

    def test_6_claimed_ground_is_overwritten(self):
        """Приехавшее в payload «rule» не имеет силы: его перечитывают."""
        record = {"head": "кит", "relation": "IS_A", "tail": "рыба",
                  "ground": "rule",
                  "provenance": [{"source": "wiki", "evidence": "кит — рыба"}]}
        self.assertEqual(GROUND_TESTIMONY, record_ground(record),
                         "основание взяли из поля, которое пишет сам канал: "
                         "любой учитель назначит себе аксиому")

    def test_7_ground_reads_the_whole_witness_list(self):
        """Пересказ, подтверждённый словом Хозяина, перестаёт быть слухом.

        Это и есть ответ Жозефиной: у одного утверждения несколько
        оснований, и они не уничтожают друг друга. Основание записи —
        сильнейшее из СВИДЕТЕЛЬСТВ, каждое из которых помнит свой источник.
        """
        record = {"provenance": [{"source": "wiki", "evidence": "статья"},
                                 {"source": "user", "evidence": "сам сказал"}]}
        self.assertEqual(GROUND_OWNER, record_ground(record))

    def test_8_empty_provenance_is_testimony_not_owner(self):
        """Запись без источника — чужая, а не «просто пока пустая»."""
        self.assertEqual(GROUND_TESTIMONY, record_ground({}))
        self.assertEqual(GROUND_TESTIMONY, record_ground({"provenance": []}))

    def test_9_staging_is_a_path_not_a_source(self):
        """Перенос из карантина не понижает основание, заработанное ранее.

        Цена названа: путь staged_candidate не даёт и повышения. Он ничего
        не утверждает об истинности — он говорит, откуда запись ПРИЕХАЛА.
        """
        record = {"provenance": [{"source": "user", "evidence": "слово"},
                                 {"source": "staged_candidate"}]}
        self.assertEqual(GROUND_OWNER, record_ground(record),
                         "карантин понизил слово Хозяина до пересказа: путь "
                         "следования выдан за основание")

    def test_10_many_retellings_do_not_upgrade_ground(self):
        """Много пересказов != лучшее основание (замеренная дыра).

        Живой замер этой сессии (предсказание Жозефины, подтверждённое
        числом): цепочка «статья Вики -> пересказ модели -> перефраз ->
        пересказ перефраза» подняла support 1 -> 5 и вес 160 -> 190,
        оставшись ОДНИМ независимым основанием.

        Здесь запирается ВТОРАЯ половина того же факта: пока все свидетели
        чужие (модель пересказывает Вики, канал неизвестен), основание
        обязано остаться testimony — сколько бы текстов ни легло в журнал.

        ГРАНИЦА ЗАПРАНОГО (по ней идёт следующий кирпич, и врать про неё
        этот тест не вправе): свидетель с source="text" поднимает основание
        до observation — так решил Хозяин («вычитано из текста»). Отличить
        НЕЗАВИСИМЫЙ текст от пересказа той же статьи текущему механизму
        нечем: для этого нужно происхождение (lineage), а не различие
        текстов. То есть дыра «пересказ приехал как текст» остаётся
        ОТКРЫТОЙ и записана в память, а не замазана этим тестом.
        """
        obs = {"head": "кит", "relation": "IS_A", "tail": "млекопитающее",
               "support": 1, "confidence": 0.0, "negated": False,
               "provenance": [{"source": "wiki",
                               "evidence": "Кит — морское млекопитающее."}]}
        for src, txt in (("llm", "Киты относятся к млекопитающим."),
                         ("gemma", "Кит — млекопитающее, дышит лёгкими."),
                         ("agent_teacher", "Киты — это млекопитающие."),
                         ("unknown_channel", "Млекопитающее, живущее в воде.")):
            add_witness(obs, source=src, evidence=txt, confidence=0.9)
        self.assertGreater(obs["support"], 3,
                           "разные тексты перестали считаться отдельными — "
                           "сравнивать нечего, тест стал пустым")
        self.assertEqual(GROUND_TESTIMONY, record_ground(obs),
                         "пересказ, повторённый много раз, дорос до статуса, "
                         "которого ни разу не проверяли")


class TestGroundReachesTheRealWriters(unittest.TestCase):
    """Основание пишут ОБА писателя, а не один из них.

    Два писателя с разными схемами — это читатель, который умеет читать
    только одного (кирпич про свидетельство-связь). Поэтому здесь идут
    НАСТОЯЩИЕ вызовы обучающего и мозга, а не имитация словаря.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        write_world(self.path, {"entities": {}, "observations": {},
                               "unknown_words": {}})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_11_learner_writes_ground(self):
        """Вычитанное из рассказа получает основание observation."""
        from brain.learner import Learner
        lr = Learner(load_world=lambda: read_world(self.path),
                     save_world=lambda w: write_world(self.path, w))
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False)
        obs = read_world(self.path).get("observations") or {}
        self.assertTrue(obs, "обучающий не оставил ни одного наблюдения")
        for key, rec in obs.items():
            self.assertIn("ground", rec,
                          f"запись {key} не хранит основание: читатель не "
                          f"сможет отличить проверку от пересказа")
            self.assertEqual(GROUND_OBSERVATION, rec["ground"],
                             f"запись {key} из рассказа получила чужой статус")

    def test_12_brain_teacher_channel_writes_ground(self):
        """Пересказ из Вики, записанный мозгом, обязан назваться пересказом."""
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=self.dir)
        brain._staging_target = self.path          # пишем в свой временный мир
        brain._record_observation("кит", {"hierarchy": ["рыба"]}, 0.85,
                                  "кит — рыба", "wiki")
        obs = read_world(self.path).get("observations") or {}
        key = "кит|IS_A|рыба"
        self.assertIn(key, obs, "мозг не записал наблюдение: проверять нечего")
        self.assertEqual(GROUND_TESTIMONY, obs[key].get("ground"),
                         "пересказ Вики записан без пометки «мне пересказали»")


if __name__ == "__main__":
    unittest.main()
