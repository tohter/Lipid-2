"""КОНТРАКТ: подтверждением считается ОТДЕЛЬНОЕ ПРОИСХОЖДЕНИЕ, а не новый
пересказ. Независимость, которую нельзя назвать, не имеет права греть.

ЗА ЧЕМ ЭТОТ ФАЙЛ.
Кирпич 2 запретил греть связь от ПЕРЕСКАЗА: свидетельство считалось
отдельным, если отличается канал ИЛИ текст-основание. Жозефина (второй
советчик, experiments/ADVISOR_GROUND_JO.txt) назвала этот критерий дыркой и
потребовала его запретить: «Википедия пишет: кит — млекопитающее. Gemma
прочитала Википедию и повторила. Другой текст переписал Википедию своими
словами. Третий текст пересказал второй. Система увидит четыре разных текста
и два разных источника. На самом деле у неё может быть одно первоначальное
основание».

ЕЁ ПРЕДСКАЗАНИЕ ПРОВЕРЕНО ЖИВЬЁМ ДО ПРАВОК (два замера этой сессии).
  1) Цепочка «статья Вики -> пересказ модели -> перефраз -> пересказ
     перефраза» подняла support 1 -> 5 и вес 160 -> 190 из 255, имея ОДНО
     независимое основание.
  2) Семь высказываний о связи «дождь CAN_DO гасить» (два рассказа, три
     канала-пересказа, слово Хозяина) дали support = 7 при ТРЁХ различных
     основаниях. То есть четыре «подтверждения» из семи нажиты одним и тем
     же способом производства знания.

ЧТО ИЗМЕРЕНО О ПРОИСХОЖДЕНИИ (и почему это не выдумка).
Происхождение УЖЕ ЕСТЬ в данных, но писатель его ВЫБРАСЫВАЕТ: учитель
Википедии кладёт заголовок статьи в base_entity["title"] (живой замер:
«Кит»), он доезжает до brain/lipid2_brain.py::_record_observation в
аргументе data — и исчезает: в записи наблюдения остаются head/relation/
tail/confidence/evidence/source/support/ground/weight, ссылки на документ
НЕТ. Узнаваемое происхождение могут назвать три канала из семи:
  слово Хозяина — своё начало (он и есть основание);
  статья Вики — doc:wiki:<заголовок>;
  рассказ из файла — doc:text:<имя файла>, ЕСЛИ вызывающий назовёт файл;
  модель Gemma — НЕЧЕГО: состав обучения неизвестен, и её ответ нельзя
  признать независимым от той же Википедии.

НОВОЕ ПРАВИЛО (слова Хозяина 08.11: «поехали» по этому шагу, цена названа).
  Свидетельство ОСТАЁТСЯ в журнале всегда — мы не выбрасываем информацию.
  support (то, что кормит достоверность) растёт ТОЛЬКО от свидетельства с
  НЕРАЗЛИЧЁННЫМ происхождением. Три состояния Жозефины:
      INDEPENDENT — происхождение названо и встречается впервые;
      DEPENDENT   — происхождение названо и уже было (тот же документ,
                    drugim словом или тем же);
      UNKNOWN     — назвать нечем. НЕ греет НИКОГДА, но сохраняется.
  Совпадение слов из двух неизвестных документов — НЕ подтверждение: два
  текста могли переписать одну статью. Это стоит дороже, чем кирпич 2, и
  это часть контракта: у старых текстов без имени документа подтверждений
  станет меньше, а не больше.

ЧЕСТНАЯ ЦЕНА (замерена ДО правки, а не после).
  support 7 -> 3 на семи высказываниях; молчание моделей (Gemma, llm)
  становится нормой: их ответы больше НИКОГДА не поднимут достоверность
  сами по себе, пока независимость не будет названа. Ответ Хозяина — да.
  Второй рассказ из ДРУГОГО файла — да, если вызывающий назвал файл.
  Что потеряно: связь, подтверждённую пятью пересказами одной статьи,
  система будет считать неподтверждённой. Это выбор по закону проекта:
  ложная уверенность в связи хуже молчаливой дыры, потому что её нечем
  перепроверить — число выглядит заработанным.

МУТАЦИИ, ОТ КОТОРЫХ ЭТО ДОРОГО.
- греть support от UNKNOWN: дыра Жозефины возвращается целиком;
- считать происхождение по каналу без документа: «wiki» и «wiki» станут
  двумя основаниями, хотя это одна и та же статья;
- выбрасывать UNKNOWN-свидетельство из журнала: мы потеряем данные,
  которые можно будет перепроверить, когда происхождение выяснится.
"""
import os
import shutil
import tempfile
import unittest

from lang.ru.lemma import install

install()

from core.records import (INDEPENDENT, UNKNOWN, add_witness,               # noqa: E402
                          independent_bases, lineage_of, merge_observation,
                          witness_state)
from core.worldfile import read_world, write_world                          # noqa: E402


def _запись(source, evidence, document=""):
    """Первое свидетельство связи — так, как его создаёт писатель."""
    lin = lineage_of(source, document)
    return {"head": "кит", "relation": "IS_A", "tail": "млекопитающее",
            "support": 1, "confidence": 0.0, "negated": False,
            "source": source, "evidence": evidence,
            "lineage": lin,
            "provenance": [{"source": source, "evidence": evidence,
                            "lineage": lin}]}


class TestLineageIsNamed(unittest.TestCase):
    """Происхождение обязано БЫТЬ НАЗВАНО или честно отсутствовать."""

    def test_1_document_names_the_lineage(self):
        """Статья Вики — происхождение; пересказ той же статьи — тоже оно.

        Ключ привязан к ДОКУМЕНТУ, а не к паре «канал + документ». Это
        исправлено по замеру, а не по вкусу: когда канал входил в ключ
        (doc:wiki:кит против doc:llm:кит), модель, назвавшая источник
        «Кит», получала ОТДЕЛЬНОЕ основание рядом с прочитанной статьёй
        «Кит» — и тест 9 дал 4 основания вместо 1.
        """
        self.assertEqual("doc:кит", lineage_of("teacher:wiki", "Кит"))
        self.assertEqual("doc:кит", lineage_of("wiki", "  Кит "))
        self.assertEqual("doc:кит", lineage_of("llm", "Кит"),
                         "пересказ статьи обязан наследовать её происхождение")
        self.assertEqual("doc:рассказ про костёр",
                         lineage_of("text", "Рассказ про костёр"))

    def test_2_owner_is_his_own_origin(self):
        """Слово Хозяина не обязана приезжать из документа."""
        self.assertEqual("owner", lineage_of("user"))
        self.assertEqual("owner", lineage_of("owner"))

    def test_3_channel_without_document_has_no_lineage(self):
        """Модель и безымянный текст НЕЧЕГО назвать. Это UNKNOWN, а не 0."""
        self.assertEqual("", lineage_of("llm"))
        self.assertEqual("", lineage_of("gemma"))
        self.assertEqual("", lineage_of("text"))
        self.assertEqual("", lineage_of("wiki"))
        self.assertEqual("", lineage_of("вася"))
        self.assertEqual("", lineage_of(None))

    def test_4_the_wiki_title_is_not_invented_by_the_writer(self):
        """Заголовок берётся из данных учителя, а не сочиняется из концепта.

        ЗАМЕР: учитель кладёт base_entity["title"]="Кит", и этот заголовок
        доезжает до писателя в data — но в запись наблюдения не попадает.
        Если писатель начнёт ставить lineage:=head, две разные статьи об
        одном слове сольются в одно происхождение, а один и тот же заголовок
        в двух каналах раздвоится.
        """
        self.assertEqual("", lineage_of("wiki", ""),
                         "происхождение выведено из имени концепта: статьи "
                         "«Кит» и «Киты» станут одним основанием или двумя — "
                         "как повезёт")


class TestWitnessStates(unittest.TestCase):
    """Три состояния свидетельства и право каждого на рост достоверности."""

    def test_5_first_witness_is_independent(self):
        obs = _запись("wiki", "Кит — млекопитающее.", "Кит")
        self.assertEqual(1, independent_bases(obs))
        self.assertEqual(INDEPENDENT, witness_state(obs, "wiki",
                                                    "другая фраза", "Киты"))

    def test_6_second_sentence_of_the_same_document_is_dependent(self):
        """Второй абзац той же статьи — НЕ второе подтверждение.

        Это ровно то, что кирпич 2 пропускал: разные предложения одного
        документа он считал разными свидетельствами, потому что сравнивал
        тексты.
        """
        obs = _запись("wiki", "Кит — морское млекопитающее.", "Кит")
        grew = add_witness(obs, source="wiki",
                           evidence="Дышат лёгкими, выкармливают детёнышей.",
                           confidence=0.9, document="Кит")
        self.assertFalse(grew, "поддержка выросла от второго абзаца той же "
                              "статьи: support кормится документом, а не его "
                              "объёмом")
        self.assertEqual(1, independent_bases(obs))
        self.assertEqual(1, int(obs["support"]),
                         "support вырос там, где произошло одно основание")

    def test_7_unknown_origin_never_warms(self):
        """Ответ модели не поднимает достоверность. Совсем. Ни разу.

        Цена принята вслух: модель, угадавшая правильно, не делает связь
        убедительнее. Её ответ остаётся в журнале и может быть перепроверен,
        но независимость её от тех же статей неизвестна — а необоснованное
        повышение доверия и есть самообман.
        """
        obs = _запись("wiki", "Кит — млекопитающее.", "Кит")
        grew = add_witness(obs, source="llm",
                           evidence="Киты относятся к млекопитающим.",
                           confidence=1.0)
        self.assertFalse(grew, "UNKNOWN-происхождение подогрело поддержку")
        self.assertEqual(1, int(obs["support"]))
        self.assertEqual(UNKNOWN, witness_state(obs, "llm", "киты — млекопитающие"))

    def test_8_two_documents_are_two_bases(self):
        """Два РАЗНЫХ документа — два основания, и это видно числом."""
        obs = _запись("wiki", "Кит — млекопитающее.", "Кит")
        grew = add_witness(obs, source="text",
                           evidence="Киты — теплокровные млекопитающие.",
                           confidence=0.6, document="зоология-лекция-3")
        self.assertTrue(grew, "два независимых источника перестали быть "
                              "двумя основаниями")
        self.assertEqual(2, independent_bases(obs))
        self.assertEqual(2, int(obs["support"]))

    def test_9_retelling_of_a_known_document_is_dependent(self):
        """Модель, пересказавшая СТАТЬЮ, которую мы читали, не независима.

        Это и есть дыра, названная Жозефиной высоким риском: четыре текста,
        одно первоначальное основание. Различие текстов больше не аргумент.
        """
        obs = _запись("wiki", "Кит — морское млекопитающее.", "Кит")
        for src, ev in (("llm", "Киты — это млекопитающие."),
                        ("text", "Кит относится к классу млекопитающих."),
                        ("gemma", "Млекопитающее, живущее в море.")):
            add_witness(obs, source=src, evidence=ev, confidence=0.9,
                        document="Кит")
        self.assertEqual(1, independent_bases(obs),
                         "три пересказа одной статьи посчитались тремя "
                         "основаниями")
        self.assertEqual(1, int(obs["support"]))

    def test_10_unknown_witness_is_KEPT_not_thrown_away(self):
        """Не греть — не значит выбросить: журнал обязан остаться полным.

        Мы отказываемся от необоснованного повышения доверия, но не от
        данных: когда происхождение выяснится, свидетельство можно будет
        пересчитать.
        """
        obs = _запись("wiki", "Кит — млекопитающее.", "Кит")
        было = len(obs["provenance"])
        add_witness(obs, source="llm", evidence="Киты — млекопитающие.",
                    confidence=0.9)
        self.assertEqual(было + 1, len(obs["provenance"]),
                         "свидетельство выброшено из журнала: перепроверять "
                         "будет нечего")
        self.assertEqual(1, int(obs["support"]),
                         "но поддержка от него не выросла")

    def test_11_owner_word_is_always_its_own_basis(self):
        """Слово Хозяина — отдельное основание даже рядом со статьёй."""
        obs = _запись("wiki", "Кит — млекопитающее.", "Кит")
        grew = add_witness(obs, source="user",
                           evidence="кит — млекопитающее, я проверял",
                           confidence=1.0)
        self.assertTrue(grew, "слово Хозяина не стало вторым основанием")
        self.assertEqual(2, independent_bases(obs))

    def test_12_no_document_means_no_growth_even_across_channels(self):
        """Два канала без документа — НЕ два основания (замеренная цена).

        Старый порядок считал это подтверждением (tests
        test_repeat_witness_contract тест 6, test_evidence_link_contract
        тест 4 — оба переписаны вместе с этим законом). Основание для
        перемены: тексты без имени документа могли быть переписаны с одной
        страницы, и отличить это нам нечем.
        """
        obs = _запись("text", "Дождь гасит огонь.")
        grew = add_witness(obs, source="wiki",
                           evidence="Дождь гасит огонь.", confidence=0.9)
        self.assertFalse(grew, "совпадение двух безымянных каналов снова "
                              "греться без происхождения — именно то, что запрещено")
        self.assertEqual(1, int(obs["support"]))


class TestRefinementCountsOnce(unittest.TestCase):
    """Происхождение, названное ПОЗЖЕ, засчитывается РОВНО ОДИН РАЗ.

    Это ответ на вопрос «зачем хранить UNKNOWN, если он не греет». Храним
    мы его не для красоты: когда происхождение станет известно, связь нужно
    пересчитать. Если бы зачёт не происходил вовсе, число подтверждений
    осталось бы навечным враньём в меньшую сторону, а врать в меньшую
    сторону мы лечили ровно тем же методом, что и в большую: молчанием.
    """

    def test_16_refinement_of_top_level_witness_counts_once(self):
        """Первый свидетель у обучающего лежит ВЕРХНИМ уровнем записи.

        ЗАМЕР 08.11: поиск происхождения только по журналу делал уточнение
        первого свидетельства невозможным — оно оставалось безымянным
        навечно, независимо от того, узнали мы документ или нет.
        """
        rec = {"head": "кит", "relation": "IS_A", "tail": "млекопитающее",
               "support": 1, "confidence": 0.6, "negated": False,
               "lineage": "", "source": "llm",
               "evidence": "Киты — млекопитающие.", "weight": 200,
               "provenance": []}
        # Тот же свидетель, но теперь мы ЗНАЕМ, откуда он приехал.
        grew = add_witness(rec, source="llm",
                           evidence="Киты — млекопитающие.",
                           confidence=0.6, relation="IS_A",
                           document="Кит")
        self.assertTrue(grew, "уточнённое происхождение не засчитано: "
                             "хранить UNKNOWN тогда не за чем")
        self.assertEqual(2, int(rec["support"]))
        self.assertEqual(1, independent_bases(rec),
                         "число оснований разошлось с числом засчитанных "
                         "подтверждений")
        # Повторное уточнение тем же документом не имеет права греть снова.
        grew = add_witness(rec, source="llm",
                           evidence="Киты — млекопитающие.",
                           confidence=0.6, relation="IS_A",
                           document="Кит")
        self.assertFalse(grew, "уточнение посчитали подтверждением дважды")
        self.assertEqual(2, int(rec["support"]))

    def test_17_anonymous_witness_is_kept_but_names_nothing(self):
        """Безымянный свидетель сохраняется, но НЕ приписывает происхождения.

        Мутация, которую ловит этот тест: «если lineage пуст, возьми
        происхождение записи». Тогда одна безымянная фраза модели
        унаследовала бы документ первого свидетеля, и журнал показал бы два
        независимых основания там, где прочитана одна статья.
        """
        rec = {"head": "кит", "relation": "IS_A", "tail": "млекопитающее",
               "support": 1, "confidence": 0.6, "negated": False,
               "lineage": "doc:кит", "source": "wiki",
               "evidence": "Кит — морское млекопитающее.", "weight": 200,
               "provenance": [{"source": "wiki",
                               "evidence": "Кит — морское млекопитающее.",
                               "lineage": "doc:кит"}]}
        grew = add_witness(rec, source="llm", evidence="Киты дышат лёгкими.",
                           confidence=0.9, relation="IS_A")
        self.assertFalse(grew,
                         "свидетельство без происхождения подогрело support")
        self.assertEqual(1, int(rec["support"]))
        self.assertEqual(2, len(rec["provenance"]),
                         "данные потеряны вместо отказа в доверии")
        self.assertFalse((rec["provenance"][-1] or {}).get("lineage"),
                         "безымянному свидетельству приписано чужое "
                         "происхождение: одна статья стала двумя основаниями")
        self.assertEqual(1, independent_bases(rec))

    def test_18_named_document_after_anonymous_one_is_a_real_basis(self):
        """Названный документ после безымянного — настоящее основание.

        Это та же цена, что и в тесте 16, но на другом писательском пути:
        безымянное свидетельство НЕ блокирует рост, оно просто не греет
        само. Иначе закон превратился бы в «один раз соврал — навечно
        немой».
        """
        rec = _запись("llm", "Кит — млекопитающее.")
        self.assertEqual(0, independent_bases(rec))
        self.assertTrue(add_witness(rec, source="wiki",
                                    evidence="Кит — млекопитающее.",
                                    confidence=0.9, document="Кит"))
        self.assertEqual(2, int(rec["support"]))
        self.assertEqual(1, independent_bases(rec))


class TestMergeKeepsTheJournal(unittest.TestCase):
    """СЛИЯНИЕ снимков обязано сохранять журнал и происхождение.

    Этот класс появился НЕ из подозрения, а из замера, сделанного этим
    кирпичом. brain/learner.py держал МЕСТНУЮ копию слияния наблюдений
    (та же болезнь, что и местное правило свидетеля до кирпича 2): она
    переносила support/confidence/negated и НЕ переносила provenance и
    lineage. Замер 08.11: отказавшее (независимость не названа)
    свидетельство добавлялось в память, а после записи на диске его не
    оказывалось. Значит обещание «UNKNOWN сохраняется и будет
    пересмотрено» было текстом — молча потерянные данные хуже потерянных
    громко, их некому восстановить.
    """

    def _commit(self, disk, mine):
        from brain.learner import Learner
        saved = {}
        lr = Learner(load_world=lambda: dict(disk),
                     save_world=lambda w: saved.update(w))
        lr._commit(mine)
        return saved

    @staticmethod
    def _rec(source, evidence, lineage, support, **extra):
        rec = {"head": "кит", "relation": "IS_A", "tail": "китообразные",
               "confidence": 0.5, "negated": False, "support": support,
               "source": source, "evidence": evidence, "lineage": lineage,
               "weight": 100}
        rec.update(extra)
        return rec

    def test_19_foreign_witness_survives_the_write(self):
        """Чужое свидетельство не имеет права исчезнуть при записи.

        Диск (писал другой процесс) и наш снимок содержат ОДНУ связь, но
        РАЗНЫЕ основания. После слияния журнал обязан содержать чужую
        строку: иначе второй писатель молча стёр опыт первого.
        """
        key = "кит|IS_A|китообразные"
        disk = {"entities": {}, "unknown_words": {},
                "observations": {key: self._rec("wiki", "статья про китов",
                                               "doc:кит", 1)}}
        mine = {"entities": {}, "unknown_words": {}, "events": [],
                "expectations": [],
                "observations": {key: self._rec(
                    "text", "киты — млекопитающие", "doc:рассказ", 1,
                    provenance=[{"source": "text",
                                 "evidence": "киты — млекопитающие",
                                 "lineage": "doc:рассказ"}])}}
        rec = self._commit(disk, mine)["observations"][key]
        self.assertEqual(1, len(rec.get("provenance") or []),
                         "журнал потерян при слиянии: чужой писатель стёр "
                         "наше свидетельство молча")
        self.assertEqual(2, independent_bases(rec),
                         "два разных основания слились в одно")
        self.assertEqual(2, int(rec["support"]),
                         "число подтверждений разошлось с числом оснований: "
                         "слияние посчитало один опыт дважды или потеряло "
                         "второй")

    def test_19b_lineage_of_the_writing_side_is_not_dropped(self):
        """Происхождение ТОЙ СТОРОНЫ, что пишет, обязано пережить запись.

        Снимок с диска приходит БЕЗ журнала (первый свидетель лежит верхним
        уровнем). Если слияние переносит support/confidence, но не lineage,
        связь на диске остаётся безымянной навечно — и каждое её повторение
        повисает UNKNOWN. Это ровно тот случай, который мутация «pass»
        делала невидимым для остальных проверок.
        """
        key = "кит|IS_A|китообразные"
        disk = {"entities": {}, "unknown_words": {},
                "observations": {key: self._rec("llm", "киты — млекопитающие",
                                               "", 1)}}
        mine = {"entities": {}, "unknown_words": {}, "events": [],
                "expectations": [],
                "observations": {key: self._rec("wiki", "статья",
                                               "doc:кит", 1)}}
        rec = self._commit(disk, mine)["observations"][key]
        self.assertEqual("doc:кит", rec.get("lineage"),
                         "происхождение записи потеряно при слиянии двух "
                         "снимков: связь осталась безымянной навечно")

    def test_20_support_follows_bases_after_merge(self):
        """Поддержка после слияния = число ОСНОВАНИЙ, а не сумма чисел.

        Иначе два процесса, независимо засчитавшие ОДНО И ТО ЖЕ основание,
        удвоили бы подтверждение — ровно та дыра, против которой закон
        написан, только приехавшая с другой стороны.
        """
        key = "кит|IS_A|китообразные"
        same = {"source": "text", "evidence": "киты — млекопитающие",
                "lineage": "doc:рассказ"}
        # У ОБОИХ снимков одна и та же строка журнала: слияние обязано
        # её слить, а не склеить (иначе журнал пухнет и его начинают
        # читать как число наблюдений).
        disk = {"entities": {}, "unknown_words": {},
                "observations": {key: self._rec(
                    "wiki", "статья", "doc:кит", 1,
                    provenance=[dict(same)])}}
        mine = {"entities": {}, "unknown_words": {}, "events": [],
                "expectations": [],
                "observations": {key: self._rec(
                    "wiki", "статья", "doc:кит", 1,
                    provenance=[dict(same)])}}
        rec = self._commit(disk, mine)["observations"][key]
        self.assertEqual(1, len(rec.get("provenance") or []),
                         "одна и та же строка журнала записана дважды")
        self.assertEqual(2, int(rec["support"]),
                         "support обязан согласоваться с числом оснований "
                         "(здесь их два: статья и рассказ)")
        self.assertEqual(2, independent_bases(rec))


class TestAliasBranchKeepsTheLaw(unittest.TestCase):
    """Ветка СИНОНИМОВ обязана греться тем же законом, что и факты.

    ЗАМЕР 08.11 (мутации «алиасная ветка потеряла документ» и «алиасная
    ветка греет support в обход закона» ВЫЖИЛИ). Факт-ветка обучала через
    add_witness, а синонимы продолжали делать support += 1 прямым счётом:
    второй прогон того же рассказа поднимал поддержку SYNONYM_OF на глазок,
    без единого нового наблюдения — тот же класс дыры, что чинил кирпич 2,
    только на второй ветке писателя. Мёртвый закон дороже отсутствующего:
    он создаёт видимость охраны.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        write_world(self.path, {"entities": {}, "observations": {},
                               "unknown_words": {}})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _lr(self):
        from brain.learner import Learner
        return Learner(load_world=lambda: read_world(self.path),
                       save_world=lambda w: write_world(self.path, w))

    def _obs(self):
        return read_world(self.path)["observations"]

    def test_21_second_reading_of_a_story_does_not_warm_a_synonym(self):
        """Тот же рассказ второй раз не подтверждает синонима."""
        lr = self._lr()
        story = "Бегемот, или гиппопотам, живёт у воды."
        lr.learn_from_text(story, source="text", allow_teacher=False)
        key = "гиппопотам|SYNONYM_OF|бегемот"
        first = int(self._obs()[key]["support"])
        lr.learn_from_text(story, source="text", allow_teacher=False)
        self.assertEqual(first, int(self._obs()[key]["support"]),
                         "пересказ одного и того же рассказа подогрел "
                         "синонима: факт-ветка охраняется, aliases — нет")

    def test_22_named_document_reaches_the_alias_record(self):
        """Синоним обязан помнить, из какого документа он взят."""
        lr = self._lr()
        lr.learn_from_text("Бегемот, или гиппопотам, живёт у воды.",
                           source="text", allow_teacher=False,
                           document="зоология-том-1")
        rec = self._obs()["гиппопотам|SYNONYM_OF|бегемот"]
        self.assertIn("зоология-том-1", (rec.get("lineage") or "").lower(),
                      "запись синонима не помнит документ: её нельзя ни "
                      "подтвердить, ни опровергнуть")

    def test_23_alias_from_another_book_is_a_second_basis(self):
        """ДРУГАЯ названная книга даёт синониму второе основание."""
        lr = self._lr()
        lr.learn_from_text("Бегемот, или гиппопотам, живёт у воды.",
                           source="text", allow_teacher=False,
                           document="зоология-том-1")
        key = "гиппопотам|SYNONYM_OF|бегемот"
        first = int(self._obs()[key]["support"])
        # ТОТ ЖЕ порядок аппозиции: «Бегемот, или гиппопотам» ставит
        # варианто бегемотом, а канон — гиппопотам; обратный порядок дал бы
        # ДРУГУЮ связь и проверял бы не закон, а падеж.
        lr.learn_from_text("Бегемот, или гиппопотам, водится в реках.",
                           source="text", allow_teacher=False,
                           document="энциклопедия-животных")
        self.assertEqual(first + 1, int(self._obs()[key]["support"]),
                         "вторая книга не стала вторым основанием: закон "
                         "слишком далеко зашёл и молчит там, где обязан "
                         "говорить")


class TestLineageThroughRealWriters(unittest.TestCase):
    """Происхождение обязано доезжать от writers, а не только существовать."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "lipid_world.json")
        write_world(self.path, {"entities": {}, "observations": {},
                               "unknown_words": {}})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_13_learner_keeps_the_document_it_was_given(self):
        """Если вызывающий назвал файл, запись помнит своё происхождение."""
        from brain.learner import Learner
        lr = Learner(load_world=lambda: read_world(self.path),
                     save_world=lambda w: write_world(self.path, w))
        lr.learn_from_text("Дождь гасит огонь.", source="text",
                           allow_teacher=False,
                           document="рассказ-про-костёр.txt")
        obs = read_world(self.path)["observations"]
        self.assertTrue(obs, "обучающий не оставил наблюдений")
        for key, rec in obs.items():
            self.assertTrue(rec.get("lineage"),
                            f"запись {key} не помнит, откуда пришла: "
                            f"её нельзя ни подтвердить, ни опровергнуть")
            self.assertIn("рассказ-про-костёр.txt",
                          (rec.get("lineage") or "").lower(),
                          f"происхождение записи {key} не совпадает с "
                          f"переданным документом: {rec['lineage']}")

    def test_14_wiki_article_title_survives_to_the_record(self):
        """Заголовок статьи Вики обязан дожить до наблюдения.

        ЗАМЕР до правки: base_entity["title"]="Кит" доезжает до
        _record_observation в data и выбрасывается — две статьи об одном
        слове и пять пересказов одной статьи были неразличимы.
        """
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=self.dir)
        brain._staging_target = self.path
        brain._record_observation("кит",
                                  {"title": "Кит", "hierarchy": ["китообразные"]},
                                  0.85, "Киты — морские млекопитающие.",
                                  "teacher:wiki")
        rec = read_world(self.path)["observations"]["кит|IS_A|китообразные"]
        self.assertIn("кит", (rec.get("lineage") or "").lower(),
                      "заголовок статьи потерян: происхождение знания "
                      "снова нельзя назвать")

    def test_15_two_articles_of_the_same_word_are_two_bases(self):
        """«Кит» и «Киты» — РАЗНЫЕ документы, а не одно происхождение."""
        from brain.lipid2_brain import Lipid2Brain
        brain = Lipid2Brain(memory_dir=self.dir)
        brain._staging_target = self.path
        brain._record_observation("кит", {"title": "Кит",
                                         "hierarchy": ["китообразные"]},
                                  0.85, "Киты — морские млекопитающие.",
                                  "teacher:wiki")
        brain._record_observation("кит", {"title": "Киты",
                                         "hierarchy": ["китообразные"]},
                                  0.85, "Китообразные — отряд млекопитающих.",
                                  "teacher:wiki")
        rec = read_world(self.path)["observations"]["кит|IS_A|китообразные"]
        self.assertEqual(2, int(rec["support"]),
                         "две разные статьи посчитаны одним основанием")
        self.assertEqual(2, independent_bases(rec))


if __name__ == "__main__":
    unittest.main()
