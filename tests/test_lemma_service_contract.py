# -*- coding: utf-8 -*-
"""
КОНТРАКТ ЯЗЫКОВЫХ СЛУЖБ ЯДРА (шаги A2.2 и A2.3 границы языка)

Зачем этот файл существует. В A2.2 ядро перестало импортировать язык: из
core/records.py убрали `import pymorphy3`, лемматизатор теперь ПРИНОСЯТ
(core.records.set_lemma_service, носитель — lang/ru/lemma.py). В A2.3 туда же
ушла часть речи: из core/actions.py убрали `from core.morph import Morph`,
распознаватель глаголов приносит тот же install(). Это закрыло швы, но открыло
новый, худший класс отказов: forgotten installer.

Почему этот класс хуже обычного бага. Функции records — свободные, их зовут 17
мест (замер AST по вызовам same_word/strip_self_name/meaningful_categories/
is_empty_entity/merge_entity), из них 6 производственных. Если служба не
внедрена, ядро НЕ падает: оно честно предупреждает в stderr и уходит на
наивное отсечение. То есть процесс продолжает работать и писать во Вселенную
худшими ответами. Именно так мозг уже мог потерять резолвер форм в A2.1 — и
весь набор тестов был зелёный (мутант «мозг не передал surface_forms» тогда
прошёл). Здесь масштаб больше, потому что писем в мир шесть, а не одно.

Поэтому этот файл проверяет ТРИ разные вещи:

  1. СПОСОБНОСТЬ СЛЫШАТЬ отказ (TestDegradationIsLoud, TestPosDegradationIsLoud)
     — деградация обязана быть видна, но не должна орать на каждую фразу.
  2. НЕВОЗМОЖНОСТЬ ЗАБЫТЬ (TestNobodyForgets) — у КАЖДОГО производственного
     входа, сводящего формы слов, служба стоит. Это проверка не кода, а
     множества входов: она же ловит и новый вход, который появится завтра.
  3. ЦЕЛОСТНОСТЬ ТОЧКИ УСТАНОВКИ (test_install_brings_BOTH_services_not_one) —
     один install() обязан внедрять ВСЕ службы. Забыть вторую строку в общем
     установщике молча нельзя: процесс с половиной языка выглядит рабочим.

Отдельно: TestProviderIsEquivalent — замена pymorphy3 -> Morph.lemma обязана
быть равной СТАРОМУ поведению, иначе «закрыли шов» означает «тихо уронили
качество». Числа в этом файле взяты из замера, см. experiments/IR_BOUNDARY.md.
"""
import ast
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import core.records as R
import core.actions as A
from lang.ru.lemma import install as install_service

# Функции records, чьё решение ЗАВИСИТ от сведения форм. Без службы они
# отвечают иначе (эвристика), поэтому любой файл, который их зовёт, обязан
# принести лемматизатор — иначе он проверяет не то, что думает.
MORPH_DEPENDENT = frozenset({
    "same_word", "strip_self_name", "meaningful_categories",
    "is_empty_entity", "merge_entity",
})

# Производственные входы, которые сводят формы слов (замер AST: импорт или
# вызов same_word / strip_self_name / meaningful_categories / is_empty_entity /
# merge_entity вне tests/, experiments/ и самого core/records.py).
#
# ЯДРО СПИСКА НЕТ намеренно: core/storage.py реэкспортирует merge_entity, но
# сам его не вызывает, а ядро внедрять язык не может по закону
# (C_core_reaches_lang = 0). Требование «принеси службу» физически может
# выполнить только тот, кому язык разрешён.
PRODUCTION_ENTRIES = (
    "brain.lipid2_brain",
    "brain.learner",
    "teachers.teacher1",
    "teachers.teacher_wiki",
    "tools.clean_world",
)

# Является ли вход библиотечным импортом (тогда служба ставится на импорте)
LIBRARY_STYLE = PRODUCTION_ENTRIES


def state_after_import(module: str) -> dict:
    """
    Состояние службы В ОТДЕЛЬНОМ ПРОЦЕССЕ после импорта module.

    Подпроцесс обязателен: служба ставится на процесс, и в общем интерпретаторе
    любой ранее импортированный brain уже поставил бы её за проверяемого —
    проверка «забыл ли вход» превратилась бы в самообман.
    """
    probe = (
        "import json,sys;sys.path.insert(0,%r);"
        "import %s;"
        "import core.records as R;"
        "print('STATE'+json.dumps(R.lemma_service_state()))" % (ROOT, module)
    )
    env = dict(os.environ, LIPID2_STATUS="0")
    res = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=900)
    for line in res.stdout.splitlines():
        if line.startswith("STATE"):
            return __import__("json").loads(line[5:])
    raise AssertionError(
        f"{module}: зонд не ответил (код {res.returncode})\n"
        f"stderr: {res.stderr[-800:]}")


class TestNobodyForgets(unittest.TestCase):
    """
    Забыть службу — не аргумент. Каждый производственный вход обязан её нести.
    """

    def test_every_production_entry_installs_the_service(self):
        for mod in PRODUCTION_ENTRIES:
            with self.subTest(entry=mod):
                st = state_after_import(mod)
                self.assertTrue(
                    st["installed"],
                    f"{mod}: сводит формы слов БЕЗ службы — пишет во "
                    f"Вселенную эвристикой (онтологический мусор с тем лицом, "
                    f"что всё работает). Принести: "
                    f"from lang.ru.lemma import install; install()")

    def test_bare_core_records_does_NOT_install(self):
        """
        Ядро обязано остаться НЕвнедрившим. Если кто-то «починит» отсутствие
        службы импортом внутри core/ — трещотка langborder это поймает, но
        поймают и здесь, раньше и по-человечески: ядро, сами себе выдавшее
        язык, перестаёт быть ядром.
        """
        st = state_after_import("core.records")
        self.assertFalse(st["installed"],
                         "core/records начал сам себя снабжать языком — это "
                         "возврат шва внутрь ядра, закон нарушен")

    def test_core_storage_stays_language_free(self):
        """core/storage реэкспортирует merge_entity, но языка знать не должен."""
        st = state_after_import("core.storage")
        self.assertFalse(st["installed"])

    def test_install_brings_BOTH_services_not_one(self):
        """
        Точка установки обязана внедрять ВСЁ, что ядро разучилось знать сама.

        Почему это отдельный замок, а не деталь. С A2.3 у lang/ru/lemma.py ДВЕ
        службы: сведение форм (core/records) и часть речи (core/actions). Один
        install() на обе выбран намеренно — но сам себя он не охраняет: строку
        set_pos_service можно вычеркнуть, и набор про это МОЛЧИТ. Проверено
        мутацией: без этой проверки вычёркивание проходило всем набором, кроме
        двух тестов чистильщика, то есть сигнал зависел от файла, который про
        службы вообще ничего не знает.

        Цена половинчатого процесса измерена: леммы сведены, глаголы распознаются
        справочником, «чинить»/«опылять»/«катиться» становятся сущностями с
        неверным диагнозом. Выглядит как работающий мозг.
        """
        probe = (
            "import json,sys;sys.path.insert(0,%r);"
            "import core.records as R, core.actions as A;"
            "import lang.ru.lemma as L;"
            "L.install();"
            "print('STATE'+json.dumps({'lemma': R.lemma_service_state(),"
            "'pos': A.pos_service_state()}))" % ROOT)
        env = dict(os.environ, LIPID2_STATUS="0")
        res = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=900)
        for line in res.stdout.splitlines():
            if line.startswith("STATE"):
                st = __import__("json").loads(line[5:])
                break
        else:
            self.fail(f"зонд install() не ответил (код {res.returncode})\n"
                      f"stderr: {res.stderr[-800:]}")
        for svc, name in (("lemma", "сведение форм (core/records)"),
                          ("pos", "часть речи (core/actions)")):
            self.assertTrue(st[svc]["installed"],
                            f"install() не принёс службу «{name}»: процесс с "
                            f"половиной языка работает и пишет в мир худшими "
                            f"ответами. Починка: lang/ru/lemma.py обязан "
                            f"внедрять обе службы одним вызовом.")
            self.assertIn("lang.ru.lemma", st[svc]["name"],
                          f"служба «{name}» внедрена не из lang/ru/lemma.py — "
                          f"кто-то завёл вторую точку установки, и забыть про "
                          f"неё снова станет возможно")

    def test_pos_dependent_entry_has_pos_service(self):
        """
        Инструмент, решающий по части речи, обязан иметь распознаватель.

        Список зависимых составлен НЕ на глаз: AST-обход всего дерева показал,
        что из производственных входов looks_like_verb зовёт ровно один файл —
        tools/clean_world.py. Требовать pos-службу от brain/, который её не
        вызывает, значило бы сочинять требование (урок 22: цифра без замера).
        """
        probe = (
            "import json,sys;sys.path.insert(0,%r);"
            "import tools.clean_world;"
            "import core.actions as A;"
            "print('STATE'+json.dumps(A.pos_service_state()))" % ROOT)
        env = dict(os.environ, LIPID2_STATUS="0")
        res = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=900)
        for line in res.stdout.splitlines():
            if line.startswith("STATE"):
                st = __import__("json").loads(line[5:])
                break
        else:
            self.fail(f"зонд pos-службы не ответил (код {res.returncode})\n"
                      f"stderr: {res.stderr[-800:]}")
        self.assertTrue(st["installed"],
                        "tools/clean_world распознаёт глаголы без распознавателя "
                        "— глаголы вне справочника получат неверный asked_empty "
                        "и исчезнут из очереди вопросов навсегда")


class TestDegradationIsLoud(unittest.TestCase):
    """Без службы — громко, но не на каждую фразу (сердце бьётся часто)."""

    def setUp(self):
        self.saved = R.lemma_service_state()
        R.clear_lemma_service()

    def tearDown(self):
        # Восстановление ОБЯЗАТЕЛЬНО: состояние службы общее на процесс, и
        # оставь мы его выключенным — все тесты после этого файла поедут на
        # эвристике, позеленев. Это тот же дефект, что глушит смысл набора.
        if self.saved["installed"]:
            install_service()
        else:
            R.clear_lemma_service()

    def test_missing_service_is_announced(self):
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            R._lemma_simple("кошки")
        text = err.getvalue()
        self.assertIn("НЕ ВНЕДРЕНА", text)
        self.assertIn("set_lemma_service", text)

    def test_warning_is_not_repeated(self):
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            for w in ("кошки", "кошке", "кошек", "кошку"):
                R._lemma_simple(w)
        self.assertEqual(err.getvalue().count("НЕ ВНЕДРЕНА"), 1,
                         "предупреждение печатается на каждое слово — через "
                         "минуту его перестанут читать, а значит оно молчит")

    def test_installation_silences_the_warning(self):
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            install_service()
            R._lemma_simple("кошки")
        self.assertEqual(err.getvalue(), "",
                         "после внедрения службы предупреждение осталось — "
                         "оно перестаёт отличать настоящий отказ от мнимого")

    def test_state_is_reportable_as_number(self):
        install_service()
        self.assertTrue(R.lemma_service_state()["installed"])
        R.clear_lemma_service()
        self.assertFalse(R.lemma_service_state()["installed"])
        install_service()   # вернуть как было: состояние-то общее на процесс


class TestPosDegradationIsLoud(unittest.TestCase):
    """
    То же требование к НОВОЙ службе (A2.2 запер его для сведения форм).

    Без замка деградация могла оказаться либо молчаливой (главный страх A2),
    либо вечно орущей — и то и другое стоит одинаково дорого: крик, который
    звучит всегда, не отличает настоящий отказ от рабочего процесса.
    """

    def setUp(self):
        self.saved = A.pos_service_state()["installed"]
        A.clear_pos_service()

    def tearDown(self):
        # Состояние общее на процесс: не вернёшь — поедут все последующие
        # файлы, позеленев на справочнике вместо морфологии (урок 32).
        if self.saved:
            install_service()
        else:
            A.clear_pos_service()

    def _noise(self, fn):
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = fn()
        return out, err.getvalue()

    def test_missing_service_is_announced_when_it_matters(self):
        out, noise = self._noise(lambda: A.looks_like_verb("чинить"))
        self.assertFalse(out, "без распознавателя слово вне справочника не "
                              "должно называться глаголом — но обязан быть шум")
        self.assertIn("НЕ ВНЕДРЕНА", noise)
        self.assertIn("set_pos_service", noise,
                      "предупреждение не говорит, как починить — значит его "
                      "проигнорируют")

    def test_reference_table_and_reflexive_do_NOT_noise(self):
        """
        Если справочника хватает — молчать.

        «гасить» есть в ACTION_PROFILES, «катиться» узнаётся по -ся: ни там ни
        там морфология не запрашивается. Шуметь на таких словах — обесценить
        сигнал: отличить настоящий отказ от обычного вызова станет нельзя.
        """
        out, noise = self._noise(
            lambda: (A.looks_like_verb("гасить"), A.looks_like_verb("катиться")))
        self.assertEqual(out, (True, True),
                         "без службы справочник и -ся обязаны работать — иначе "
                         "отказ ловится раньше, чем начинается деградация")
        self.assertEqual(noise, "", "шум там, где служба не нужна")

    def test_warning_is_not_repeated(self):
        _, noise = self._noise(
            lambda: [A.looks_like_verb(w)
                     for w in ("чинить", "опылять", "мнёт", "гасить")])
        self.assertEqual(noise.count("НЕ ВНЕДРЕНА"), 1,
                         "предупреждение на каждое слово — через минуту его "
                         "не читают, а значит оно молчит")

    def test_installation_silences_the_warning(self):
        _, noise = self._noise(lambda: (install_service(),
                                        A.looks_like_verb("чинить")))
        self.assertEqual(noise, "",
                         "после внедрения шум остался — он перестаёт отличать "
                         "настоящий отказ от мнимого")
        self.assertTrue(A.looks_like_verb("чинить"),
                        "со службой остаточный глагол обязан стать глаголом")


class TestProviderIsEquivalent(unittest.TestCase):
    """
    Замена движка обязана быть равной старой, а не «похожей».

    Эвристика отличается ОТВЕТОМ, и тесты на паре слов это не поймают: нужен
    замер на настоящих словах проекта. Слова ниже — из живого data/memory/
    и корпуса, плюс известные провалы отсечения (сеть/сети, кошка/кошачьи).
    """

    def setUp(self):
        # Состояние службы ОБЩЕЕ НА ПРОЦЕСС. Этот класс ставит сломанный
        # лемматизатор и снимает службу; без восстановления весь набор после
        # него едет на эвристике (ЗАМЕРЕНО на полном прогоне: 9 падений в
        # test_records_contract и test_memory_isolation_contract, которых у
        # файла поодиночке нет). Тест, отравивший процесс, хуже падающего
        # теста: он меняет чужие результаты, оставаясь зелёным.
        self.saved = R.lemma_service_state()["installed"]

    def tearDown(self):
        if self.saved:
            install_service()
        else:                                        # pragma: no cover
            R.clear_lemma_service()

    WORDS = ("сеть", "сети", "сетей", "кошка", "кошки", "кошачьи",
             "пчела", "пчёлы", "бегемот", "гиппопотам", "флюгер", "флюгера",
             "ввод", "молока", "собаке", "человек", "бегущего", "красивой")

    def test_same_word_agrees_with_the_old_behavior(self):
        install_service()
        pairs_same = [("кошки", "кошка"), ("сети", "сеть"),
                      ("флюгера", "флюгер"), ("собаке", "собака")]
        pairs_diff = [("кошка", "кошачьи"), ("бегемот", "гиппопотам"),
                      ("сеть", "бегемот")]
        for a, b in pairs_same:
            with self.subTest(pair=(a, b)):
                self.assertTrue(R.same_word(a, b),
                                f"{a}/{b} перестали считаться одной формой — "
                                f"тавтология начнёт попадать в мир")
        for a, b in pairs_diff:
            with self.subTest(pair=(a, b)):
                self.assertFalse(R.same_word(a, b),
                                 f"{a}/{b} склеились — настоящую категорию "
                                 f"выбросят как «форму самого слова»")

    def test_heuristic_is_NOT_used_when_service_is_present(self):
        """
        Общий префикс службой запрещён. Эвристика («кошк» ~ «кошач» -> одно
        слово) при живой службе означала бы, что решение принимает не язык.
        """
        install_service()
        self.assertFalse(R.same_word("кошка", "кошачьи"))
        self.assertFalse(R.same_word("сеть", "сенокос"))

    def test_provider_equals_pymorphy3_normal_form(self):
        """
        ПРИМАЮЩИЙ ядро обязан получать ту же лемму, что давал pymorphy3.

        Это и есть настоящий замер эквивалентности (1048 слов, 0 расхождений —
        experiments/IR_BOUNDARY.md), а не обещание. Сверка ЧЕСТНАЯ: наличие
        движка узнаётся через find_spec БЕЗ импорта, а отсутствие pymorphy3 в
        машине обязан ловить другой тест — TestHeavyStackIsAbsent. Оставлять
        здесь skip значило бы позеленеть в точности тогда, когда проверять не
        с чем, то есть перестать что-либо утверждать молча.
        """
        from importlib.util import find_spec
        self.assertIsNotNone(find_spec("pymorphy3"),
                            "pymorphy3 исчез из окружения, а этот тест обязан "
                            "сравнивать со старым движком — проверьте окружение")
        import pymorphy3
        install_service()
        mp = pymorphy3.MorphAnalyzer()
        words = list(self.WORDS)
        # Живые слова проекта: то, на чём решение действительно принимается.
        import json
        for rel in ("data/memory/lipid_world.json",
                    "experiments/corpus_dev.json",
                    "experiments/corpus_heldout.json"):
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p):
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    blob = json.load(fh)
            except Exception:                        # pragma: no cover
                continue
            import re
            for tok in re.findall(r"[а-яёА-ЯЁ-]{4,}",
                                  json.dumps(blob, ensure_ascii=False)):
                words.append(tok.lower())
        seen, bad = set(), []
        for w in words:
            if w in seen or not w.strip("-"):
                continue
            seen.add(w)
            try:
                old = (mp.parse(w)[0].normal_form or "").lower()
            except Exception:                        # pragma: no cover
                continue
            new = R._lemma_simple(w)
            if new and new != old:
                bad.append((w, old, new))
        self.assertGreater(len(seen), 300,
                           f"сверяли всего {len(seen)} слов — это не замер, а "
                           f"надежда: подберите данные, иначе проверка пуста")
        self.assertEqual(bad, [],
                         f"расхождений службы со старым движком: {len(bad)} "
                         f"из {len(seen)}, первые: {bad[:5]}")

    def test_service_failure_degrades_to_empty_not_to_garbage(self):
        """
        Упавший лемматизатор обязан дать «не знаю» (""), а не вымышленную
        лемму: пустая лемма в same_word значит «не та же форма», вымышленная —
        что две разные формы станут одной и сотрут различие в мире.
        """
        def broken(word):
            raise RuntimeError("сломался")
        R.set_lemma_service(broken, name="тест.сломанный")
        import io
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = R._lemma_simple("кошки")
        self.assertEqual(out, "")
        self.assertIn("не разобрал", err.getvalue())


class TestLanguageStaysOutsideCore(unittest.TestCase):
    """
    Ядро не смеет импортировать pymorphy3 — даже «временно, до A3».

    Проверка AST, а не grep: grep видит «pymorphy3» в докстринге и слепнет к
    `import pymorphy3.unit`, а докстрингов здесь много и они объясняют историю.
    """

    def test_records_has_no_language_import(self):
        src = open(os.path.join(ROOT, "core", "records.py"),
                   encoding="utf-8").read()  # noqa: SIM115
        tree = ast.parse(src)
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        for banned in ("pymorphy3", "lang", "core.morph", "core.lemma",
                       "core.extract"):
            self.assertNotIn(banned, mods,
                             f"core/records снова знает язык: {banned}")
        # Не «список запрещённых», а список РАЗРЕШЁННЫХ: запрет перечисляет
        # то, о чём мы подумали, разрешение ловит то, о чём не подумали.
        self.assertLessEqual(mods, {"__future__", "re", "sys", "typing"},
                             f"records потянул постороннее: "
                             f"{sorted(mods - {'__future__', 're', 'sys', 'typing'})} — "
                             f"лёгкий канал знания обязан оставаться лёгким "
                             f"по факту, а не по надежде")

    def test_records_imports_without_heavy_stack(self):
        """
        Запрет должен действовать и как факт, а не только как AST: на машине
        без pymorphy3 импорт records обязан пройти и остаться полезным.
        """
        probe = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "BLOCK = ('pymorphy3', 'torch', 'numpy', 'scipy',\n"
            "         'sentence_transformers')\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in BLOCK:\n"
            "            raise ImportError('заблокировано тестом: ' + name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "import core.records as R\n"
            "print('ИМПОРТ ok', R.lemma_service_state()['installed'])\n"
            % ROOT)
        res = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                             env=dict(os.environ, LIPID2_STATUS="0"),
                             capture_output=True, text=True, timeout=900)
        self.assertIn("ИМПОРТ ok", res.stdout,
                      f"records не импортируется без тяжёлого стека: "
                      f"{res.stderr[-500:]}")
        self.assertIn("ИМПОРТ ok False", res.stdout,
                      "без тяжёлого стека records сам себе внедрил язык — "
                      "это возврат шва внутрь ядра")

    def test_provider_carries_the_russian(self):
        """Служба-носитель обязана жить в lang/, иначе смысл шага потерян."""
        path = os.path.join(ROOT, "lang", "ru", "lemma.py")
        self.assertTrue(os.path.exists(path),
                        "носитель службы исчез из lang/ — куда тогда переехал "
                        "русский?")

    def test_no_TEST_silently_runs_on_heuristic(self):
        """
        Тот же запрет, что для производства, но для tests/ и других папок.

        Почему проверкой, а не списком ниже. Три файла набора после A2.2
        МОЛЧА поехали на эвристике (замер: test_records_contract,
        test_hierarchy_determinism_contract, test_teacher_prompt_contract) —
        они зовут морфозависимые функции, не импортируя ни brain, ни службу.
        Позеленели. Тест, который проверяет на эвристике, думая что проверяет
        на языке, хуже отсутствующего: он рисует зелёное покрытие там, где
        ничего не сверяется.

        СПОСОБ — рантайм-зонд, а не разбор импортов. Разбор импортов дал 6
        «виновников» вместо 3 настоящих: файл, который импортирует
        brain.learner, получает службу ОТ Него и вполне честен — язык знает
        разрешённый слой. Отличить «мне принесли» от «мне не принесли» может
        только состояние процесса после импорта, поэтому спрашиваем у процесса.
        """
        offenders, checked = [], 0
        folders = ("tests", "brain", "tools", "teachers", "ui")
        for folder in folders:
            base = os.path.join(ROOT, folder)
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in sorted(filenames):
                    if not fn.endswith(".py"):
                        continue
                    rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
                    if rel == os.path.join("tests",
                                           "test_lemma_service_contract.py"):
                        continue        # этот файл сам и есть служба
                    src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
                    tree = ast.parse(src)
                    used = set()
                    for n in ast.walk(tree):
                        if isinstance(n, ast.Call):
                            if isinstance(n.func, ast.Name):
                                used.add(n.func.id)
                            elif isinstance(n.func, ast.Attribute):
                                # R.same_word(...) — вызов через модульный
                                # алиас: по одному имени функции он не виден,
                                # а запрет обязан ловить и такую запись
                                used.add(n.func.attr)
                        elif isinstance(n, ast.ImportFrom) and n.module == "core.records":
                            used |= {a.name for a in n.names}
                    if not (used & MORPH_DEPENDENT):
                        continue
                    if "lang.ru.lemma" in src or "set_lemma_service" in src:
                        continue        # внёс сам, явно — это и есть починка
                    checked += 1
                    mod = rel[:-3].replace(os.sep, ".")
                    try:
                        st = state_after_import(mod)
                    except AssertionError as exc:
                        offenders.append(f"{rel} (не импортируется: {exc})")
                        continue
                    if not st["installed"]:
                        offenders.append(rel)
        self.assertGreater(checked, 0,
                           "ни одного файла не проверили — запрет мерит пустоту")
        self.assertEqual(offenders, [],
                         f"сводят формы слов БЕЗ службы — проверяют на "
                         f"эвристике, полагая что проверяют на языке: "
                         f"{offenders}. Починка: from lang.ru.lemma import "
                         f"install; install()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
