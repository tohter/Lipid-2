"""
core/cortex.py — предиктивный цикл, гомеостаз и гормоны.

ЧТО ЗДЕСЬ ИЗМЕНИЛОСЬ И ПОЧЕМУ (замеры 2026-07-13, полный протокол —
tests/test_cortex_experience_contract.py)
--------------------------------------------------------------------------------
Цикл был написан корректно как СТЕД и жил в legacy/. Попытка подключить его к
живому мозгу «как есть» дала четыре измеренных дефекта:

 A. ПРЕДСКАЗАНИЕ НЕ ПРО СЛЕДУЮЩЕЕ СЛОВО ОПЫТА.
    _predict брал самый тяжёлый синапс, а у узла из Вселенной самый тяжёлый —
    ВСЕГДА IS_A (180 против CAN_DO 150). ЗАМЕРЕНО:
        _predict(«кошка») -> «кошачьи»
    Цикл предсказывал категорию концепта, то есть на фразе «кошка сидит на
    подоконнике» угадывал «кошачьи» вместо «подоконник». Он не ошибался — он
    отвечал на другой вопрос.

 B. ХИТ БЫЛ МЯГЧЕ, ЧЕМ ВЫГЛЯДЕЛО.
    hit = overlap(pred, actual) >= 13 ИЛИ имя==факт. При шумовом поле ~10 и
    форме того же слова 18-19 «попаданием» считалось «предсказал кошку, пришла
    кошка в другом падеже» и даже «огонь -> пламя» (overlap 17). ЗАМЕРЕНО на 9
    controlных парах: мягкий критерий даёт 4 «попадания», строгий — 1.
    В отчёте это выглядело как 59.6% точности там, где строго 0 угаданных
    продолжений. Пороги 13/17 при этом НЕ отменяются: они про РАСПОЗНАВАНИЕ
    входа (замер на 15 000 слов), а не про сверку предсказания с фактом.
    РАЗНЫЕ ОПЕРАЦИИ — РАЗНЫЕ МЕРЫ. Мягкая мера осталась, но называется
    отдельно (soft_overlap_hits) и в точность не входит.

 C. МУСОР В ГРАФЕ.
    Поток токенизировался посимвольно и каждый промах порождал
    get_or_create_node(actual): после 20 предложений из 14 узлов стало 43,
    включая сущности «а», «и», «на», «когда», «что», а «кошка» и «кошки»
    стали ДВУМЯ узлами (дубли). Ныне: вход лемматизируется и фильтруется тем же
    стоп-листом core/morph, что и экстрактор; цикл НЕ имеет права создавать
    узлы графа — он читает и считает.

 D. ОПЫТ ИСЧЕЗАЛ ПРИ ПЕРЕЗАПУСКЕ.
    _on_hit/_on_miss писали в out_synapses_typed, persist для них был
    save_to_disk() -> lipid_db.json, а мозг собирает граф из
    lipid_world.json. Итог: наработанное STDP обнулялось при закрытии окна.
    Ныне опыт живёт в core/transitions.py и сохраняется секцией «transitions»
    в Единой Вселенной (переживает перезапуск, переносим в C).

 E. ГЛАВНОЕ: ЦИКЛ ЗАТИРАЛ УВЕРЕННОСТЬ ОНТОЛОГИИ.
    Тот же weight, что learner хранит как УВЕРЕННОСТЬ в факте (90+130*conf,
    правило «вес никогда не понижаем»), цикл трактовал как ЧАСТОТУ ПЕРЕХОДА.
    ЗАМЕРЕНО на «кошка -> кошачьи» (IS_A, вес 180 = онтология из Вселенной):
        один промах: 180 -> 116      одно попадание: 116 -> 231
    После этого по весу нельзя отличить «факт сомнительный» от «следующее слово
    редко угадывали». Ныне у цикла ЕСТЬ СВОЁ ЧИСЛО (сила перехода 0..15), и к
    weight у него ДОСТУПА НЕТ — ни на чтение в предсказании, ни на запись.

 ЧЕМ ЦИКЛ ПОСЛЕ ЭТОГО ЯВЛЯЕТСЯ: наблюдателем за последовательностью опыта,
 который считает ожидания и ошибается вслух числами. Он НЕ добывает знания:
 bootstrap неизвестного — работа Learner'а и трёх каналов с бюджетами,
 лимитами и статусами asked_empty/not_asked. Прежний _ask_teacher был
 duck-typing'ом поверх учителей: он не трогал channel_calls (значит, его работу
 было НЕЧЕМ проверить — тот же дефект, что чинили в срезе 312040b), не уважал
 переключатели каналов и писал integrate_knowledge(weight=180) в обход
 merge_entity, то есть перезаписывал известные записи. Путь удалён; если в цикл
 передают учителя, он КРИЧИТ, а не молча игнорирует.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import List, Optional, Tuple

# Та же конвенция, что в core/graph.py: модуль можно запустить как скрипт
# (python core/cortex.py) — ниже self-check, полезный без тестового раннера.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sdr import SDR1024
from core.graph import GraphMemory, Node
from core.transitions import TransitionTable


def _complain(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")
    sys.stderr.flush()


# ─────────────────────────────────────────────────────────────
# ГОМЕОСТАЗ И ГОРМОНЫ (перенесено из архивного монолита в legacy/)
# ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Homeostasis:
    predictability: int = 200  # 0..255 — насколько мир предсказуем


@dataclasses.dataclass
class Hormones:
    dopamine: int = 0            # закрепление успеха
    norepinephrine: int = 0      # удивление, пластичность
    cortisol: int = 0            # стресс ошибки

    def decay(self):
        # ДЕКАЙТ ПОСЛЕДНИМ и только у владельца шкалы. Мозг передаёт циклу СВОИ
        # гормоны (GUI показывает одну шкалу, а не две расходящиеся), и если бы
        # каждый цикл гасил чужие значения на своём тике, «дофамин» в окне
        # перестал бы означать что-либо понятное: его гасили бы два writers
        # с разной частотой.
        self.dopamine = max(0, self.dopamine - 40)
        self.norepinephrine = max(0, self.norepinephrine - 60)
        self.cortisol = max(0, self.cortisol - 30)

    def as_dict(self) -> dict:
        return {"dopamine": self.dopamine, "norepinephrine": self.norepinephrine,
                "cortisol": self.cortisol}


def _empty_stats() -> dict:
    """Нулевой набор счётчиков. ОДНО место: два набора (сеанс и жизнь)
    обязаны иметь ОДИНАКОВЫЕ ключи, иначе lifetime тихо отстанет по полю
    и сравнить их станет нельзя."""
    return {
        "ticks": 0,
        "hits": 0,               # СТРОГОЕ совпадение предсказания с фактом
        "soft_overlap_hits": 0,  # близко по битам, но НЕ то слово (метрика)
        "misses": 0,             # было ожидание и оно не сбылось
        "no_expectation": 0,     # опыта нет — это НЕ ошибка, см. observe
        "surprises": 0,          # реальность оказалась словом вне ожиданий
        "unknown_input": 0,      # распознан не был текущий концепт
        "skipped": 0,            # служебное слово: тика не было
        "new_transitions": 0,
    }


# ─────────────────────────────────────────────────────────────
# ПРЕДИКТИВНОЕ КОРТИКАЛЬНОЕ ЯДРО
# ─────────────────────────────────────────────────────────────

class PredictiveCortex:
    """
    Петля «восприятие -> предсказание -> ошибка -> пластичность» над ТАБЛИЦЕЙ
    ПЕРЕХОДОВ, а не над весами онтологии (см. docstring модуля, пункты A/E).

    ОТВЕТСТВЕННОСТЬ:
      * предсказать продолжение по накопленному опыту;
      * измерить ошибку двумя честными мерами (строгая и мягкая);
      * обновить СИЛУ ПЕРЕХОДА (единственное, что цикл имеет право менять);
      * ничего не записывать во Вселенную и в граф.

    ПРАВА ЗАПИСИ: только transitions + собственные гормоны/статистика.
    """

    # ── ПОРОГИ (все — из замеров, experiments/RESULTS.md) ─────
    #
    # Шумовой пол живого языка ~10 бит из 20 и НЕ растёт с размером графа
    # (500 -> 15 000 узлов, experiments/final_eval.py). Цена ошибки асимметрична:
    #          t     FP      FN
    #         12    11.1%    0.0%
    #         13     0.0%   25.0%
    # Держим FP = 0: ложное «узнал» = Учитель не позван = ложь в графе,
    # исправлять нечем; пропуск стоит один лишний вызов LLM.
    UNKNOWN_CEILING = 13   # ниже — точно неизвестен
    KNOWN_FLOOR = 17       # выше — точно известен (своя форма слова)

    # мягкая МЕРА близости предсказания (не порог решения!): overlap бита в бит.
    SOFT_OVERLAP = 13

    def __init__(self, graph: GraphMemory, teacher=None,
                 transitions: Optional[TransitionTable] = None,
                 hormones: Optional[Hormones] = None):
        """
        teacher: параметр оставлен ТОЛЬКО чтобы не молчать. У цикла нет права
        добывать знания — у Learner'а для этого есть бюджеты, лимиты и статусы.
        Передать учителя = получить предупреждение в stderr, а не тихий обход
        правил записи.
        """
        self.graph = graph
        self.transitions = transitions if transitions is not None else TransitionTable()
        self.homeostasis = Homeostasis()
        # Мозг передаёт СВОИ гормоны: две независимые шкалы «дофамина» в одном
        # процессе — это гарантия, что GUI показывает не то, что происходит.
        self.hormones = hormones if hormones is not None else Hormones()
        self.tick_count = 0
        # ДВЕ пары счётчиков, и это не украшение.
        #   session  — числа ТЕКУЩЕГО потока (обнуляются в process_stream)
        #   lifetime — всё, что цикл видел за жизнь процесса
        # Раньше счётчик был ОДИН и пожизненный: отчёт второго урока печатал
        # сумму первого и второго. Ровно тот дефект, что уже чинили с
        # channel_calls в мозгу: «угадано 18» после третьего урока означает
        # «неизвестно что», а не «на этом тексте 18». lifetime оставлен
        # намеренно — накопленная точность по всей жизни тоже полезна, но
        # обязана называть СВОЁ имя.
        self.stats = _empty_stats()
        self.lifetime = _empty_stats()
        self._noise_floor = 0
        self._floor_stamps = 0
        self._last: dict = {}
        # КТО гасит гормоны: владелец шкалы (цикл один) или хозяин (мозг).
        self._owns_hormones = hormones is None
        if teacher is not None:
            _complain("[cortex] ⛔ PredictiveCortex НЕ принимает Учителя: добывать "
                      "знание — задача Learner'а (бюджет, лимиты, статусы "
                      "asked_empty/not_asked, запись через merge_entity). "
                      "Учитель ПРОИГНОРИРОВАН циклом, а не подключён.")

    # ── ПУБЛИЧНЫЕ МЕТОДЫ ─────────────────────────────────────

    def _bump(self, key: str) -> None:
        """
        Единственная точка, где счётчик увеличивается.

        Оба набора (сеанс и жизнь) правятся ЗДЕСЬ, а не в двух местах внутри
        observe(): расхождение между ними — тот же тихий дефект, что «session
        забыли обнулить». Ключ обязано видеть _empty_stats(): опечатка обязана
        упасть, а не завести новую счётную корзинку из ниоткуда.
        """
        if key not in self.stats:
            raise KeyError(f"неизвестный счётчик {key!r}; известны: "
                           f"{sorted(self.stats)}")
        self.stats[key] += 1
        self.lifetime[key] += 1

    def reset_session(self) -> None:
        """Начать новый сеанс: числа ТЕКУЩЕГО потока обнуляются, накопленное
        за жизнь процесса остаётся в self.lifetime."""
        self.stats = _empty_stats()

    def observe(self, current: str, following: str, verbose: bool = False) -> dict:
        """
        Один тик: чем закончилось «{current} ...» при наблюдении {following}.

        Ожидается, что ОБА слова уже приведены к лемме/имени узла вызывающим
        (см. process_stream и brain.experience_from_text). Здесь нет
        get_or_create_node: право создавать узлы у того, кто понимает, что он
        создаёт — у Learner'а и каналов знания.
        """
        current = (current or "").lower().strip()
        following = (following or "").lower().strip()
        if not current or not following or current == following:
            # current == following — не «успех предсказания», а отсутствие
            # наблюдения: цикл не знает слов. Считать это попаданием = рисовать
            # точность из воздуха.
            self._bump("skipped")
            self._last = {"skipped": True, "reason": "пустой или повторный тик"}
            return self._last

        self.tick_count += 1
        self._bump("ticks")
        self._floor_stamps += 1
        if self._floor_stamps % 64 == 1:
            self.recalibrate_floor()

        if not self._is_known(current):
            self._bump("unknown_input")
            self.hormones.norepinephrine = min(255, self.hormones.norepinephrine + 160)
            if verbose:
                print(f"── «{current}»: не распознан (шумовой пол {self._noise_floor}) "
                      f"— опыта не накапливаем")

        predicted = self.transitions.top_name(current)
        strength = self.transitions.get(current, following)
        expected_strength = self.transitions.get(current, predicted) if predicted else 0

        strict_hit = predicted is not None and predicted == following
        soft_hit = (predicted is not None
                    and SDR1024(predicted).overlap(SDR1024(following)) >= self.SOFT_OVERLAP)

        # ── ПЛАСТИЧНОСТЬ: только сила перехода. weight онтологии не трогаем.
        if predicted is None:
            self._bump("no_expectation")
            self.transitions.confirm(current, following)
            self._bump("new_transitions")
            self.hormones.norepinephrine = min(255, self.hormones.norepinephrine + 40)
            outcome = "no_expectation"
        elif strict_hit:
            self._bump("hits")
            self.transitions.confirm(current, following)
            self.homeostasis.predictability = min(255, self.homeostasis.predictability + 15)
            self.hormones.dopamine = min(255, self.hormones.dopamine + 120)
            outcome = "hit"
        else:
            self._bump("misses")
            if soft_hit:
                self._bump("soft_overlap_hits")
            # Ожидание было и не сбылось: оно слабеет. Реальность записывается
            # как новое ожидание — иначе цикл никогда не научится.
            self.transitions.surprise(current, predicted)
            if self.transitions.get(current, following) == 0:
                self._bump("surprises")
                self._bump("new_transitions")
            self.transitions.confirm(current, following)
            self.hormones.norepinephrine = min(255, self.hormones.norepinephrine + 120)
            self.hormones.cortisol = min(255, self.hormones.cortisol + 100)
            self.homeostasis.predictability = max(0, self.homeostasis.predictability - 25)
            outcome = "miss"

        if self._owns_hormones:
            # Шкала принадлежит циклу -> он её и гасит. Если шкалу одолжил мозг,
            # decay() вызывает ОН: иначе два владельца гасили бы значения с
            # разной частотой и «дофамин» в GUI стал бы нечитаемым.
            self.hormones.decay()
        self._last = {
            "current": current, "predicted": predicted, "actual": following,
            "outcome": outcome, "strict_hit": strict_hit, "soft_hit": soft_hit,
            "strength_before": strength, "expected_strength": expected_strength,
            "strength_after": self.transitions.get(current, following),
        }
        if verbose:
            mark = {"hit": "✅", "miss": "💥", "no_expectation": "🌱"}[outcome]
            print(f"── «{current}» → ждал «{predicted or '?'}», факт «{following}» "
                  f"{mark} (сила {self._last['strength_after']})")
        return self._last

    def process_stream(self, text: str, verbose: bool = True) -> dict:
        """
        Поток опыта: соседние значимые слова ВНУТРИ предложения.

        Граница предложения — РАЗРЫВ опыта, а не соседство. Первый вариант
        склеивал весь текст в одну цепочку, и на «…на подоконнике. Кошка спит.»
        возникал переход «подоконник -> кошка»: ЗАМЕРЕНО (ticks=7 вместо 5 на
        трёх предложениях). Иначе цикл выучал бы порядок предложений документа,
        а не структуру мира.

        Лемматизация и стоп-фильтр — здесь (это вход цикла), а не в observe():
        без них цикл плодил узлы и считал предлог «на» концептом (дефект C).

        СЧЁТЧИКИ СЕАНСА ОБНУЛЯЮТСЯ ЗДЕСЬ: возвращаемые метрики описывают
        ИМЕННО этот текст. До этого они копились с запуска процесса, и отчёт
        второго урока печатал сумму первого и второго. lifetime остаётся
        пожизненным и доступен отдельным ключом metrics()["lifetime"].
        """
        self.reset_session()
        sentences = [s for s in self.experience_sentences(text) if len(s) >= 2]
        if not sentences:
            print("⚠️  Нет ни одного предложения из двух значимых слов — "
                  "предсказывать нечего")
            return self.metrics()
        ticks = sum(len(s) - 1 for s in sentences)
        if verbose:
            print(f"\n{'=' * 60}\n📡 ПОТОК: {len(sentences)} предложений, "
                  f"{ticks} тиков опыта\n{'=' * 60}")
        for words in sentences:
            for i in range(len(words) - 1):
                self.observe(words[i], words[i + 1], verbose=verbose)
        if verbose:
            self.print_xray()
        return self.metrics()

    def experience_sentences(self, text: str) -> List[List[str]]:
        """
        Текст -> по предложениям: цепочки лемм значимых слов.

        Служебные слова (предлоги, союзы, местоимения, глаголы-паразиты)
        выбрасываются ТЕМ ЖЕ стоп-листом, что у экстрактора: цикл и экстрактор
        обязаны видеть ОДИН и тот же лексикон, иначе их отчёты про один текст не
        сходятся, а сравнить их — единственный способ проверить оба.
        """
        from core.morph import Morph, split_sentences
        out: List[List[str]] = []
        for sentence in split_sentences(text):
            chain: List[str] = []
            for tok in _analyze(sentence):
                if not tok["content"]:
                    continue
                lem = (tok["lemma"] or tok["word"]).lower().strip()
                if not lem or Morph.is_stop(lem):
                    continue
                chain.append(lem)
            if chain:
                out.append(chain)
        return out

    def experience_stream(self, text: str) -> List[str]:
        """
        Плоская версия потока — для диагностики и отчётов. НЕ для обучения:
        она теряет границы предложений (см. process_stream).
        """
        return [w for chain in self.experience_sentences(text) for w in chain]

    def predict(self, word: str) -> Optional[str]:
        """Публичное предсказание: чего цикл ждёт после слова. Ничего не пишет."""
        return self.transitions.top_name((word or "").lower().strip())

    def query(self, word: str) -> None:
        """Что известно о концепте — БЕЗ обучения и без записи."""
        word = (word or "").lower().strip()
        input_sdr = SDR1024(word)
        best_node, overlap = self._recall(input_sdr)
        thr = self.unknown_threshold()
        verdict = "УЗНАН" if overlap >= thr else "НЕИЗВЕСТЕН"
        print(f"\n── Запрос: «{word}»")
        print(f"   SDR-перекрытие с лучшим узлом: {overlap}/{SDR1024.TARGET_ON_BITS}"
              f" (порог {thr}, шумовой пол графа {self._noise_floor}) → {verdict}")
        top = self.transitions.top(word, 3)
        if top:
            print("   Ожидания по опыту (сила 0..15): " +
                  ", ".join(f"«{t}» {s}" for t, s in top))
        else:
            print("   Ожиданий нет: этот контекст ещё не наблюдался")
        matches = self.graph.recall_by_sdr(input_sdr, top_k=5)
        if matches:
            print("   SDR-резонанс:")
            for name, ov, sim in matches:
                bar = "█" * int(sim * 10) + "░" * (10 - int(sim * 10))
                print(f"   • [{name:15s}] {bar} {ov}/20 ({sim * 100:.0f}%)")
        self.graph.print_node_associations(word)

    def metrics(self) -> dict:
        """
        Публичные ЧИСЛА ТЕКУЩЕГО ПОТОКА. ДВЕ точности и объяснение, почему они
        разные: одна цифра «точность» уже вела проект к самообману (дефект B).

        «Текущего потока» — не «за жизнь процесса»: см. self.stats/self.lifetime.
        Пожизненные числа лежат в metrics()["lifetime"], а не под теми же
        именами, чтобы нельзя было перепутать «на этом уроке» и «когда-либо».
        """
        ticks = self.stats["ticks"]
        decided = self.stats["hits"] + self.stats["misses"]
        return {
            **{k: v for k, v in self.stats.items()},
            # строгая: угаданное ПРОДОЛЖЕНИЕ (имя в имя)
            "accuracy_strict": (self.stats["hits"] / ticks) if ticks else 0.0,
            # доля тиков, где было ЧТО проверять (без неё «0 ошибок» означает
            # «опыта нет», а не «ни разу не ошиблись»)
            "coverage": (decided / ticks) if ticks else 0.0,
            "accuracy_when_expected": (self.stats["hits"] / decided) if decided else 0.0,
            "transitions": self.transitions.size(),
            "predictability": self.homeostasis.predictability,
            "hormones": self.hormones.as_dict(),
            "noise_floor": self._noise_floor,
            "hormones_shared_with_brain": not self._owns_hormones,
            # Сумма за жизнь процесса. Названа СВОИМ ключом: смешанная с
            # текущими числами, она превращала «на этом уроке» в «за всё время».
            "lifetime": {k: v for k, v in self.lifetime.items()},
        }

    def print_xray(self):
        m = self.metrics()

        def bar(v):
            return "█" * (v // 25) + "░" * (10 - v // 25) + f" ({v:3d}/255)"

        print("\n┌" + "─" * 56 + "┐")
        print("│ 🧬 СОСТОЯНИЕ ЯДРА                                        │")
        print("├" + "─" * 56 + "┤")
        print(f"│ Предсказуемость : {bar(m['predictability'])}    │")
        print(f"│ Дофамин         : {bar(m['hormones']['dopamine'])}    │")
        print(f"│ Норадреналин    : {bar(m['hormones']['norepinephrine'])}    │")
        print(f"│ Кортизол        : {bar(m['hormones']['cortisol'])}    │")
        print("├" + "─" * 56 + "┤")
        print(f"│ Тиков: {m['ticks']:<6} строгая точн.: {m['accuracy_strict'] * 100:5.1f}%  "
              f"переходов: {m['transitions']:<5} │")
        print(f"│ Попаданий: {m['hits']:<5} Ошибок: {m['misses']:<5} "
              f"без ожидания: {m['no_expectation']:<5}     │")
        print(f"│ Близко по битам, но не то слово: {m['soft_overlap_hits']}   "
              f"покрытие: {m['coverage'] * 100:4.0f}%│")
        print("└" + "─" * 56 + "┘")

    # ── ВНУТРЕННИЕ МЕХАНИЗМЫ ─────────────────────────────────

    def _is_known(self, word: str) -> bool:
        """Распознан ли концепт (точное имя или SDR выше адаптивного порога)."""
        if word in self.graph.nodes:
            return True
        _node, overlap = self._recall(SDR1024(word))
        return overlap >= self.unknown_threshold()

    def _recall(self, sdr: SDR1024) -> Tuple[Optional[Node], int]:
        """
        Лучший узел по SDR-перекрытию через индекс (матрица, не Python-цикл).
        ЗДЕСЬ ЭТО ТОЛЬКО ДИАГНОСТИКА распознавания входа: предсказание больше
        не ходит в граф (дефект A: самый тяжёлый синапс — это категория).
        """
        key, ov, _second = self.graph.index.best_and_second(sdr)
        if key is None:
            return None, 0
        return self.graph.nodes.get(key), ov

    # ── шумовой пол и порог неизвестного ────────────────────
    def recalibrate_floor(self) -> int:
        """
        Шумовой пол графа: насколько типичное слово графа пересекается со своим
        ближайшим соседом. Меряется САМИМ ГРАФОМ, а не текущим входом: первый
        вариант брал перекрытие входа, и узнавшее слово (19-20 бит) уносило
        «пол» за потолок — следующее незнакомое слово объявлялось известным.
        Замер на 15 000 русских слов: пол ~10 и почти не растёт со словарём.
        """
        floor = self.graph.index.self_probe_floor()
        if floor.get("n"):
            self._noise_floor = int(floor["median"])
        return self._noise_floor

    def unknown_threshold(self) -> int:
        """Пол шума + запас, зажатый между 13 и 17. Вывод — в experiments/."""
        return max(self.UNKNOWN_CEILING,
                   min(self.KNOWN_FLOOR, self._noise_floor + 4))


def _analyze(sentence: str) -> List[dict]:
    """
    Разбор предложения. Импорт локальный и один на все вызовы: цикл обязан
    запускаться там, где нет pymorphy3 (эвристика + available=False), а не
    падать на импорте.
    """
    from core.morph import analyze
    return analyze(sentence)


if __name__ == "__main__":
    # Быстрая самопроверка без тестового раннера: python core/cortex.py
    graph = GraphMemory()
    for w in ("кошка", "собака", "вода", "пламя", "огонь", "подоконник",
              "спать", "молоко"):
        graph.get_or_create_node(w)
    graph.sync_lemmas()
    nodes_before = set(graph.nodes)
    loop = PredictiveCortex(graph)
    text = ("Кошка сидит на подоконнике. Кошка пьёт молоко. "
            "Собака бегает во дворе. Кошка пьёт молоко.")
    loop.process_stream(text, verbose=True)
    m = loop.metrics()
    print("\nПосле цикла узел «пить» создан?", "пить" in graph.nodes,
          "| новых узлов:", set(graph.nodes) - nodes_before)
    print("ожидаемое продолжение после «кошка»:", loop.predict("кошка"))
    loop.query("кошка")
    print("\n⚠  ТИКИ:", m["ticks"], "СТРОГАЯ ТОЧНОСТЬ:",
          f"{m['accuracy_strict']:.0%}", "ПОКРЫТИЕ:", f"{m['coverage']:.0%}",
          "ПЕРЕХОДОВ:", m["transitions"])
