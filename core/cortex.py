import dataclasses
from typing import Optional, Tuple
from core.sdr import SDR1024
from core.graph import GraphMemory, Node
from core.storage import NodeType, RelationType


# ─────────────────────────────────────────────────────────────
# ГОМЕОСТАЗ И ГОРМОНЫ (перенесено из архивного монолита в legacy/)
# ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Homeostasis:
    predictability: int = 200  # 0..255 — насколько мир предсказуем

@dataclasses.dataclass
class Hormones:
    dopamine:       int = 0    # Закрепление успеха
    norepinephrine: int = 0    # Удивление, пластичность
    cortisol:       int = 0    # Стресс ошибки

    def decay(self):
        self.dopamine       = max(0, self.dopamine       - 40)
        self.norepinephrine = max(0, self.norepinephrine - 60)
        self.cortisol       = max(0, self.cortisol       - 30)


# ─────────────────────────────────────────────────────────────
# ПРЕДИКТИВНОЕ КОРТИКАЛЬНОЕ ЯДРО
# ─────────────────────────────────────────────────────────────

class PredictiveCortex:
    """
    Соединяет SDR (core/sdr.py) + Граф (core/graph.py) + Биохимию (этот модуль)
    в единую петлю: восприятие → предсказание → ошибка → пластичность.

    Учитель (LLMTeacher) вызывается ТОЛЬКО при полностью неизвестном концепте.
    После bootstrap мозг учится самостоятельно через поток опыта.
    """

    # ── ПОРОГИ ────────────────────────────────────────────────
    #
    # Старые значения (UNKNOWN=2, HIT=4) были выведены для буквенного хэша,
    # где ФОН = 0.68/20, а «кошка~кот» = 3/20. С эмбеддинговым кодировщиком
    # шкала перевернулась, и по замерам на 20 000 реальных русских слов
    # (experiments/final_eval.py):
    #
    #   несвязанные слова ............... 0-1 бит из 20
    #   сильный синоним (огонь/пламя) ... 17
    #   форма того же слова (кошка/кошке) 19
    #   ЛУЧШИЙ чужой узел при N=20 000 ... 8-10 (медиана), p99 = 16-17
    #
    # То есть 2 теперь означает «почти чужое», а 4 — «шум». Порог сдвинут
    # вверх, к шумовому полу живого языка.
    UNKNOWN_CEILING       = 13   # ниже — точно неизвестен, Учитель обязателен
    KNOWN_FLOOR           = 17   # выше — точно известен (своя форма слова)
    # Перекрытие, достаточное для «попадания» предсказания.
    HIT_THRESHOLD         = 13
    # Начальный вес нового синапса-гипотезы
    NEW_SYNAPSE_WEIGHT    = 60

    def __init__(self, graph: GraphMemory, teacher=None):
        self.graph       = graph
        self.teacher     = teacher       # LLMTeacher или None (автономный режим)
        self.homeostasis = Homeostasis()
        self.hormones    = Hormones()
        self.tick_count  = 0
        self.stats       = {"hits": 0, "misses": 0, "unknowns": 0,
                            "new_synapses": 0, "ambiguous": 0}
        # Оценка шумового пола графа пересчитывается порциями: она нужна, чтобы
        # «неизвестное» не зависело от размера словаря.
        self._noise_floor  = 0
        self._floor_stamps = 0
        self._last_second  = 0      # перекрытие с лучшим ЧУЖИМ (для диагностики)

    # ── ПУБЛИЧНЫЕ МЕТОДЫ ─────────────────────────────────────

    def process_stream(self, text: str, verbose: bool = True):
        """
        Обрабатывает текст как поток опыта.
        Каждая пара соседних слов (N, N+1) — один тик предиктивного цикла.
        """
        words = [w.strip(".,!?;:\"'()") for w in text.lower().split() if w.strip(".,!?;:\"'()")]
        if len(words) < 2:
            print("⚠️  Слишком мало токенов для обучения.")
            return

        print(f"\n{'='*60}")
        print(f"📡 ПОТОК: {len(words)} токенов, {len(words)-1} тиков")
        print(f"{'='*60}")

        for i in range(len(words) - 1):
            self._tick(
                input_word      = words[i],
                actual_next     = words[i + 1],
                verbose         = verbose
            )

        self.print_xray()

    def query(self, word: str) -> None:
        """Запрос что мозг знает о концепте — без обучения."""
        input_sdr  = SDR1024(word.lower())
        best_node, overlap = self._recall(input_sdr)
        thr = self.unknown_threshold()
        verdict = "УЗНАН" if overlap >= thr else "НЕИЗВЕСТЕН"

        print(f"\n── Запрос: «{word}»")
        print(f"   SDR-перекрытие с лучшим узлом: {overlap}/{SDR1024.TARGET_ON_BITS}"
              f" (порог {thr}, шумовой пол графа {self._noise_floor}) → {verdict}")

        # SDR-ассоциации по всему графу
        matches = self.graph.recall_by_sdr(input_sdr, top_k=5)
        if matches:
            print("   SDR-резонанс:")
            for name, ov, sim in matches:
                bar = "█" * int(sim * 10) + "░" * (10 - int(sim * 10))
                print(f"   • [{name:15s}] {bar} {ov}/20 ({sim*100:.0f}%)")

        # Типизированные связи из графа
        self.graph.print_node_associations(word)

    def print_xray(self):
        """Рентгеновский снимок внутреннего состояния."""
        def bar(v):
            return "█" * (v // 25) + "░" * (10 - v // 25) + f" ({v:3d}/255)"

        total = self.stats["hits"] + self.stats["misses"]
        acc   = (self.stats["hits"] / total * 100) if total > 0 else 0.0

        print("\n┌" + "─" * 56 + "┐")
        print("│ 🧬 СОСТОЯНИЕ ЯДРА                                        │")
        print("├" + "─" * 56 + "┤")
        print(f"│ Предсказуемость : {bar(self.homeostasis.predictability)}    │")
        print(f"│ Дофамин         : {bar(self.hormones.dopamine)}    │")
        print(f"│ Норадреналин    : {bar(self.hormones.norepinephrine)}    │")
        print(f"│ Кортизол        : {bar(self.hormones.cortisol)}    │")
        print("├" + "─" * 56 + "┤")
        print(f"│ Тиков: {self.tick_count:<6} Точность: {acc:5.1f}%  "
              f"Новых синапсов: {self.stats['new_synapses']:<5}  │")
        print(f"│ Попаданий: {self.stats['hits']:<5} Ошибок: {self.stats['misses']:<5} "
              f"Неизвестных: {self.stats['unknowns']:<5}      │")
        print("└" + "─" * 56 + "┘")

    # ── ВНУТРЕННИЕ МЕХАНИЗМЫ ─────────────────────────────────

    def _tick(self, input_word: str, actual_next: str, verbose: bool):
        """Один тик предиктивного цикла."""
        self.tick_count += 1
        # Пол шума — свойство графа, поэтому пересчитываем его редко и пачкой,
        # а не на каждом входе (иначе одно узнавшееся слово уносило бы порог за потолок).
        self._floor_stamps += 1
        if self._floor_stamps % 64 == 1:
            self.recalibrate_floor()
        input_sdr      = SDR1024(input_word)
        actual_next_sdr = SDR1024(actual_next)

        # 1. Распознавание текущего концепта
        best_node, best_overlap = self._recall(input_sdr)

        # Точное имя в графе — это узнавание сильнее любого SDR-шума.
        recognized_exactly = input_word in self.graph.nodes
        threshold = self.unknown_threshold()

        if best_overlap < threshold and not recognized_exactly:
            # Полностью неизвестен → bootstrap
            self.stats["unknowns"] += 1
            self.hormones.norepinephrine = min(255, self.hormones.norepinephrine + 160)
            if verbose:
                print(f"\n── #{self.tick_count} «{input_word}»: ❓ НЕИЗВЕСТЕН "
                      f"(overlap={best_overlap}, порог={threshold}, "
                      f"шум.пол={self._noise_floor})")
            best_node = self._bootstrap_unknown(input_word, verbose)
        else:
            if best_node is None:
                best_node = self.graph.get_or_create_node(input_word, NodeType.ENTITY)

        # 2. Предсказание следующего концепта (лучший синапс по весу)
        predicted_node, predicted_weight = self._predict(best_node)

        # 3. Сравнение предсказания с реальностью
        if predicted_node is not None:
            # Совпадение = либо точное имя, либо SDR-перекрытие выше порога.
            match_overlap = predicted_node.sdr.overlap(actual_next_sdr)
            hit = match_overlap >= self.HIT_THRESHOLD or predicted_node.name == actual_next
        else:
            match_overlap = 0
            hit = False

        if verbose:
            pred_str = f"«{predicted_node.name}»" if predicted_node else "?"
            mark = "✅" if hit else "💥"
            print(f"\n── #{self.tick_count} «{input_word}» → pred:{pred_str} "
                  f"real:«{actual_next}» {mark}")

        # 4. Пластичность — STDP
        if hit:
            self._on_hit(best_node, predicted_node, actual_next, match_overlap, verbose)
        else:
            self._on_miss(best_node, predicted_node, actual_next, verbose)

        self.hormones.decay()

    def _recall(self, sdr: SDR1024) -> Tuple[Optional[Node], int]:
        """
        Лучший узел по SDR-перекрытию через индекс (матрица, а не Python-цикл).
        Прежний вариант был O(N) тяжёлых операций на каждый тик и упирался
        в скорость раньше, чем в качество.
        """
        key, ov, second = self.graph.index.best_and_second(sdr)
        self._last_second = second
        if key is None:
            return None, 0
        return self.graph.nodes.get(key), ov

    # ── шумовой пол и порог неизвестного ────────────────────
    def recalibrate_floor(self):
        """
        Оценка шумового пола: НАСКОЛЬКО сильно типичное слово графа пересекается
        со своим ближайшим соседом. Меряется самими узлами графа, а не текущим
        входом.

        Почему не от входа: первый вариант брал перекрытие текущего слова. Если
        слово узнавалось (19-20 бит), «пол» выпрыгивал на 20, порог уезжал за
        потолок, и следующее незнакомое слово объявлялось известным. Пол — свойство
        ГРАФА, а не запроса.

        Замер на 15 000 настоящих русских слов (hunspell ru_RU):
            размер графа   медиана пола   лучший overlap у СЛОВА, КОТОРОГО НЕТ
                   500            10                    8
                 2 000             9                   11
                 8 000            10                   12
                15 000            10                   12
        Пол держится на ~10 битах и почти НЕ растёт со словарём. У буквенного
        хэша он рос лавинообразно — из-за этого фиксированный UNKNOWN_THRESHOLD=2
        молча превращался в «всё неизвестно» по мере роста графа.
        """
        floor = self.graph.index.self_probe_floor()
        if floor.get("n"):
            self._noise_floor = int(floor["median"])
        return self._noise_floor

    def unknown_threshold(self) -> int:
        """
        Порог неопределённости: пол шумов + запас, зажатый между потолком и
        нижним этажом.

        Запас +4 и порядок зажатки выбраны по замеру цены ошибки (t — порог,
        FP — незнакомое слово ложно объявлено известным, FN — форму слова в
        графе не узнали):
                 t     FP      FN
                11    22.2%    0.0%
                12    11.1%    0.0%
                13     0.0%   25.0%
                14     0.0%   25.0%
        Держим FP = 0. Асимметрия сознательная: ложное «узнал» означает, что
        Учитель НЕ вызван и в граф записано чужое знание — его нечем поправить.
        Промах по форме слова стоит только одного лишнего вызова Учителя.
        """
        return max(self.UNKNOWN_CEILING,
                   min(self.KNOWN_FLOOR, self._noise_floor + 4))

    def _predict(self, node: Optional[Node]) -> Tuple[Optional[Node], int]:
        """Лучший синапс по весу → предсказание следующего концепта."""
        if node is None or not node.out_synapses_typed:
            return None, 0

        best_target = None
        best_weight = -1
        for target_name, syn_info in node.out_synapses_typed.items():
            w = syn_info["weight"]
            if w > best_weight:
                best_weight = w
                best_target = target_name

        if best_target and best_target in self.graph.nodes:
            return self.graph.nodes[best_target], best_weight
        return None, 0

    def _on_hit(self, src_node: Node, pred_node: Node,
                actual_word: str, overlap: int, verbose: bool):
        """Совпадение — дофамин + STDP-усиление правильного синапса."""
        self.stats["hits"] += 1
        self.hormones.dopamine = min(255, self.hormones.dopamine + 120)
        self.homeostasis.predictability = min(255, self.homeostasis.predictability + 15)

        delta = max(10, self.hormones.dopamine >> 2)
        syn   = src_node.out_synapses_typed.get(pred_node.name)
        if syn:
            old_w         = syn["weight"]
            syn["weight"] = min(255, old_w + delta)
            if verbose:
                print(f"   📈 «{src_node.name}»→«{pred_node.name}»: {old_w}→{syn['weight']}")

    def _on_miss(self, src_node: Node, pred_node: Optional[Node],
                 actual_word: str, verbose: bool):
        """Ошибка — кортизол + ослабление неверного синапса + новая гипотеза."""
        self.stats["misses"] += 1
        self.hormones.norepinephrine = min(255, self.hormones.norepinephrine + 120)
        self.hormones.cortisol       = min(255, self.hormones.cortisol       + 100)
        self.homeostasis.predictability = max(0, self.homeostasis.predictability - 25)

        # Ослабляем только НЕВЕРНЫЙ предсказанный синапс
        if pred_node is not None:
            syn = src_node.out_synapses_typed.get(pred_node.name)
            if syn:
                penalty       = self.hormones.cortisol >> 1
                old_w         = syn["weight"]
                syn["weight"] = max(0, old_w - penalty)
                if verbose:
                    print(f"   📉 «{src_node.name}»→«{pred_node.name}»: {old_w}→{syn['weight']}")

        # Создаём или укрепляем синапс к реальному следующему слову
        target_node = self.graph.get_or_create_node(actual_word, NodeType.ENTITY)
        existing    = src_node.out_synapses_typed.get(actual_word)

        if existing:
            old_w           = existing["weight"]
            existing["weight"] = min(255, old_w + self.NEW_SYNAPSE_WEIGHT)
            if verbose:
                print(f"   💡 «{src_node.name}»→«{actual_word}»: {old_w}→{existing['weight']}")
        else:
            src_node.connect_to(target_node, rel_type=RelationType.CAUSES,
                                weight=self.NEW_SYNAPSE_WEIGHT)
            self.stats["new_synapses"] += 1
            if verbose:
                print(f"   💡 НОВЫЙ СИНАПС: «{src_node.name}»→«{actual_word}» "
                      f"[вес={self.NEW_SYNAPSE_WEIGHT}]")

    def _ask_teacher(self, word: str):
        """
        Duck-typed запрос к Учителю. Учителя в проекте РАЗНЫЕ по интерфейсу:
          - AgentTeacherV2  -> run_agent(base_concept, derived_concept, filename), ничего не возвращает
          - WikiTeacher     -> fetch_concept(word) -> {"base_entity": {...}, "derived_entity": {...}}
        Возвращает dict в формате GraphMemory.integrate_knowledge либо None.
        Никакой путь Учителя не должен ронять предиктивный цикл.
        """
        if self.teacher is None:
            return None

        raw = None
        try:
            for method in ("fetch_concept_properties", "fetch_concept", "run"):
                fn = getattr(self.teacher, method, None)
                if callable(fn):
                    raw = fn(word)
                    break
            else:
                # Учитель умеет только писать во Вселенную (AgentTeacherV2) —
                # дергаем run_agent и забираем результат из графа всезнания.
                run = getattr(self.teacher, "run_agent", None)
                if callable(run):
                    # Учитель с тех пор ВОЗВРАЩАЕТ записанное; чтение с диска
                    # осталось запасным путём для учителей, которые возвращают
                    # None, но знание всё равно оставляют во Вселенной.
                    raw = run(base_concept=word, derived_concept=word)
                    if raw is None:
                        raw = self._read_teacher_world(word)
        except Exception as exc:                       # сеть/LLM/парсинг — не роняем цикл
            print(f"   ⚠️  Учитель молчит ({type(exc).__name__}: {exc})")
            return None

        return self._normalize_teacher_knowledge(raw)

    def _read_teacher_world(self, word: str):
        """Вычитывает концепт из Единой Вселенной, если Учитель пишет только туда."""
        path = getattr(self.teacher, "world_filepath", None)
        if not path:
            return None
        try:
            import json, os
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                world = json.load(f)
            return world.get("entities", {}).get(word.lower().strip())
        except Exception:
            return None

    @staticmethod
    def _normalize_teacher_knowledge(raw):
        """
        Приводит ответы любого Учителя к одному виду:
            {"concept", "category", "properties", "actions"}
        Иначе integrate_knowledge проглатывает ответ молча.
        """
        if not isinstance(raw, dict):
            return None
        if "base_entity" in raw:                        # формат Вселенной (Wiki/LLM)
            raw = raw.get("base_entity") or {}
        if not raw or "concept" not in raw:
            return None

        hierarchy = raw.get("hierarchy") or []
        category = raw.get("category") or (hierarchy[0] if hierarchy else "")

        actions = []
        for act in raw.get("actions", []) or []:
            if isinstance(act, dict):
                actions.append(act.get("name", ""))
            elif isinstance(act, str):
                actions.append(act)

        return {
            "concept": str(raw["concept"]).lower().strip(),
            "category": str(category).lower().strip() if category else "",
            "properties": [str(p).lower().strip() for p in raw.get("properties", []) or [] if p],
            "actions": [a.lower().strip() for a in actions if a],
        }

    def _bootstrap_unknown(self, word: str, verbose: bool) -> Node:
        """
        Концепт полностью неизвестен.
        Учитель даёт начальные IS_A / HAS_PROPERTY — дальше учимся сами.
        """
        if self.teacher is not None:
            if verbose:
                print(f"   📡 Bootstrap: запрос к Учителю для «{word}»...")
            knowledge = self._ask_teacher(word)
            if knowledge:
                self.graph.integrate_knowledge(knowledge, weight=180)
                node = self.graph.nodes.get(word.lower())
                if node:
                    if verbose:
                        print(f"   📚 Учитель дал базу для «{word}»")
                    return node

        # Учителя нет или не ответил → пустой узел, опыт научит
        node = self.graph.get_or_create_node(word, NodeType.ENTITY)
        if verbose:
            print(f"   🆕 Пустой узел «{word}» (автономный режим)")
        return node
