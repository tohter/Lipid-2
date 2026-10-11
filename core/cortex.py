import dataclasses
from typing import Optional, Tuple
from core.sdr import SDR1024
from core.graph import GraphMemory, Node
from core.storage import NodeType, RelationType


# ─────────────────────────────────────────────────────────────
# ГОМЕОСТАЗ И ГОРМОНЫ (из kurwa.py, адаптировано под граф)
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
    Соединяет SDR (sdr.py) + Граф (graph.py) + Биохимию (kurwa.py)
    в единую петлю: восприятие → предсказание → ошибка → пластичность.

    Учитель (LLMTeacher) вызывается ТОЛЬКО при полностью неизвестном концепте.
    После bootstrap мозг учится самостоятельно через поток опыта.
    """

    # SDR-перекрытие (из 20 бит) ниже которого концепт считается неизвестным
    UNKNOWN_THRESHOLD     = 2
    # Перекрытие достаточное для «попадания» предсказания
    HIT_THRESHOLD         = 4
    # Начальный вес нового синапса-гипотезы
    NEW_SYNAPSE_WEIGHT    = 60

    def __init__(self, graph: GraphMemory, teacher=None):
        self.graph       = graph
        self.teacher     = teacher       # LLMTeacher или None (автономный режим)
        self.homeostasis = Homeostasis()
        self.hormones    = Hormones()
        self.tick_count  = 0
        self.stats       = {"hits": 0, "misses": 0, "unknowns": 0, "new_synapses": 0}

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

        print(f"\n── Запрос: «{word}» (SDR-перекрытие с лучшим узлом: {overlap}/20)")

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
        input_sdr      = SDR1024(input_word)
        actual_next_sdr = SDR1024(actual_next)

        # 1. Распознавание текущего концепта
        best_node, best_overlap = self._recall(input_sdr)

        if best_overlap < self.UNKNOWN_THRESHOLD:
            # Полностью неизвестен → bootstrap
            self.stats["unknowns"] += 1
            self.hormones.norepinephrine = min(255, self.hormones.norepinephrine + 160)
            if verbose:
                print(f"\n── #{self.tick_count} «{input_word}»: ❓ НЕИЗВЕСТЕН (overlap={best_overlap})")
            best_node = self._bootstrap_unknown(input_word, verbose)
        else:
            if best_node is None:
                best_node = self.graph.get_or_create_node(input_word, NodeType.ENTITY)

        # 2. Предсказание следующего концепта (лучший синапс по весу)
        predicted_node, predicted_weight = self._predict(best_node)

        # 3. Сравнение предсказания с реальностью
        if predicted_node is not None:
            match_overlap = predicted_node.sdr.overlap(actual_next_sdr)
            hit = match_overlap >= self.HIT_THRESHOLD
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
        """Поиск лучшего узла по SDR-перекрытию (аппаратный popcount & AND)."""
        best_node    = None
        best_overlap = 0
        for node in self.graph.nodes.values():
            ov = sdr.overlap(node.sdr)
            if ov > best_overlap:
                best_overlap = ov
                best_node    = node
        return best_node, best_overlap

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

        # Ослабляем только НЕВЕРНЫЙ предсказанный синапс (как в kurwa.py)
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

    def _bootstrap_unknown(self, word: str, verbose: bool) -> Node:
        """
        Концепт полностью неизвестен.
        Учитель даёт начальные IS_A / HAS_PROPERTY — дальше учимся сами.
        """
        if self.teacher is not None:
            if verbose:
                print(f"   📡 Bootstrap: запрос к Учителю для «{word}»...")
            knowledge = self.teacher.fetch_concept_properties(word)
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
