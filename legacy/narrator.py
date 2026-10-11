"""
narrator.py — Двухпроходный нарративный парсер.

Проход 1 (Surface): агенты + события явно в тексте
Проход 2 (Deep):    скрытые концепты + каузальные связи

Результат интегрируется в GraphMemory и позволяет:
  - simulate() — прогнать нарратив по тикам
  - answer_why(event) — объяснить почему что-то случилось
"""

import json
import re
import time
import urllib.request
from typing import Optional

from graph import GraphMemory, Node
from storage import NodeType, RelationType


# ═══════════════════════════════════════════════════════════════
# СИСТЕМНЫЕ ПРОМПТЫ
# ═══════════════════════════════════════════════════════════════

_PASS1_SYS = """\
Ты — нарративный анализатор. Извлеки из текста ВСЕХ агентов и ВСЕ события строго в хронологическом порядке.

Правила:
1. lemma — одно слово, именительный падеж, по-русски
2. type события: KINETIC (физ. действие), METABOLIC (внутр. состояние), SEMIOTIC (речь/информация/восприятие)
3. note — краткое пояснение события (5-8 слов)
4. Отвечай ТОЛЬКО валидным JSON. Первый символ ответа — {

Формат:
{
  "agents": [
    {"id": "cat",    "lemma": "кошка",    "category": "agent"},
    {"id": "canary", "lemma": "канарейка","category": "agent"}
  ],
  "events": [
    {"id": "e1", "tick": 1, "actor": "cat",    "action": "поделилась", "target": "canary",  "type": "SEMIOTIC",  "note": "передала секрет проекта"},
    {"id": "e2", "tick": 2, "actor": "canary", "action": "раскрыла",   "target": "улица",   "type": "SEMIOTIC",  "note": "рассказала всем через форточку"},
    {"id": "e3", "tick": 3, "actor": "cat",    "action": "увидела",    "target": "e2",      "type": "SEMIOTIC",  "note": "кошка узнала о предательстве"},
    {"id": "e4", "tick": 4, "actor": "cat",    "action": "разозлилась","target": null,      "type": "METABOLIC", "note": "внутреннее эмоциональное состояние"},
    {"id": "e5", "tick": 5, "actor": "cat",    "action": "сожрала",    "target": "canary",  "type": "KINETIC",   "note": "физическое уничтожение"}
  ]
}"""

_PASS2_SYS = """\
Ты — психологический аналитик нарратива.
Тебе дан список событий истории. Определи скрытые концепты (эмоции, моральные понятия, социальные состояния), которые объясняют ПОЧЕМУ события произошли именно так.

Правила:
1. lemma концепта — одно слово, именительный падеж, по-русски.
2. intensity — сила связи: 1..255.
3. relation — один глагол из списка: активирует / разрушает / вызывает / усиливает / приводит_к.
4. cause и effect — ID события (e1, e2...) ИЛИ ID концепта.
5. agent — ID агента, ЧЬЁ внутреннее состояние (эмоция, доверие, гнев) меняется под действием этой связи (например, если канарейка предала, состояние меняется у КОШКИ).
6. ВАЖНО: Каузальная цепочка ОБЯЗАТЕЛЬНО должна доходить до ПОСЛЕДНЕГО KINETIC-события (физического действия).
7. Отвечай ТОЛЬКО валидным JSON. Первый символ ответа — {

Формат:
{
  "concepts": [
    {"id": "trust",     "lemma": "доверие",      "category": "concept"},
    {"id": "betrayal",  "lemma": "предательство","category": "concept"},
    {"id": "anger",     "lemma": "гнев",         "category": "concept"},
    {"id": "revenge",   "lemma": "месть",        "category": "concept"}
  ],
  "causal_links": [
    {"cause": "e1",        "effect": "trust",     "agent": "cat", "intensity": 200, "relation": "активирует"},
    {"cause": "e2",        "effect": "betrayal",  "agent": "cat", "intensity": 255, "relation": "активирует"},
    {"cause": "betrayal",  "effect": "trust",     "agent": "cat", "intensity": 220, "relation": "разрушает"},
    {"cause": "betrayal",  "effect": "anger",     "agent": "cat", "intensity": 250, "relation": "вызывает"},
    {"cause": "anger",     "effect": "revenge",   "agent": "cat", "intensity": 200, "relation": "вызывает"},
    {"cause": "revenge",   "effect": "e5",        "agent": "cat", "intensity": 255, "relation": "приводит_к"}
  ]
}"""


# ═══════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЙ ПАРСЕР JSON (переиспользуем логику lipid_agent)
# ═══════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call\|?>.*?(?:\n|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'</?\w[\w\s=":\'.-]{0,30}?>', '', text)
    text = re.sub(r'^\s*(thought|answer|output)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'```[a-z]*\s*|```\s*', '', text)
    return text.strip()


def _find_json_blocks(text: str) -> list[str]:
    """Скобочный обход — все сбалансированные JSON-блоки, от длинного к короткому."""
    blocks, i, n = [], 0, len(text)
    while i < n:
        if text[i] != '{':
            i += 1
            continue
        depth, end = 0, -1
        for j in range(i, n):
            if text[j] == '{':   depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end >= 0:
            blocks.append(text[i:end + 1])
            i = end + 1
        else:
            # Обрезанный блок — пробуем закрыть
            frag = text[i:]
            last = frag.rfind('}')
            if last >= 0:
                trimmed  = frag[:last + 1]
                n_sq = max(0, trimmed.count('[') - trimmed.count(']'))
                n_br = max(0, trimmed.count('{') - trimmed.count('}'))
                blocks.append(trimmed + ']' * n_sq + '}' * n_br)
            break
    blocks.sort(key=len, reverse=True)
    return blocks


def _repair(s: str) -> str:
    s = re.sub(r',\s*([\}\]])', r'\1', s)
    s = re.sub(r'(?<!\\)\n', ' ', s)
    s = re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r' "\1":', s)
    s = re.sub(r'("lemma"|"action"|"relation"|"note")\s*:\s*-?\d+', r'\1: ""', s)
    return s


def _parse_llm_json(raw: str) -> dict:
    text = _clean(raw)
    for block in _find_json_blocks(text):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_repair(block))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Не удалось распарсить JSON. Начало: {raw[:150]!r}")


# ═══════════════════════════════════════════════════════════════
# LLM-КЛИЕНТ (минимальный, без зависимостей)
# ═══════════════════════════════════════════════════════════════

class _LLMClient:
    def __init__(self, host: str):
        self.url = f"{host.rstrip('/')}/v1/completions"

    def call(self, system: str, user_text: str,
             temperature: float = 0.1, max_tokens: int = 1500) -> dict:
        prompt = (
            f"<start_of_turn>user\n{system}\n\n"
            f"Текст для анализа:\n\"\"\"{user_text}\"\"\"\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
            f"<think>\n</think>\n"
        )
        payload  = json.dumps({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json"}
        )
        for attempt in range(1, 3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    raw  = data["choices"][0]["text"]
                    return _parse_llm_json(raw)
            except ValueError as e:
                print(f"   ⚠️  Парсинг (попытка {attempt}/2): {e}")
                return {}
            except Exception as e:
                print(f"   ❌ Запрос (попытка {attempt}/2): {e}")
                if attempt < 2:
                    time.sleep(2)
        return {}


# ═══════════════════════════════════════════════════════════════
# НАРРАТИВНЫЙ ПАРСЕР
# ═══════════════════════════════════════════════════════════════

class NarrativeParser:
    """
    Двухпроходный анализатор нарратива.

    pass1 → поверхностный слой (агенты + события)
    pass2 → глубинный слой    (скрытые концепты + каузальные связи)

    После parse(text) доступны:
      .answer_why("сожрала") — каузальная цепочка назад
      .simulate()            — прогон нарратива по тикам
      .print_graph_summary() — что попало в граф
    """

    def __init__(self, llm_host: str, graph: GraphMemory):
        self.llm   = _LLMClient(llm_host)
        self.graph = graph

        # Нарративный контекст
        self.agents:       dict = {}   # id → {"lemma":..., "node":Node}
        self.events:       dict = {}   # id → {"tick","actor","action","target","type","note"}
        self.concepts:     dict = {}   # id → {"lemma":..., "node":Node}
        self.causal_links: list = []   # [{cause, effect, agent, intensity, relation}]

        # Обратный индекс для Q&A: effect_id → [link, ...]
        self._reverse: dict = {}

    # ── ПУБЛИЧНЫЙ ИНТЕРФЕЙС ─────────────────────────────────

    def parse(self, text: str) -> bool:
        """Полный двухпроходный анализ текста."""
        print("\n" + "═" * 60)
        print("📖 НАРРАТИВНЫЙ АНАЛИЗ")
        print("═" * 60)

        # ПРОХОД 1 — поверхностный
        print("\n🔍 [Проход 1] Поверхностная структура...")
        surface = self.llm.call(_PASS1_SYS, text)
        if not surface:
            print("   ❌ Проход 1 не дал результата.")
            return False
        self._load_surface(surface)
        self._print_surface()

        # ПРОХОД 2 — глубинный
        print("\n🧠 [Проход 2] Скрытые концепты и каузальность...")
        events_ctx = json.dumps(
            {"events": list(self.events.values())},
            ensure_ascii=False, indent=2
        )
        deep = self.llm.call(_PASS2_SYS, events_ctx)
        if not deep:
            print("   ⚠️  Проход 2 не дал результата. Работаем только с поверхностным слоем.")
        else:
            self._load_deep(deep)
            self._print_deep()

        # Интеграция в граф
        self._integrate_to_graph()
        print(f"\n✅ Граф пополнен. Узлов всего: {len(self.graph.nodes)}")
        return True

    def answer_why(self, query: str) -> str:
        """
        Отвечает на вопрос «почему произошло X?»
        Ищет событие или концепт по лемме или действию,
        затем строит каузальную цепочку назад.
        """
        target_id = self._find_id(query)
        if not target_id:
            return f"❓ «{query}» не найдено в нарративе."

        chain = []
        self._trace_causes(target_id, chain, visited=set(), depth=0)

        if not chain:
            return f"ℹ️  Прямых причин для «{self._name(target_id)}» не найдено."

        lines = [f"\n❓ Почему «{self._name(target_id)}»?\n"]
        seen_effects = set()
        for link in chain:
            eff = link["effect"]
            if eff in seen_effects:
                continue
            seen_effects.add(eff)
            cause_name  = self._name(link["cause"])
            effect_name = self._name(link["effect"])
            relation    = link["relation"]
            strength    = self._strength_word(link["intensity"])
            agent_str   = f" [{self._name(link.get('agent',''))}]" if link.get("agent") else ""
            lines.append(f"  • «{cause_name}» {strength} {relation} «{effect_name}»{agent_str}")

        return "\n".join(lines)

    def simulate(self):
        """Прогоняет нарратив по тикам — показывает эволюцию состояний агентов."""
        print("\n" + "═" * 60)
        print("⚡ СИМУЛЯЦИЯ НАРРАТИВА (по тикам)")
        print("═" * 60)

        agent_state = {
            aid: {"calm": 200, "anger": 0, "trust": 128}
            for aid in self.agents
        }

        sorted_events = sorted(self.events.values(), key=lambda e: e["tick"])

        for ev in sorted_events:
            tick   = ev["tick"]
            actor  = ev["actor"]
            action = ev["action"]
            target = ev.get("target", "—")
            etype  = ev["type"]
            note   = ev.get("note", "")
            eid    = ev["id"]

            print(f"\n── Тик {tick} [{etype}]: {self._name(actor)} → {action} → {self._name(target)}")
            if note:
                print(f"   📝 {note}")

            # 1. ПРЯМОЙ ЭФФЕКТ ДЛЯ METABOLIC / SEMIOTIC СОБЫТИЙ (Независимо от LLM-связей)
            if actor in agent_state:
                st = agent_state[actor]
                act_lower = action.lower()
                note_lower = note.lower()

                if etype == "METABOLIC" or any(w in act_lower or w in note_lower for w in ("разозлилась", "гнев", "обида", "ярость")):
                    st["anger"] = min(255, st["anger"] + 150)
                    st["calm"]  = max(0,   st["calm"]  - 150)
                    print(f"   🔥 Внутренний аффект актора «{self._name(actor)}»: anger +150, calm -150")

            # 2. ОБРАБОТКА КАУЗАЛЬНЫХ СВЯЗЕЙ С УЧЕТОМ ЦЕЛЕВОГО АГЕНТА
            for lk in self.causal_links:
                effect_name = self._name(lk["effect"]).lower()
                cause_name  = self._name(lk["cause"]).lower()
                relation    = lk["relation"]
                intensity   = lk["intensity"]

                # Определяем, чье состояние меняется (если LLM ошибласть с agent, применяем к актору события)
                target_agent_id = lk.get("agent")
                if not target_agent_id or target_agent_id not in agent_state:
                    target_agent_id = actor

                st = agent_state.get(target_agent_id)
                if not st:
                    continue

                # Реакция на текущий шаг/событие
                if lk["cause"] == eid or cause_name in action.lower():
                    print(f"   🔗 {relation}: «{self._name(lk['effect'])}» (субъект: {self._name(target_agent_id)}, интенсивность: {intensity})")

                    if relation == "разрушает" and "доверие" in effect_name:
                        st["trust"] = max(0, st["trust"] - intensity)
                        st["anger"] = min(255, st["anger"] + intensity // 2)
                        st["calm"]  = max(0, st["calm"] - intensity // 2)

                    elif relation in ("вызывает", "активирует", "приводит_к") and any(w in effect_name for w in ("предательство", "гнев", "месть")):
                        st["anger"] = min(255, st["anger"] + intensity // 2)
                        st["calm"]  = max(0, st["calm"] - intensity // 2)

                    elif relation == "активирует" and "доверие" in effect_name:
                        st["trust"] = min(255, st["trust"] + intensity // 2)

            # Вывод состояния актора
            if actor in agent_state:
                st = agent_state[actor]
                def bar(v): return "█" * (v // 32) + "░" * (8 - v // 32)
                print(f"   🧬 {self._name(actor)}: calm={bar(st['calm'])} anger={bar(st['anger'])} trust={bar(st['trust'])}")

                if st["anger"] >= 180 and etype == "KINETIC":
                    print(f"   💥 ПОРОГ ГНЕВА ПРЕВЫШЁН ({st['anger']}/255) → Физическая агрессия!")

        print("\n" + "─" * 60)
        print("ФИНАЛЬНЫЕ СОСТОЯНИЯ АГЕНТОВ:")
        for aid, st in agent_state.items():
            def bar(v): return "█" * (v // 32) + "░" * (8 - v // 32)
            print(f"  {self._name(aid):12s} │ "
                  f"calm={bar(st['calm'])} anger={bar(st['anger'])} trust={bar(st['trust'])}")

    def print_graph_summary(self):
        """Показывает что попало в граф из нарратива."""
        print("\n📊 ГРАФ НАРРАТИВА:")
        print(f"   Агентов: {len(self.agents)}, "
              f"Событий: {len(self.events)}, "
              f"Концептов: {len(self.concepts)}, "
              f"Каузальных связей: {len(self.causal_links)}")
        print("\n   Каузальная цепочка:")
        for lk in self.causal_links:
            agent_str = f" [{self._name(lk.get('agent',''))}]" if lk.get("agent") else ""
            print(f"   «{self._name(lk['cause']):14s}» "
                  f"--{lk['relation']}({lk['intensity']:3d})--> "
                  f"«{self._name(lk['effect'])}»{agent_str}")

    # ── ЗАГРУЗКА ДАННЫХ ПРОХОДОВ ────────────────────────────

    def _load_surface(self, data: dict):
        for ag in data.get("agents", []):
            aid = ag.get("id", "")
            if aid:
                self.agents[aid] = {
                    "lemma": ag.get("lemma", aid),
                    "category": ag.get("category", "agent")
                }

        for ev in data.get("events", []):
            eid = ev.get("id", "")
            if eid:
                self.events[eid] = {
                    "id":     eid,
                    "tick":   ev.get("tick", 0),
                    "actor":  ev.get("actor", ""),
                    "action": ev.get("action", ""),
                    "target": ev.get("target"),
                    "type":   ev.get("type", "SEMIOTIC"),
                    "note":   ev.get("note", "")
                }

    def _load_deep(self, data: dict):
        for con in data.get("concepts", []):
            cid = con.get("id", "")
            if cid:
                self.concepts[cid] = {
                    "lemma": con.get("lemma", cid),
                    "category": con.get("category", "concept")
                }

        for lk in data.get("causal_links", []):
            cause  = lk.get("cause", "")
            effect = lk.get("effect", "")
            if cause and effect:
                link = {
                    "cause":    cause,
                    "effect":   effect,
                    "agent":    lk.get("agent", ""),
                    "intensity": int(lk.get("intensity", 100)),
                    "relation": lk.get("relation", "вызывает")
                }
                self.causal_links.append(link)
                # Строим обратный индекс
                if effect not in self._reverse:
                    self._reverse[effect] = []
                self._reverse[effect].append(link)

    # ── ИНТЕГРАЦИЯ В ГРАФ ───────────────────────────────────

    def _integrate_to_graph(self):
        """Записывает все сущности и связи в GraphMemory."""
        # Агенты
        for aid, ag in self.agents.items():
            node = self.graph.get_or_create_node(ag["lemma"], NodeType.ENTITY)
            self.agents[aid]["node"] = node

        # Концепты
        for cid, con in self.concepts.items():
            node = self.graph.get_or_create_node(con["lemma"], NodeType.ENTITY)
            self.concepts[cid]["node"] = node

        # События как ACTION-узлы
        for eid, ev in self.events.items():
            node_name = ev["action"]
            node = self.graph.get_or_create_node(node_name, NodeType.ACTION)
            self.events[eid]["node"] = node

            # Связываем актора с действием
            actor_lemma = self._agent_lemma(ev["actor"])
            if actor_lemma:
                actor_node = self.graph.get_or_create_node(actor_lemma, NodeType.ENTITY)
                actor_node.connect_to(node, rel_type=RelationType.CAN_DO, weight=200)

        # Каузальные связи
        for lk in self.causal_links:
            cause_node  = self._resolve_node(lk["cause"])
            effect_node = self._resolve_node(lk["effect"])
            if cause_node and effect_node:
                cause_node.connect_to(
                    effect_node,
                    rel_type=RelationType.CAUSES,
                    weight=lk["intensity"]
                )

    # ── ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ──────────────────────────────

    def _resolve_node(self, nid: str) -> Optional[Node]:
        """ID (e1, trust, cat) → Node в графе."""
        if nid in self.agents:
            return self.agents[nid].get("node")
        if nid in self.concepts:
            return self.concepts[nid].get("node")
        if nid in self.events:
            return self.events[nid].get("node")
        # Прямое имя
        return self.graph.nodes.get(nid.lower())

    def _find_id(self, query: str) -> Optional[str]:
        """Ищет ID по лемме, действию или частичному совпадению."""
        q = query.lower().strip()
        # Прямой ID
        for d in [self.events, self.agents, self.concepts]:
            if q in d:
                return q
        # По лемме концепта
        for cid, con in self.concepts.items():
            if q in con["lemma"].lower():
                return cid
        # По лемме агента
        for aid, ag in self.agents.items():
            if q in ag["lemma"].lower():
                return aid
        # По действию события
        for eid, ev in self.events.items():
            if q in ev["action"].lower() or q in ev.get("note", "").lower():
                return eid
        return None

    def _trace_causes(self, node_id: str, result: list, visited: set, depth: int):
        """DFS назад по каузальным связям."""
        if depth > 6 or node_id in visited:
            return
        visited.add(node_id)
        for lk in self._reverse.get(node_id, []):
            result.append(lk)
            self._trace_causes(lk["cause"], result, visited, depth + 1)

    def _name(self, nid: str) -> str:
        """ID → человекочитаемое имя."""
        if not nid:
            return "?"
        if nid in self.agents:
            return self.agents[nid]["lemma"]
        if nid in self.concepts:
            return self.concepts[nid]["lemma"]
        if nid in self.events:
            ev = self.events[nid]
            return f"{ev['action']}({self._name(ev['actor'])})"
        return nid

    def _agent_lemma(self, agent_id: str) -> Optional[str]:
        ag = self.agents.get(agent_id)
        return ag["lemma"] if ag else None

    @staticmethod
    def _strength_word(intensity: int) -> str:
        if intensity >= 220: return "сильно"
        if intensity >= 150: return "умеренно"
        if intensity >= 80:  return "слабо"
        return "едва"

    def _print_surface(self):
        print(f"   Агентов: {len(self.agents)}, Событий: {len(self.events)}")
        for eid, ev in sorted(self.events.items(), key=lambda x: x[1]["tick"]):
            tgt = self._name(ev.get("target") or "") if ev.get("target") else "—"
            print(f"   t={ev['tick']} [{ev['type'][:3]}] "
                  f"{self._name(ev['actor'])} → {ev['action']} → {tgt}")

    def _print_deep(self):
        print(f"   Концептов: {len(self.concepts)}, "
              f"Каузальных связей: {len(self.causal_links)}")
        for lk in self.causal_links:
            print(f"   «{self._name(lk['cause']):14s}» "
                  f"--{lk['relation']}--> «{self._name(lk['effect'])}»")
