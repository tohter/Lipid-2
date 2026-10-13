# Контракт пластичности: weight (онтология) ↔ transition (опыт)

**Дата:** 2026-10-13  
**Статус:** черновик на основе замеров `experiments/plasticity_sweep.json`

---

## Результаты базовой линии (дефолт)

| Метрика | Урок 1 | Урок 10 | Transfer |
|---------|--------|---------|----------|
| accuracy_strict | 0.0% | 50.0% | 33.3% |
| coverage | 40% | 100% | 0% |
| transitions | 3 | 10 | — |
| predictability | 150 | 15 | — |

**Проблемы:**
- Transfer 33.3% — это честно: 2 из 6 переходов видели в обучении (бегает→дворе, спит→подоконнике). Остальные — новые комбинации.
- Predictability падает 150→15, не насыщаясь на целевом значении (100-120).
- Нет затухания переходов (decay) — сила нарастает до 15 и живёт вечно.
- delta_surprise=0 даёт accuracy 66.7% (быстрее), но убивает адаптацию.

---

## Рекомендуемый контракт

### 1. Параметры обучения (подтверждены замерами)

```python
@dataclass
class PlasticityConfig:
    delta_confirm: int = 2          # шаг силы при попадании/новом переходе
    delta_surprise: int = 1         # шаг слабения ожидании при промахе
    dopamine_gain_hit: int = 120    # закрепление успеха
    norepinephrine_gain_no_exp: int = 40
    norepinephrine_gain_miss: int = 120
    norepinephrine_gain_new: int = 40
    cortisol_gain_miss: int = 100
    predictability_gain_hit: int = 10   # было 15 — гомеостаз садится
    predictability_loss_miss: int = 10  # было 25 — симметрия
    hormone_decay_dopa: int = 30        # было 40 — дольше живут
    hormone_decay_norep: int = 50       # было 60
    hormone_decay_cort: int = 20        # было 30
    soft_overlap_threshold: int = 13    # совпадает с UNKNOWN_CEILING
```

### 2. Правило переноса опыт → онтология (СОН)

```python
# Вызывается в фоне (sleep cycle), когда накоплено достаточно опыта
def consolidate_during_sleep(graph: GraphMemory, cortex: PredictiveCortex):
    for head in cortex.transitions.heads():
        for tail in cortex.transitions.known_tails(head):
            strength = cortex.transitions.get(head, tail)
            if strength < CONSOLIDATION_THRESHOLD:  # 8 из 15
                continue
            
            # Есть ли CAN_DO в онтологии?
            if graph.has_edge(head, tail, RelationType.CAN_DO):
                # Опыт подтверждает — повышаем вес (но не выше 200)
                w = graph.get_weight(head, tail, RelationType.CAN_DO)
                new_w = min(200, w + (strength - 7) * 10)
                graph.set_weight(head, tail, RelationType.CAN_DO, new_w)
            else:
                # Опыт требует новый CAN_DO — создаём с весом по формуле
                weight = 80 + strength * 8  # 8..15 -> 144..200
                graph.add_edge(head, tail, RelationType.CAN_DO, weight, 
                               source="experience:consolidation")
```

### 3. Правило согласования (конфликт онтологии и опыта)

```python
# Если онтология говорит "да" (weight высокий), а опыт "нет" (strength низкий)
if weight >= 150 and strength <= 3:
    # Проблема: факт в онтологии, но в опыте не встречается
    # Действие: снизить вес, но не ниже 80 (базовое "может")
    graph.set_weight(head, tail, RelationType.CAN_DO, max(80, weight - 30))

# Если онтология "нет" (нет CAN_DO), а опыт "да" (strength высокий)
# → сработает consolidate_during_sleep (создаст CAN_DO)

# Если онтология "да" (weight высокий), опыт "да" (strength высокий)
# → ничего не делаем, подтверждение
```

### 4. Затухание переходов (DECAY) — обязательно для сна

```python
def decay_transitions(cortex: PredictiveCortex, rate: float = 0.05):
    """Периодическое затухание всех сил переходов (son)."""
    for head in cortex.transitions.heads():
        for tail in cortex.transitions.known_tails(head):
            s = cortex.transitions.get(head, tail)
            if s > 0:
                new_s = max(0, int(s * (1 - rate)))
                cortex.transitions._adjust(head, tail, new_s - s)
    cortex.transitions.compact()
```

**Частота:** раз в N уроков (N=50) или при predictability > 200.

---

## Числа для реализации (P0 #16 done → P1)

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| `delta_confirm` | 2 | 1 слишком медленно, 3 — перелет на 15 за 5 уроков |
| `delta_surprise` | 1 | 0 убивает адаптацию, 2 — слишком агрессивно стирает |
| `CONSOLIDATION_THRESHOLD` | 8/15 | 50% силы — достаточное подтверждение |
| `weight_from_strength` | 80 + s*8 | 144..200, в окне CAN_DO (90..200) |
| `conflict_weight_down` | -30 за сон | Постепенное снижение, не мгновенное |
| `transition_decay_rate` | 5% за сон | Медленное забывание редких паттернов |
| `predictability_target` | 100-120 | Гомеостаз должен садиться сюда |

---

## Следующие шаги (после согласования)

1. **Внедрить PlasticityConfig в brain/lipid2_brain.py** — передать в PredictiveCortex
2. **Написать `consolidate_during_sleep()`** в `brain/learner.py` или новом `brain/sleep.py`
3. **Добавить decay_transitions()** — вызывать из sleep
4. **Тест контракта** — `tests/test_plasticity_contract.py`:
   - опыт учит переход → sleep создаёт CAN_DO с правильным весом
   - конфликт онтология/опыт → вес скорректировался
   - decay не убивает частые переходы