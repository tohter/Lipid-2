"""
core/synapses.py: исходящие синапсы узла как СПИСОК рёбер.

Дыра 4 (латентная). Исторически out_synapses_typed был словарём по имени
цели, и connect_to к той же цели ПЕРЕЗАПИСЫВАЛ первое ребро:
    out_synapses_typed[target] = {"rel_type":..., "weight":...}
Пока ядро только складывало связи, потерю не было видно; как только ядро
начнут читать для рассуждений, одна реальная связь окажется пустой. В Lipid1
хранилище было списком рёбер. Здесь возвращаем список, но КАЖДОЕ ребро несёт
своё target: {"target", "rel_type", "weight"[, "source"]}.

Отдельный модуль (а не в core/graph.py): и graph, и storage обязаны видеть
этот тип, а импорт storage->graph создал бы цикл. Это единственное место, где
тип определён.

Обратная совместимость чтения сохранена через __getitem__/get/items:
- out_synapses_typed[цель]  -> ПЕРВОЕ ребро к цели (как читал словарь)
- out_synapses_typed[индекс] -> ребро по позиции (список)
- .get(цель) / .items()     -> как у словаря (первое ребро на цель)
- .raw_edges() / итерация  -> ВСЕ рёбра (нужно ядру и сериализатору)
- .all_for(цель)            -> список всех рёбер к цели
"""


class SynapseList(list):
    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        return self.first_for(key)

    def get(self, target, default=None):
        first = self.first_for(target)
        return first if first is not None else default

    def items(self):
        seen = set()
        for e in list.__iter__(self):
            t = e.get("target")
            if t not in seen:
                seen.add(t)
                yield t, e

    def first_for(self, target):
        for e in list.__iter__(self):
            if e.get("target") == target:
                return e
        return None

    def all_for(self, target):
        return [e for e in list.__iter__(self) if e.get("target") == target]

    def raw_edges(self):
        return list.__iter__(self)