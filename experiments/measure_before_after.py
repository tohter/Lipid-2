"""Одна и та же цепочка пересказов на ДВУХ версиях ядра: HEAD и работ.

Скрипт нужен затем, чтобы «до» не становилось памятью: один и тот же код
прогоняется против двух деревьев, путь импортируемого модуля печатается,
поэтому нельзя ненароком померить рабочее дерево и назвать цифру старой.

    git worktree add -q /tmp/head_wt HEAD
    LIPID2_STATUS=0 .venv/bin/python experiments/measure_before_after.py /tmp/head_wt HEAD
    LIPID2_STATUS=0 .venv/bin/python experiments/measure_before_after.py . РАБОТ
    git worktree remove /tmp/head_wt

Ответ про дождь и пожар здесь не нужен и не хранится — меряется аппарат.
"""
import sys
ROOT, МЕТКА = sys.argv[1], sys.argv[2]
sys.path.insert(0, ROOT)
from lang.ru.lemma import install; install()
from core.records import add_witness, fact_weight
print("замеряемый файл ядра:", sys.modules['core.records'].__file__)

def запись():
    return {"head": "дождь", "relation": "CAN_DO", "tail": "гасить",
            "confidence": 0.62, "negated": False, "support": 1,
            "source": "wiki", "evidence": "дождь помогает тушить пожары"}

# Уверенность, КОТОРУЮ КАЖДЫЙ КАНАЛ ВЫДАЁТ О САМОМ СЕБЕ в живом прогоне.
канал = [("model", "дождь гасит пожар", 0.75),
         ("gemma", "дождевая вода тушит огонь", 0.80),
         ("llm", "дождь способен гасить пламя", 0.90),
         ("model", "огонь гасится дождевой водой", 0.85)]

obs = запись()
obs["weight"] = fact_weight(obs["confidence"], False, "CAN_DO", 1)
print(f"{МЕТКА} до:            support = {obs['support']} вес = {obs['weight']}")
for s, e, c in канал:
    try:
        add_witness(obs, source=s, evidence=e, confidence=c,
                    relation="CAN_DO", document="")
    except TypeError:
        add_witness(obs, source=s, evidence=e, confidence=c, relation="CAN_DO")
основания = "?"
try:
    from core.records import independent_bases
    основания = independent_bases(obs)
except ImportError:
    pass
print(f"{МЕТКА} после 4 пересказов: support = {obs['support']} "
      f"вес = {obs['weight']} отдельных оснований = {основания}")
