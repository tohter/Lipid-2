"""
Эксперимент: как превратить dense-эмбеддинг в SDR-1024 с 20 активными битами,
СОХРАНИВ семантическое перекрытие.

Кандидаты:
  A) Random Projection Pursuit (Salakhov): 1024 случайных гиперплоскостей,
     активны 20 с наибольшим проекцией. Exactly-20 по построению.
  B) Гиперплоскостной LSH: bit = sign(v·r) > 0, затем прунинг до 20 по |проекции|.
  C) Топ-k размерных осей (не годится, но для полноты).

Метрика качества = ранжирование: overlap(синоним) > overlap(орфо-двойник)
> overlap(несвязанные). Именно это ядро использует как «узнал / не узнал».
"""
import numpy as np
from sentence_transformers import SentenceTransformer

TOTAL_BITS = 1024
ON = 20

SYN = [("кошка", "кот"), ("грусть", "печаль"), ("дом", "жилище"), ("бег", "беготня"),
       ("любовь", "нежность"), ("смелый", "храбрый")]
ORTH = [("кот", "код"), ("дом", "том"), ("лес", "леск"), ("кот", "пот"), ("дом", "домик")]
UNREL = [("кошка", "табуретка"), ("грусть", "табуретка"), ("любовь", "гвоздь"),
         ("стол", "река"), ("огонь", "книга"), ("дерево", "самолёт")]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

all_words = list(dict.fromkeys([w for p in SYN + ORTH + UNREL for w in p]))
# Калибровочный корпус — нужен только для центрирования (метод C-центрированный)
CALIB = """кошка собака дерево стол табуретка дом окно любовь грусть страх бег сон
хлеб вода огонь дождь лес поле гора море рыба птица конь волк слово звук свет тень""".split()

E = model.encode(all_words, normalize_embeddings=True)
EC = model.encode(CALIB, normalize_embeddings=True)
MEAN = EC.mean(axis=0)
MEAN /= np.linalg.norm(MEAN)


def rpp(E, R, on=ON):
    """A: top-`on` гиперплоскостей по величине проекции."""
    s = E @ R.T
    idx = np.argpartition(-s, on - 1, axis=1)[:, :on]
    mask = np.zeros((E.shape[0], R.shape[0]), dtype=np.uint8)
    for r, cols in enumerate(idx):
        mask[r, cols] = 1
    return mask


def lsh_prune(E, R, on=ON):
    """B: знаковый LSH, при нехватке 20 — берём самые «уверенные» единичные."""
    s = E @ R.T
    mask = (s > 0).astype(np.uint8)
    out = np.zeros_like(mask)
    for r in range(E.shape[0]):
        cand = np.flatnonzero(mask[r])
        if len(cand) >= on:
            pick = cand[np.argsort(-s[r, cand])[:on]]
        else:  # добрать отрицательными с наибольшим |проецией|
            rest = np.flatnonzero(mask[r] == 0)
            need = on - len(cand)
            extra = rest[np.argsort(-np.abs(s[r, rest]))[:need]]
            pick = np.concatenate([cand, extra])
        out[r, pick] = 1
    return mask


def report(name, mask):
    def ov(i, j):
        return int((mask[i] & mask[j]).sum())
    i = {w: k for k, w in enumerate(all_words)}
    groups = {}
    for label, ps in (("syn", SYN), ("orth", ORTH), ("unrel", UNREL)):
        v = [ov(i[a], i[b]) for a, b in ps]
        groups[label] = v
    s, o, u = np.mean(groups["syn"]), np.mean(groups["orth"]), np.mean(groups["unrel"])
    ok = s > o and s > u and u <= 3
    print(f"  {name:34s} syn={s:5.2f}  orth={o:5.2f}  unrel={u:4.2f}   "
          f"syn>orth:{'ДА ' if s > o else 'НЕТ'}  unrel<=3:{'ДА' if u <= 3 else 'НЕТ'}  -> {'OK' if ok else 'ПЛОХО'}")
    print(f"     попарно syn={groups['syn']} orth={groups['orth']} unrel={groups['unrel']}")
    return ok


for nhyper in (1024, 4096, 16384):
    for seed in (1,):
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((nhyper, E.shape[1]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        print(f"\n=== n_hyperplanes={nhyper}, seed={seed}, без центрирования ===")
        report("A rpp", rpp(E, R))
        report("B lsh_prune", lsh_prune(E, R))
        Ec = E - (E @ MEAN)[:, None] * MEAN
        Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
        report("A rpp + центрирование", rpp(Ec, R))
        report("B lsh + центрирование", lsh_prune(Ec, R))
