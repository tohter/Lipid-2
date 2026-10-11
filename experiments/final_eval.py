"""
ФИНАЛЬНЫЙ ЗАМЕР. Что нужно предиктивному ядру:

(A) РАСПОЗНАВАНИЕ: «этот концепт уже есть в графе?» — query совпадает со своим
    узлом (overlap = ON) против лучшего ЧУЖОГО узла. Меряем margin и порог FP<1%.
(B) СЕМАНТИКА: overlap(синоним) > overlap(несвязанное) на курируемых парах.
(C) ПРЕЦИЗИЯ СОСЕДЕЙ: precision@10 — доля SDR-топ-10, которые и по косинусу в
    истинных ближайших (замена сломанного recall@10: при 20 битах переполнение
    связок делает argsort внутри связок произвольным).

Перебираем: центрирование вкл/выкл, число активных бит 16/20/32/48/64, n_hyper.
Вывод: какой минимальной разреженностью не жертвовать узнаванием.
"""
import os
import numpy as np
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

TOTAL = 1024
NPY = "experiments/pool_emb.npy"

SYN = [("кошка", "кот"), ("грусть", "печаль"), ("храбрый", "смелый"), ("дом", "жилище"),
       ("бег", "беготня"), ("любовь", "нежность"), ("путь", "дорога"), ("огонь", "пламя")]
UNREL = [("кошка", "табуретка"), ("грусть", "гвоздь"), ("стол", "река"),
         ("огонь", "книга"), ("дом", "алгебра"), ("рыба", "цемент"),
         ("птица", "молоток"), ("вода", "война")]

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
DIM = model.get_sentence_embedding_dimension()
E = np.load(NPY)
WORDS = np.load(NPY.replace(".npy", "_words.npy"), allow_pickle=True)
WIDX = {w: i for i, w in enumerate(WORDS)}
print(f"Пул: {len(E)} реальных русских слов\n", flush=True)

CENT = E.mean(0); CENT /= np.linalg.norm(CENT)


def prep(center):
    Et = E - (E @ CENT)[:, None] * CENT if center else E
    Et = Et / np.maximum(np.linalg.norm(Et, axis=1, keepdims=True), 1e-9)
    return Et.astype(np.float32)


def encode(Et, R, on):
    proj = Et @ R.T
    idx = np.argpartition(-proj, on - 1, axis=1)[:, :on] % TOTAL
    out = np.zeros((proj.shape[0], TOTAL), dtype=np.uint8)
    out[np.arange(proj.shape[0])[:, None], idx] = 1
    return out


def sparse_of(mask):
    rows, coll = np.nonzero(mask)
    return sp.csr_matrix((np.ones(coll.size, dtype=np.uint8), (rows, coll)),
                         shape=(mask.shape[0], TOTAL))


rng = np.random.default_rng(0)
Q = rng.choice(len(E), 300, replace=False)

hdr = ("центр  бит  nh     | margin узнавания  | порог FP<1% | FP чужих | "
       "  syn unrel   gap  | prec@10")
print(hdr)
for center in (True, False):
    Et = prep(center)
    for nh in (1024, 4096):
        R = rng.standard_normal((nh, DIM)).astype(np.float32)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        for on in (16, 20, 32, 48, 64):
            mask = encode(Et, R, on)
            S = sparse_of(mask)
            # ---- (A) узнавание: свой узел (ON) против лучшего чужого
            lib = np.setdiff1d(np.arange(len(E)), Q)
            C = (S[Q] @ S[lib].T).toarray()
            best_other = C.max(1)
            margin = on - np.median(best_other)
            thr99 = int(np.percentile(best_other, 99)) + 1     # лишь 1% запросов ошибётся
            # ---- (B) семантика
            wsyn = [(WIDX[a], WIDX[b]) for a, b in SYN if a in WIDX and b in WIDX]
            wun = [(WIDX[a], WIDX[b]) for a, b in UNREL if a in WIDX and b in WIDX]
            s_ov = np.mean([int((mask[i] & mask[j]).sum()) for i, j in wsyn]) if wsyn else np.nan
            u_ov = np.mean([int((mask[i] & mask[j]).sum()) for i, j in wun]) if wun else np.nan
            # ---- (C) precision@10 против косинусных соседей
            Sim = Et[Q] @ Et.T
            Sim[np.arange(len(Q)), Q] = -9
            cos10 = set()
            for i in range(len(Q)):
                cos10.update(np.argsort(-Sim[i])[:10].tolist())
            CS = (S[Q] @ S.T).toarray()
            prec = []
            for i in range(len(Q)):
                top = np.argsort(-CS[i])[:10]
                prec.append(len([t for t in top if t in cos10]) / 10.0)
            print(f"{str(center):6s} {on:4d} {nh:6d} | "
                  f"свой={on:3d} чужой={np.median(best_other):4.1f} m={margin:5.1f} | "
                  f"{thr99:6d}/{on:<4d} | {np.median((C >= thr99).sum(1)):7.2f} | "
                  f"{s_ov:5.1f} {u_ov:5.1f} {s_ov - u_ov:5.1f} | {np.mean(prec):5.2f}", flush=True)
