"""
Решающий эксперимент: СХЕМА бинаризации, а не только модель.

Метрика — не курируемые пары (их 10, они подтасовываются), а сохранение
косинусной близости ВСЕГО реального словаря:
  * Spearman corr(overlap_SDR, cosine) на случайных парах слов.
  * Шумовой пол: медиана max overlap чужого слова при N узлов.
  * Recall@10: долю истинных косинусных соседей, которых SDR тоже зовёт в топ-10.

Схемы:
  topk            — активны 20 гиперплоскостей с наибольшей проекцией (текущая).
                    Проблема: все 20 бита «спорят» за один и тот же конец
                    эмбеддинг-пространства, биты коррелированы.
  block_winner    — 20 блоков по 512 гиперплоскостей, в каждом winner (1 бит).
                    Биты структурно декоррелированы: каждый отвечает за свой
                    подсрез пространства. Exactly-20 по построению.
  block_topk      — 4 блока x 5 бит.
  block_lsh       — block_winner, но winner по знаку (контроль: должно быть хуже).
"""
import os
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

TOTAL, ON = 1024, 20
NPY = "experiments/pool_emb.npy"


def load_pool(model):
    if os.path.exists(NPY):
        return np.load(NPY)
    words = [w.strip() for w in open("experiments/ru_words.txt", encoding="utf-8")
             if 3 <= len(w.strip()) <= 10]
    seen, pool = set(), []
    for w in words:
        if w[:5] in seen:
            continue
        seen.add(w[:5])
        pool.append(w)
        if len(pool) >= 20000:
            break
    E = model.encode(pool, normalize_embeddings=True, batch_size=512,
                     show_progress_bar=False).astype(np.float32)
    np.save(NPY, E)
    np.save(NPY.replace(".npy", "_words.npy"), np.array(pool, dtype=object))
    return E


model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
DIM = model.get_sentence_embedding_dimension()
E = load_pool(model)
WORDS = np.load(NPY.replace(".npy", "_words.npy"), allow_pickle=True)
print(f"Пул: {len(E)} реальных русских слов, dim={DIM}\n", flush=True)

CENT = E.mean(0)
CENT /= np.linalg.norm(CENT)
Ec = E - (E @ CENT)[:, None] * CENT
Ec /= np.maximum(np.linalg.norm(Ec, axis=1, keepdims=True), 1e-9)


def encode(Et, R, scheme, on=ON):
    proj = Et @ R.T
    n = proj.shape[0]
    idx = np.zeros((n, on), dtype=np.int32)
    if scheme == "topk":
        idx[:] = np.argpartition(-proj, on - 1, axis=1)[:, :on]
    elif scheme in ("block_winner", "block_lsh"):
        nb, bw = on, proj.shape[1] // on
        P = proj[:, :nb * bw].reshape(n, nb, bw)
        if scheme == "block_winner":
            idx[:] = (bw * np.arange(nb))[None, :] + P.argmax(2)
        else:                                             # winner по |значению|
            idx[:] = (bw * np.arange(nb))[None, :] + np.abs(P).argmax(2)
    elif scheme == "block_topk":
        per = on // 4
        nb, bw = 4, proj.shape[1] // 4
        P = proj[:, :nb * bw].reshape(n, nb, bw)
        pick = np.argsort(-P, axis=2)[:, :, :per]
        idx[:] = ((bw * np.arange(nb))[:, None] + pick).reshape(n, on)
    out = np.zeros((n, TOTAL), dtype=np.uint8)
    if scheme in ("topk", "block_winner", "block_lsh", "block_topk"):
        # индекс бит = winner-гиперплоскость mod 1024
        idx = idx % TOTAL
        out[np.arange(n)[:, None], idx] = 1
    return out


def metrics(mask, tag):
    rows, coll = np.nonzero(mask)
    S = sp.csr_matrix((np.ones(coll.size, dtype=np.uint8), (rows, coll)),
                      shape=(mask.shape[0], TOTAL))
    rng = np.random.default_rng(0)
    qa, qb = rng.choice(len(E), 3000, replace=False), rng.choice(len(E), 3000, replace=False)
    keep = qa != qb
    qa, qb = qa[keep], qb[keep]
    ov = np.array([int((mask[i] & mask[j]).sum()) for i, j in zip(qa[:1500], qb[:1500])])
    cos = np.einsum("ij,ij->i", E[qa[:1500]], E[qb[:1500]])
    rho = spearmanr(ov, cos).statistic
    # шумовой пол
    q = rng.choice(len(E), 200, replace=False)
    lib = rng.permutation(len(E))[:8200]
    lib = np.setdiff1d(lib, q)[:8000]
    C = (S[q] @ S[lib].T).toarray()
    best = C.max(1)
    # recall@10 настоящих косинусных соседей
    sub = rng.choice(len(E), 1500, replace=False)
    Ssub = S[sub]
    Sim = (Ec[sub] @ Ec.T)
    Sim[np.arange(len(sub)), sub] = -9.0
    cos_top = np.argsort(-Sim, axis=1)[:, :10]
    sdr_top = np.argsort(-(Ssub @ S.T).toarray(), axis=1)[:, :10]
    hit = np.mean([len(set(cos_top[i]) & set(sdr_top[i])) / 10 for i in range(150)])
    print(f"  {tag:26s} spearman={rho:5.3f}  recall@10={hit:5.2f}  "
          f"шумовой пол: медиана best={np.median(best):4.1f} p99={np.percentile(best,99):4.1f}")
    return rho


for nh in (1024, 4096, 16384):
    rng = np.random.default_rng(1)
    R = rng.standard_normal((nh, DIM)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    print(f"── n_hyper={nh}")
    for scheme in ("topk", "block_winner", "block_topk", "block_lsh"):
        for ctr in (False, True):
            Et = Ec if ctr else E
            tag = f"{scheme}{' центр' if ctr else '       '}"
            try:
                metrics(encode(Et, R, scheme), tag)
            except Exception as exc:
                print(f"  {tag:26s} СБОЙ {type(exc).__name__}: {exc}")
    print(flush=True)
