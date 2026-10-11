"""
Выбор конфигурации SDR-энкодера по двум осям:
  1. Семантический градиент: синоним >> орфо-двойник >> несвязанное.
  2. Безопасность фиксированного порога на РЕАЛЬНОМ русском словаре (hunspell ru_RU),
     а не на f"слово{i}" — синтетика сама по себе почти ортогональна и скрывает
     ложные срабатывания (я на этом почти обманулась).

Вопрос, который решает скрипт: выживает ли абсолютный UNKNOWN_THRESHOLD при росте
словаря, или порогу обязателен относительный (margin) вид.

Эмбеддинги пула считаются один раз и переиспользуются всеми конфигурациями.
"""
import os
import random

import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer

TOTAL, ON = 1024, 20
POOL = 41000
NQUERIES = 300

CHAIN = [("кошка", "кот"), ("кошка", "кошачьи"), ("кошка", "млекопитающее"),
         ("кошка", "животное"), ("кошка", "хищник"),
         ("кошка", "табуретка"), ("кошка", "кирпич"), ("кошка", "алгебра")]
SYN = [("грусть", "печаль"), ("дом", "жилище"), ("храбрый", "смелый"), ("бег", "беготня")]
ORTH = [("кот", "код"), ("дом", "том"), ("кот", "пот"), ("бок", "бок о бок")]
CALIB = ["кошка", "кот", "собака", "стол", "табуретка", "дом", "окно", "дверь", "хлеб",
         "вода", "огонь", "дождь", "лес", "поле", "гора", "море", "река", "рыба",
         "птица", "конь", "волк", "лиса", "мышь", "книга", "слово", "буква", "звук",
         "свет", "тень", "город", "улица", "машина", "колесо", "нож", "молоток",
         "кирпич", "цемент", "алгебра", "число", "любовь", "грусть", "страх", "гнев"]

print("Загрузка модели...", flush=True)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
DIM = model.get_sentence_embedding_dimension()


def diverse(sample, n):
    """Слова с разными корнями — иначе словарь из 20к форм одного слова хитрит метрику."""
    seen, out = set(), []
    for w in sample:
        k = w[:5]
        if k in seen:
            continue
        seen.add(k)
        out.append(w)
        if len(out) >= n:
            break
    return out


raw = [w.strip() for w in open("experiments/ru_words.txt", encoding="utf-8")
       if 3 <= len(w.strip()) <= 10]
random.seed(0)
random.shuffle(words := raw)
VOCAB = diverse(words, POOL)
print(f"Реальный словарь: {len(VOCAB)} слов из {len(raw)} (уникальный корень)\n", flush=True)

print("Кодирую пул один раз...", flush=True)
E = model.encode(VOCAB, normalize_embeddings=True, batch_size=512, show_progress_bar=False)
E = E.astype(np.float32)
CUR = np.array(CALIB, dtype=object)
E_cur = model.encode(CALIB, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
MEAN = E_cur.mean(0); MEAN /= np.linalg.norm(MEAN)


def to_sdr(Et, R, on=ON, chunk=4000):
    """top-`on` гиперплоскостей (Random Projection Pursuit) -> packed sparse CSR."""
    rows = []
    for i in range(0, len(Et), chunk):
        s = Et[i:i + chunk] @ R.T
        idx = np.argpartition(-s, on - 1, axis=1)[:, :on]
        rows.append(idx)
    idx = np.concatenate(rows)
    data = np.ones(idx.size, dtype=np.uint8)
    r = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    return sparse.csr_matrix((data, (r, idx.ravel())), shape=(idx.shape[0], TOTAL))


def overlap_counts(qcsr, lcsr):
    return (qcsr @ lcsr.T).toarray().astype(np.int32)


def cfg_encode(center, seed):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((NHYPER, DIM))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    Et = E
    if center:
        Et = E - (E @ MEAN)[:, None] * MEAN
        Et /= np.maximum(np.linalg.norm(Et, axis=1, keepdims=True), 1e-9)
    Ec = E_cur if not center else None
    if center:
        Ec = E_cur - (E_cur @ MEAN)[:, None] * MEAN
        Ec /= np.maximum(np.linalg.norm(Ec, axis=1, keepdims=True), 1e-9)
    return to_sdr(Et, R), to_sdr(Ec, R)


CFGS = [("1024 гиперпл, без центр.", 1024, False),
        ("1024 гиперпл, центр.", 1024, True),
        ("4096 гиперпл, центр.", 4096, True),
        ("16384 гиперпл, центр.", 16384, True)]

for name, NHYPER, CTR in CFGS:
    V, CURR = cfg_encode(CTR, seed=1)

    # ── 1. градиент на курируемых парах ──
    uni = list(dict.fromkeys([w for p in CHAIN + SYN + ORTH for w in p]))
    sub = [i for i, w in enumerate(VOCAB) if w in set(uni)]
    lab = [VOCAB[i] for i in sub]
    Vsub = V[sub]
    O = (Vsub @ Vsub.T).toarray()
    ix = {w: lab.index(w) for w in lab}
    ov = lambda a, b: int(O[ix[a], ix[b]]) if a in ix and b in ix else -1
    ch = [ov(a, b) for a, b in CHAIN]
    sy = [ov(a, b) for a, b in SYN]
    orr = [ov(a, b) for a, b in ORTH]
    print(f"── {name}")
    print(f"   кошка→ [кот {ch[0]}, кошачьи {ch[1]}, млеко {ch[2]}, животное {ch[3]}, "
          f"хищник {ch[4]}] | шум: табуретка {ch[5]}, кирпич {ch[6]}, алгебра {ch[7]}")
    miss = [p for p in SYN + ORTH if ov(*p) < 0]
    if miss:
        print(f"   (нет в выборке: {[p for p in miss]})")
    print(f"   синонимы {sy} (ср {np.mean(sy):.1f})   орфо-двойники {orr} (ср {np.mean(orr):.1f})")
    sep = np.mean(sy) > np.mean(orr)
    print(f"   разделение смысл/орфография: {'OK' if sep else 'ПЛОХО'}")

    # ── 2. ложные срабатывания на реальном словаре ──
    for N in (1000, 10000, 40000):
        lo, hi = N, N + N * 3
        lib_idx = [i for i in range(lo, min(hi, len(VOCAB)))][:N]
        q_idx = [i for i in range(0, lo)][:NQUERIES]
        C = overlap_counts(V[q_idx], V[lib_idx])
        best, p99 = np.median(C.max(1)), np.percentile(C.max(1), 99)
        ge = lambda t: np.median((C >= t).sum(1))
        print(f"   N={N:6d} | лучший overlap: медиана {best:4.1f}, p99 {p99:4.1f} "
              f"| чужих с ov>=3: {ge(3):7.1f}  ov>=5: {ge(5):6.1f}  ov>=7: {ge(7):5.1f}")
    print(flush=True)
