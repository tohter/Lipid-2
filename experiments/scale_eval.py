"""Выбор конфигурации: иерархический градиент + ложные срабатывания на масштабе."""
import numpy as np, random
from sentence_transformers import SentenceTransformer

TOTAL, ON = 1024, 20
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def build(nhyper, center, seed=1, dim=384):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((nhyper, dim)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    calib = """кошка собака дерево стол табуретка дом окно любовь грусть страх бег сон
хлеб вода огонь дождь лес поле гора море рыба птица конь волк слово звук свет тень
машина город река книга буква звук тень ночь день друг враг голос слух глаз рука""".split()
    M = model.encode(calib, normalize_embeddings=True).mean(axis=0); M /= np.linalg.norm(M)
    return R, (M if center else None)

def enc(words, R, M, on=ON):
    E = model.encode(list(words), normalize_embeddings=True)
    if M is not None:
        E = E - (E @ M)[:, None] * M
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    s = E @ R.T
    idx = np.argpartition(-s, on - 1, axis=1)[:, :on]
    out = []
    for cols in idx:
        m = np.zeros(TOTAL, dtype=np.uint8); m[cols] = 1; out.append(m)
    return out

CHAIN = [("кошка","кот"),("кошка","кошачьи"),("кошка","млекопитающее"),("кошка","животное"),
         ("кошка","табуретка"),("кошка","кирпич"),("кошка","алгебра")]
MORPH = [("кошка","кошки"),("кот","коты"),("бег","бегу"),("дом","дома")]

for nhyper, center in ((1024,False),(1024,True),(4096,True),(16384,True)):
    R, M = build(nhyper, center)
    def enc_w(ws): return enc(ws, R, M)
    # градиент
    words = list(dict.fromkeys([w for p in CHAIN+MORPH for w in p]))
    v = dict(zip(words, enc_w(words)))
    ov = lambda a,b: int((v[a] & v[b]).sum())
    chain = [ov(a,b) for a,b in CHAIN]; morph=[ov(a,b) for a,b in MORPH]
    # масштаб: ложные ≥3 на 50k случайныхEmbedding-слов из большого списка
    base = list(dict.fromkeys(" ".join(open('/dev/null').read() for _ in []).split())) or \
           [f"слово{i}" for i in range(50000)]
    q = enc_w(base[:50]); lib = enc_w(base[50:20050])
    L = np.packbits(np.array(lib, dtype=np.uint8), axis=1)
    Q = np.packbits(np.array(q, dtype=np.uint8), axis=1)
    fp=[]
    for i in range(len(Q)):
        c = np.unpackbits(Q[i][None,:] & L, axis=1).reshape(len(L),-1).sum(1)
        fp.append(c)
    fp=np.array(fp)
    med_ge3 = np.median((fp>=3).sum(1)); med_best = np.median(fp.max(1))
    print(f"nhyper={nhyper:6d} center={center!s:5s} | градиент {chain} | морфо {morph} | "
          f"N=20k: ложных>=3 медиана {med_ge3:5.1f}, лучший overlap медиана {med_best:.1f}")
