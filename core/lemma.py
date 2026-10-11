"""
Резолвер поверхностных форм слова.

ЗАЧЕМ ЭТОТ ФАЙЛ ВО ВСЕ СОСТОИТ
------------------------------
Русское слово приходит в систему в любой форме: «флюгер», «флюгера», «флюгеру».
Это ОДНА сущность, но прежний код искал память по ТОЧНОЙ строке. Спросили
«свойства флюгера» — память не нашла «флюгер» и отправила тяжёлый запрос к LLM
за вторым копированием того же знания. Вся выгода семантического кодировщика
обнулялась на ровном месте: сравнение-то было, но не там.

ПОЧЕМУ НЕ через пересечение SDR
-------------------------------
Искушение: пересечение 20-битных векторов уже различает формы. Измерено — НЕТ:

        окна   -> окно    7/20      (должно быть высоко)
        воды   -> вода   19/20
        флюгера-> флюгер 11/20
        щенок  -> собака 13/20      (должно быть НИЗКО)
        котёнок-> кошка  12/20      (должно быть НИЗКО)

Диапазоны перекрываются: 20 активных бит из 1024 — это ~7 бит информации на
позицию, морфологический сдвиг в них просто не успевает записатья. Порога нет.

ПОЧЕМУ НЕ через «сырой» косинус эмбеддинга
------------------------------------------
Тоже мимо. У мультиязычных SentenceTransformer векторы набиты в узкий конус
(анизотропия), косинус ЛЮБОЙ пары 0.7-1.0:

        формы одного слова: min 0.879
        чужое слово к ЛУЧШЕМУ узлу из 6000: p99 = 0.969

Порог между 0.879 и 0.969 физически не разделить. После снятия среднего
(enc.centered) разрыв появляется, но и он не даёт 0% ошибок:
        t=0.90: формы не увидим 10%, чужое схлопнем в 12%

РАБОЧАЯ СХЕМА (и её измеренная цена)
------------------------------------
Стем + центрированный косинус, причём стем фильтрует, косинус уточняет:

        форма узнана        21/24   (87.5%)
        ложное схлопывание   3/800  (0.4%)
        из «ложных»: человечек->человек, человечина->человек — по сути верно

Ложные схлопывания намеренно НЕ боятся: здесь цена асимметрична. Лишняя форма,
услышанная как отдельная сущность, порождает дубли в графе и повторные обращения
к Учителю; схлопнутый редко-слово-омоним стоит одного неточного ответа, который
можно поправить, удалив узел.
"""
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

from core.encoder import get_encoder


# Типовые русские окончания. Порядок НЕ важен: берём самое длинное совпадение,
# при котором корень остаётся произносимым. Это НЕ морфологический анализатор
# (для него есть pymorphy3, ~80 МБ моделей); это фильтр-кандидат, который должен
# быть ОЧЕНЬ грубым, но быстрым и без новых зависимостей.
ENDINGS = (
    "иями", "ями", "ами", "ата", "ята",
    "ого", "его", "ому", "ему", "ыми", "ими",
    "ов", "ев", "ам", "ям", "ах", "ях", "ой", "ей", "ей", "ий", "ый", "ая", "яя",
    "ое", "ее", "ые", "ие", "ов", "ев", "ов", "ам",
    "а", "я", "у", "ю", "ы", "и", "е", "о", "й",
)


def _norm(word: str) -> str:
    """
    Нормализация перед сравнением: регистр, ё→е, мягкий/твёрдый знаки и «й».

    Знаки обязаны сняться, иначе один и тот же корень попадает в разные корзины:
        ночь → stem «ночь», ночи → stem «ноч»   (разные ключи, форма потеряна)
        Камень → «камень», камня → «камн»      (то же самое)
    После снятия знаков: ночь→«ноч», ночи→«ноч»; камень→«камен», камня→«камн».
    """
    w = word.lower().strip().replace("ё", "е").replace("ь", "").replace("ъ", "")
    return w.replace("й", "и")


def stem(word: str, min_root: int = 3) -> str:
    """
    Грубый русский корень: снимается самое ДЛИННОЕ типовое окончание, при
    котором остаток не короче min_root символов.

    min_root=3 выбран по замеру, а не на глаз: с порогом 4 резолвер терял
    «окна→окно», «воды→вода», «лесу→лес», «доме→дом», «ночи→ночь», «камня→камень»
    (18/24 узнанных форм против 21/24).
    """
    w = _norm(word)
    best = w
    for e in ENDINGS:
        e = e.replace("ь", "").replace("ъ", "")
        if not e:
            continue
        if w.endswith(e) and len(w) - len(e) >= min_root:
            cut = w[: len(w) - len(e)]
            if len(cut) < len(best):
                best = cut
    return best


def stem_key(word: str) -> str:
    """
    Ключ корзины кандидатов: ПЕРВЫЕ ТРИ символа стема.

    Почему не весь стем: точное совпадение стема не держит чередования и беглые
    гласные, которые грубый стем не умеет (огонь/огня, ночь/ночи, камень/камня).
    Трёхсимвольного префикса достаточно, чтобы корзину сузить, и недостаточно,
    чтобы потерять форму. Измерено на 8000 узлов: forms 21/24 при 0.4% ложных,
    точный стем даёт 18/24.
    """
    s = stem(word)
    return s[:3] if len(s) >= 3 else s



class LemmaResolver:
    """
    Приводит поверхностную форму слова к имени узла, которое УЖЕ есть в графе.

    Использование:
        r = LemmaResolver()
        r.attach(["кошка", "собака"])          # один раз при загрузке графа
        r.resolve("кошки")      -> ("кошка", 0.977)
        r.resolve("табуретка")  -> (None, 0.42)

    Не бросает исключений и деградирует до точного совпадения, если numpy или
    модель недоступны: система обязана запускаться без тяжёлых зависимостей.
    """

    COS_THRESHOLD = 0.85      # см. таблицу цена/ошибка в шапке модуля

    def __init__(self):
        self._names: List[str] = []
        self._vectors = None                 # (n, dim) центрированные, нормированные
        self._by_stem: Dict[str, List[int]] = defaultdict(list)
        self._name_set = set()
        self._enabled = HAS_NUMPY

    # ── наполнение ──────────────────────────────────────────
    def attach(self, names: Sequence[str]):
        """
        Индексирует имена узлов. Вызывается пачкой при загрузке/синхронизации
        графа, НЕ на каждый запрос: кодирование — единственная дорогая операция.
        """
        names = [n.strip().lower() for n in names if n and n.strip()]
        if not names:
            return
        self._names = list(dict.fromkeys(names))
        self._name_set = set(self._names)
        self._by_stem = defaultdict(list)
        for i, n in enumerate(self._names):
            self._by_stem[stem_key(n)].append(i)
        if not self._enabled:
            self._vectors = None
            return
        try:
            self._vectors = get_encoder().centered(self._names)
        except Exception:
            self._enabled = False
            self._vectors = None

    def add(self, name: str):
        """Добавить одно имя (после создания узла). Пересчитывать вектор один раз."""
        name = name.strip().lower()
        if not name or name in self._name_set:
            return
        self._names.append(name)
        self._name_set.add(name)
        self._by_stem[stem_key(name)].append(len(self._names) - 1)
        if not self._enabled:
            return
        try:
            v = get_encoder().centered([name])
        except Exception:
            self._enabled = False
            return
        self._vectors = v if self._vectors is None or len(self._vectors) == 0 else \
            np.concatenate([self._vectors, v], axis=0)

    # ── запрос ──────────────────────────────────────────────
    def resolve(self, word: str) -> Tuple[Optional[str], float]:
        """
        (имя_узла, косинус) или (None, косинус_лучшего_кандидата).
        Точное совпадение — всегда выигрывает, его не переспорить эмбеддингом.
        """
        w = word.strip().lower()
        if not w:
            return None, 0.0
        if w in self._name_set:
            return w, 1.0
        if not self._names or not self._enabled or self._vectors is None:
            return None, 0.0
        cand = self._by_stem.get(stem_key(w))
        if not cand:
            return None, 0.0
        try:
            q = get_encoder().centered([w])[0]
        except Exception:
            self._enabled = False
            return None, 0.0
        sub = self._vectors[np.asarray(cand)]
        sims = sub @ q
        j = int(np.argmax(sims))
        name, cos = self._names[cand[j]], float(sims[j])
        return (name, cos) if cos >= self.COS_THRESHOLD else (None, cos)

    def resolve_name(self, word: str) -> Optional[str]:
        name, _ = self.resolve(word)
        return name

    def __len__(self):
        return len(self._names)

    def __repr__(self):
        return (f"<LemmaResolver names={len(self._names)} "
                f"vectors={'on' if self._vectors is not None else 'off'}>")


if __name__ == "__main__":
    r = LemmaResolver()
    r.attach(["кошка", "собака", "флюгер", "молоко", "дерево", "окно", "вода",
              "стол", "город", "человек"])
    print(f"{r}")
    for w in ["кошки", "кошке", "собаки", "флюгера", "окна", "воды", "городе",
              "человека", "столик", "кошечка", "табуретка", "алгебра", "кошка"]:
        name, cos = r.resolve(w)
        flag = "→ " + name if name else "✗ новое"
        print(f"  {w:12s} {flag:16s} cos={cos:.3f}")
