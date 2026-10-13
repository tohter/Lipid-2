"""
core/morph.py — Морфологический слой: леммы, части речи, падежи, границы предложений.

ЗАЧЕМ ОН НУЖЕН, ЕСЛИ УЖЕ ЕСТЬ core/lemma.py
  LemmaResolver отвечает на «ФОРМА -> УЗЕЛ ГРАФА» через эмбеддинги: он нужен,
  когда морфология не уверена или слова в словаре нет. Но на ~95% русских слов
  словарь даёт лемму ТОЧНО и за микросекунды, без модели. Шуметь косинусом там,
  где есть словарь, — значит портить точность.
      morph.py  : «какая это часть речи, падеж, лемма»
      lemma.py  : «на какой узел графа это похоже»
  Там, где pymorphy3 не уверен (омонимия, низкий score), решение передаётся
  LemmaResolver — см. confidence() и candidates().

ГЛАВНОЕ СЛЕДСТВИЕ ДЛЯ ВЫЗВАЮЩИХ: parse() НЕ ЕДИНСТВЕННЫЙ ОТВЕТ
  pymorphy3 возвращает СПИСОК разборов, отсортированный по вероятности, и у
  омонимичных форм этот список принципиально неоднозначен:
      «грозы» -> [гроза:мн.ч.им.(0.43), гроза:ед.ч.род.(0.29), гроза:мн.ч.вин.(0.29)]
      «хвост» -> [хвост:ед.ч.вин.(0.57), хвост:ед.ч.им.(0.43)]
  Выбирать «самый вероятный» — ошибка: в «кошка боится грозы» нужен ВТОРОЙ
  вариант, хотя он вероятнее по словарю. Синтаксис предложения знает лучше
  частотного словаря, поэтому экстрактор обязан фильтровать варианты через
  candidates(), а не брать parse()[0]. (Я на это наступил: «кошка боится грозы»
  молча теряла объект, а «хвост собаки» разворачивал связь в обратную сторону.)

НОРМАЛИЗАЦИЯ ГРАММЕМ
  pymorphy3 отдаёт полные имена: 'nomn', 'accs', 'ablt', 'sing'. Сравнивать их
  с 'nom'/'acc' в правилах — гарантированная опечатка; я в неё и попал (все
  глагольные связи исчезли, потому что правило ждало 'acc', а падеж назывался
  'accs'). Нормализация делается ОДИН раз здесь, правила оперируют короткими
  именами.

ЕСЛИ pymorphy3 НЕ УСТАНОВЛЕН
  Модуль не падает и не молчит: он работает на эвристике по окончаниям,
  помечает available=False, и Learner честно пишет quality['morph']=False в
  отчёт. Причина та же, что с деградацией SDR на буквенном хэше: приложение не
  должно притворяться здоровым.
"""
from typing import Dict, FrozenSet, Iterable, List, Optional

try:
    from pymorphy3 import MorphAnalyzer
    _HAS_MORPH = True
except Exception:                        # ImportError и любая прочая порча импорта
    _HAS_MORPH = False

import re

# ── нормализация граммем (см. docstring) ───────────────────
CASE = {"nomn": "nom", "accs": "acc", "voct": "voc", "datv": "dat",
        "gent": "gen", "ablt": "ins", "loct": "loc", "gen1": "gen",
        "gen2": "gen", "acc2": "acc", "loc2": "loc", "": ""}
NUMBER = {"sing": "sg", "plur": "pl", "": ""}
GENDER = {"masc": "m", "femn": "f", "neut": "n", "mult": "pl", "": ""}

# Части речи, из которых получается узел графа.
CONTENT_POS = ("NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "GRND", "COMP", "NUMR")

# Союзы, предлоги, частицы, местоимения, глаголы-паразиты. pymorphy3 их
# трактует нестабильно (НУ/ЧТО/КАК омонимичны), поэтому список задан явно:
# «не-узел» должен решаться ДО разбора, а не после.
STOP = {
    # союзы и частицы
    "и", "а", "но", "да", "или", "тоже", "также", "же", "бы", "б", "как", "будто",
    "словно", "якобы", "ибо", "зато", "если", "ли", "лишь", "только", "хоть",
    "хотя", "просто", "ведь", "ну", "вот", "вон", "ведь", "всё", "все", "всю",
    # предлоги
    "в", "во", "с", "со", "у", "о", "об", "от", "до", "из", "изо", "к", "ко",
    "на", "над", "под", "по", "за", "перед", "пред", "между", "при", "про",
    "для", "без", "возле", "около", "вокруг", "после", "через", "внутри",
    # указательные и личные местоимения
    "это", "этот", "эта", "эти", "тот", "та", "те", "такой", "такая", "такие",
    "таком", "каком", "какой", "какая", "какие", "чей", "чья", "чьи",
    "он", "она", "оно", "они", "его", "её", "их", "мне", "тебе", "ему", "ей",
    "нас", "вас", "ими", "нами", "я", "ты", "мы", "вы", "себя", "кому", "чего",
    "кого", "чем", "ком", "нём", "ней", "него", "нею", "неё",
    "мой", "твой", "наш", "ваш", "свой", "твоя", "моей", "моего", "твоего",
    # наречия-паразиты и время
    "очень", "совсем", "весьма", "уж", "уже", "еще", "ещё", "тут", "там",
    "здесь", "сегодня", "вчера", "завтра", "сейчас", "потом", "снова", "опять",
    "всегда", "никогда", "иногда", "часто", "редко", "вдруг", "наконец",
    "конечно", "может", "быть", "было", "были", "будет", "буду", "будешь",
    "был", "была", "есть", "имеет", "имею", "имеешь", "имеются", "находится",
    "является", "тоже",
    # глаголы речи/мысли: это НЕ знание о мире, а обстановка разговора
    "хочу", "хочешь", "хотел", "хотела", "хотели", "хотеть", "могу", "может",
    "можешь", "мочь", "знаю", "знаешь", "знает", "знать", "думай", "думает",
    "думать", "понял", "поняла", "понять", "расскажи", "рассказал", "рассказать",
    "ответь", "ответить", "спроси", "спросить", "видел", "видела", "видеть",
    "слышал", "слышать", "кажется", "кажется", "какой",
    # светская вежливость
    "привет", "пока", "спасибо", "пожалуйста", "здравствуйте", "доброе",
    "добрый", "добрая", "утро", "вечер",
}

# Существительное/глагол, которые сами по себе ничего не значат в графе.
STOP_LEMMAS = {"быть", "являться", "представлять", "казаться", "стать",
               "становиться", "называться", "считаться", "иметь", "иметься",
               "статься", "нравиться", "надеть", "делать", "сделать", "стать"}


class MorphAnalyzerFacade:
    """Единая точка морфологического анализа. Формат возврата не зависит от того,
    установлен pymorphy3 или нет."""

    def __init__(self):
        self.available = _HAS_MORPH
        self._m = None
        self._cache: Dict[str, List[dict]] = {}
        if _HAS_MORPH:
            try:
                self._m = MorphAnalyzer()
            except Exception as exc:      # словарь бит/не скачан
                print(f"[morph]: pymorphy3 не инициализировался ({exc}); "
                      f"работаю на эвристике — качество разбора ниже")
                self.available = False

    # ── базовый разбор ─────────────────────────────────────

    def parse_all(self, word: str) -> List[dict]:
        """
        ВСЕ варианты разбора по убыванию вероятности.
        Каждый: {word, lemma, pos, case, number, gender, tense, person,
                 animacy, score, grammemes}.
        Смотри docstring модуля: выбирать [0] вместо фильтра по синтаксису —
        терять правильные факты на омонимичных формах.
        """
        key = word.lower().strip(".,!?;:«»\"'()")
        if key in self._cache:
            return self._cache[key]
        out: List[dict] = []
        if self._m is not None:
            try:
                for p in self._m.parse(key):
                    tag = p.tag
                    out.append({
                        "word": key,
                        "lemma": p.normal_form,
                        "pos": str(tag.POS) if tag.POS else "",
                        "case": CASE.get(str(tag.case), str(tag.case or "")),
                        "number": NUMBER.get(str(tag.number), str(tag.number or "")),
                        "gender": GENDER.get(str(tag.gender), str(tag.gender or "")),
                        "tense": str(tag.tense) if tag.tense else "",
                        "person": str(tag.person) if tag.person else "",
                        "animacy": str(tag.animacy) if tag.animacy else "",
                        "score": float(p.score) if p.score is not None else 0.5,
                        "grammemes": frozenset(str(g) for g in tag.grammemes),
                    })
            except Exception:
                out = []
        if not out:
            out = [self._parse_heuristic(key)]
        self._cache[key] = out
        return out

    def parse(self, word: str) -> dict:
        """Наиболее вероятный разбор. Когда синтаксис известен — używай parse_all/candidates."""
        return self.parse_all(word)[0]

    def _parse_heuristic(self, w: str) -> dict:
        """
        Аварийный разбор без словаря: часть речи по окончанию, лемма усечением.
        Это НЕ замена pymorphy3: падежа тут нет, а значит правила экстрактора
        деградируют до «свойство у ближайшего существительного».
        """
        pos, lemma = "", w
        if w.endswith(("ировать", "ствовать", "овать", "ывать", "ивать",
                       "ать", "ять", "еть", "ти", "ться")):
            pos, lemma = "VERB", w
        elif w.endswith(("ый", "ий", "ая", "яя", "ое", "ее", "ые", "ие", "ими", "ых", "их")):
            pos = "ADJF"
            for suf in ("иями", "ейми", "ыми", "ими", "ая", "яя", "ый", "ий",
                        "ое", "ее", "ые", "ие", "ых", "их", "ой"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    lemma = w[:-len(suf)]
                    break
        else:
            for suf in ("иями", "ейми", "ами", "ями", "ию", "ью", "ые", "ие",
                        "ов", "ев", "ам", "ям", "ого", "его", "ому", "ем",
                        "ах", "ях", "а", "я", "ы", "и", "у", "ю", "е", "о", "й"):
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    lemma = w[:-len(suf)]
                    break
            pos = "NOUN"
        return {"word": w, "lemma": lemma, "pos": pos, "case": "", "number": "",
                "gender": "", "tense": "", "person": "", "animacy": "",
                "score": 0.25, "grammemes": frozenset()}

    # ── выбор варианта по синтаксису ───────────────────────

    def candidates(self, token: dict, pos: Iterable[str] = (), case: str = "",
                   number: str = "") -> List[dict]:
        """
        Варианты разбора токена, подходящие под требуемую синтаксическую роль.
        Именно так омонимия превращается из шума в сигнал:
            candidates(tok, pos=('NOUN',), case='gen')   # «боится (чего?) грозы»
            candidates(tok, pos=('NOUN',), case='nom')   # «хвост (что?) мокрый»
        Пустой список = токен НЕ может быть этой ролью, выдумывать связь нельзя.
        """
        want_pos = set(pos)
        out = []
        for v in token["variants"]:
            if want_pos and v["pos"] not in want_pos:
                continue
            if case and v["case"] != case:
                continue
            if number and v["number"] and v["number"] != number:
                continue
            out.append(v)
        return out

    # ── производные вопросы ────────────────────────────────

    @staticmethod
    def role_from(info: dict) -> str:
        """Узловая роль РАЗБОРА: subject/property/action/stop.
        Смотри Learner: существительное -> subject, прилагательное -> свойство,
        глагол -> действие."""
        pos = info.get("pos", "")
        if pos in ("NOUN", "NUMR"):
            return "subject"
        if pos in ("ADJF", "ADJS", "COMP"):
            return "property"
        if pos in ("VERB", "INFN", "PRTF", "GRND"):
            return "action"
        return "stop"

    def is_stop(self, word_or_lemma: str) -> bool:
        w = word_or_lemma.lower().strip(".,!?;:«»\"'()")
        return w in STOP or w in STOP_LEMMAS

    def lemma(self, word: str) -> str:
        return self.parse(word)["lemma"]

    def pos(self, word: str) -> str:
        return self.parse(word)["pos"]

    def confidence(self, word: str) -> float:
        """1.0 — морфология уверена и вариант единственный; <0.5 — омонимия."""
        vs = self.parse_all(word)
        c = vs[0].get("score", 0.5) if vs else 0.0
        if len({v["lemma"] for v in vs}) > 1:
            c *= 0.7
        return round(min(1.0, c), 3)


Morph = MorphAnalyzerFacade()

# Многоточие — ПЕРВЫМ в разделителе, иначе «...» дало бы пачку пустых «предложений».
_SENT_SPLIT = re.compile(r"[.!?…]+")
# Числа ЦИФРАМИ раньше не попадали в токены ВООБЩЕ: регулярное выражение
# требовало буквы в начале, и «5000 рублей» превращалось в одинокое «рублей».
# ЗАМЕРЕНО: pymorphy3 разбирает «5000» как POS=None (не NUMR), поэтому
# недостаточно добавить цифры в regex — их часть речи выставляется в analyze()
# ЯВНО (см. NUMERIC_RE ниже). Иначе токен получит content=False и снова исчезнет,
# а цена останется неразобранной.
_WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9][а-яёА-ЯЁa-zA-Z0-9\-]*")
# Число цифрами (целое или десятичное), не приклеенное к буквам.
_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")


def split_sentences(text: str) -> List[str]:
    """Текст -> предложения. Пустые и односимвольные отбрасываются."""
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) >= 2]


def analyze(sentence: str) -> List[dict]:
    """
    Предложение -> токены:
      {word, variants:[...], content:bool, role:str, score:float}
    content=False у предлогов, союзов, местоимений и служебных глаголов —
    узлами они не становятся, но ПРЕДЛОГ всё равно нужен экстрактору (он
    решает «кошка НА подоконнике» vs «кошка ПОДОКОННИК»), поэтому токены
    остаются в списке.
    """
    toks = []
    for w in _WORD_RE.findall(sentence or ""):
        raw = w
        key = w.lower()
        variants = Morph.parse_all(key)
        numeric = bool(_NUMERIC_RE.match(key))
        if numeric:
            # Цифра — это число. pymorphy3 даёт для неё POS=None, поэтому
            # роль/тип проставляем сами: иначе токен невидим для экстрактора
            # и цена («5000 рублей») теряется, как было ЗАМЕРЕНО.
            variants = [{"pos": "NUMR", "lemma": key, "case": "",
                         "number": "", "gender": "", "score": 0.95}]
        has_content = any(v["pos"] in CONTENT_POS for v in variants)
        content = has_content and not Morph.is_stop(key)
        if numeric:
            content = True
        best = variants[0]
        toks.append({
            "word": raw.lower(),
            "variants": variants,
            "lemma": best["lemma"],
            "pos": best["pos"],
            "case": best["case"],
            "number": best["number"],
            "gender": best["gender"],
            "score": best.get("score", 0.5),
            "role": Morph.role_from(best) if content else "stop",
            "content": content,
        })
    return toks


if __name__ == "__main__":
    import sys
    print(f"pymorphy3: {'да' if Morph.available else 'НЕТ (эвристика!)'}")
    sent = " ".join(sys.argv[1:]) or "Кошка сидит на широком подоконнике."
    for s in split_sentences(sent):
        print(f"«{s}»")
        for t in analyze(s):
            opts = " | ".join(f"{v['lemma']}:{v['pos']}/{v['case']}{v['number']}"
                              f"({v['score']:.2f})" for v in t["variants"][:3])
            print(f"  {t['word']:14s} role={t['role']:9s} {opts}")
