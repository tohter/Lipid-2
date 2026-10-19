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

# СОЮЗНОЕ СЛОВО ПРИДАТОЧНОГО (очередь, п.18).
# ЗАМЕРЕНО: у ВСЕХ падежных форм «который» (который/которая/которые/которого/
# которой/которым/которых/которыми) pymorphy3 выдаёт граммема Subx+Anph+Apro, а
# у «какой», «чей», «пушистый», «дом», «бросить» их нет ни в одном варианте.
# Отсюда признак берётся из морфологии, а не из списка слов: список обязан
# перечислить восемь падежных форм и молча пропустит ту, которую автор списка
# не вспомнил (тот же приём, что с нормализацией граммем выше).
RELATIVE_GRAMMEMES = frozenset({"Subx", "Anph", "Apro"})

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
        """Наиболее вероятный разбор. Когда синтаксис известен — используй parse_all/candidates."""
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

    def is_relative_pronoun(self, word: str) -> bool:
        """
        Союзное слово придаточного: «пёс, КОТОРЫЙ охранял дом».

        Оно НЕ называет ничего: оно указывает на слово главного предложения и
        служит связкой. В графе у него узла быть не может.
        ЗАМЕРЕНО ДО ПРАВКИ (п.18): «пёс --HAS_PROPERTY--> который»,
        «мяч --HAS_PROPERTY--> который», «берег --HAS_PROPERTY--> который» — то
        есть служебное слово осело в онтологии как СВОЙСТВО сущности.

        БЕЗ pymorphy3 возвращается False, и поведение остаётся прежним (узлы из
        союзных слов). Это не тихая деградация: analyze() помечает available=False,
        Learner пишет quality['morph']=False в отчёт, а экстрактор на придаточном
        обязан оставить заметку (core.extract._has_subordinate).
        """
        if not _HAS_MORPH:
            return False
        w = word.lower().strip(".,!?;:«»\"'()")
        if not w:
            return False
        vs = self.parse_all(w)
        # all(), а не any(): вариант без Subx значит, что слово прочитано НЕ как
        # союзное (омонимия), а вычёркивать значащее слово опаснее, чем оставить
        # прежний шум — шум хотя бы объявлен заметкой.
        return bool(vs) and all(
            RELATIVE_GRAMMEMES <= set(v.get("grammemes") or ()) for v in vs)

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
    for m in _WORD_RE.finditer(sentence or ""):
        w = m.group()
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
        # СОЮЗНОЕ СЛОВО придаточного. Оно морфологически прилагательное (ADJF),
        # поэтому в content попадает, а узлом быть не должно — см.
        # RELATIVE_GRAMMEMES. Для экстрактора токен остаётся РАЗЛИЧИМЫМ (ключ
        # "relative"): по нему придаточное потом найдёт свою границу.
        relative = content and Morph.is_relative_pronoun(key)
        if relative:
            content = False
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
            "relative": relative,
            # Позиция слова в ПРЕДЛОЖЕНИИ. Без неё нельзя вернуть вырезанную
            # часть текста: морфология знает слова, а границы придаточного —
            # это запятые, то есть места МЕЖДУ словами (см. split_clauses).
            "start": m.start(),
            "end": m.end(),
        })
    return toks


# Союзы придаточного. Слой морфологии, потому что их читает и extract,
# и разметка токенов: два списка разъезжаются неизбежно.
# Это слова, которые морфология сама называет CONJ/ADVB (замер: «когда» ->
# CONJ|ADVB, «где» -> ADVB|CONJ, «чтобы» -> CONJ). Слова с падежными формами
# («который») здесь НЕ перечислены — они ловятся граммемами,
# см. RELATIVE_GRAMMEMES: список обязан был бы помнить все восемь падежей и
# молча пропустить тот, который автор списка не вспомнил.
# «после»/«перед» ИСКЛЮЧЕНЫ намеренно: как предлоги они стоят в любом
# «после обеда», и объявлять там придаточное — обесценивать предупреждение.
SUBORD_CONJ = ("когда", "пока", "чтобы", "где", "если", "хотя",
               "поскольку", "будто", "словно", "чем", "как")


def is_subordinate_marker(tok: dict) -> bool:
    """Токен — маркер придаточного: союзное слово или подчинительный союз."""
    return bool(tok.get("relative")) or (
        str(tok.get("word", "")).lower() in SUBORD_CONJ
        and tok.get("pos") in ("CONJ", "ADVB"))


def split_clauses(sentence: str) -> List[dict]:
    """
    Предложение -> части в ПОРЯДКЕ ТЕКСТА: [{"text":..., "sub": bool}].

    ЗАЧЕМ (замер, п.18). У сложноподчинённого предложения ищутся ОДИН субъект
    и ОДНО сказуемое на весь текст, и второе сказуемое остаётся без пары:
    «Мяч, который бросил мальчик, укатился» теряло «укатился» entirely, а
    «Дом, где жил пёс, сгорел» записывало «дом CAN_DO жила».

    ГРАНИЦА — не произвольная запятая (по запятой рассыпались бы и
    однородные члены, и аппозиции, и деепричастные обороты — а это уже
    поведение, зафиксированное другими тестами). Граница строится ВОКРУГ
    МАРКЕРА придаточного: союзное слово или подчинительный союз, ищутся
    ближайшие запятые слева и справа от него; без запятой часть идёт до конца
    предложения. Всё остальное предложение остаётся ОДНОЙ частью.

    «Река, берега которой заросли, замёрзла» — маркер «которой» стоит НЕ в
    начале части; левая запятая отстоит на два слова, и без расширания назад
    придаточным объявлялось бы «которой заросли» без субъекта.

    Одна запятая в тексте = и одна часть (придаточного нет) -> len(...)==1,
    и вызывающий работает ровно как раньше.
    """
    toks = analyze(sentence)
    if not toks:
        return []
    marks = [i for i, t in enumerate(toks) if is_subordinate_marker(t)]
    if not marks:
        return [{"text": sentence.strip(), "sub": False}]
    # Запятые/точка с запятой МЕЖДУ словами: граница части = позиция слова,
    # за которым такая запятая стоит.
    def comma_before(i: int) -> bool:
        if i <= 0:
            return False
        gap = sentence[toks[i - 1]["end"]:toks[i]["start"]]
        return "," in gap or ";" in gap

    spans = []
    for i in marks:
        start = i
        while start > 0 and not comma_before(start):
            start -= 1
        end = i + 1
        while end < len(toks) and not comma_before(end):
            end += 1
        spans.append((start, end))
    # вложенные/пересекающиеся маркеры склеиваются в одну часть
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    out, pos = [], 0
    for idx, (s, e) in enumerate(merged):
        if s > pos:
            out.append({"text": sentence[toks[pos]["start"]:toks[s]["start"]].strip(),
                        "sub": False})
        out.append({"text": sentence[toks[s]["start"]:toks[e - 1]["end"]].strip(),
                    "sub": True})
        pos = e
    if pos < len(toks):
        out.append({"text": sentence[toks[pos]["start"]:toks[-1]["end"]].strip(),
                    "sub": False})
    return [c for c in out if c["text"]]


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
