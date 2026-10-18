"""
core/extract.py — Извлечение фактов из простого русского предложения.

КАЗУС РЕШАЕТ ВСЁ, ПОРЯДОК СЛОВ — ПОЧТИ НИЧЕГО
  «рыбу съела кошка» и «кошка съела рыбу» — разные факты. Порядок слов в русском
  свободный, падеж не врёт. Поэтому роль существительного выводится из ЕГО
  ПАДЕЖА и предлога перед ним, а не из позиции в предложении.

  Вин.п. без предлога      -> OBJECT_OF   («съела рыбу»)
  Род.п.  без предлога     -> OBJECT_OF   («боится грозы») / PART_OF в бесглагольном («хвост собаки»)
  Тв.п.   без предлога     -> INSTRUMENT  («открывает мышь» — мышь орудие, не объект)
  Dat.п. без предлога      -> LOCATED_IN  («кошке холодно» — адресат)
  предлог места (в/на/у…)  -> LOCATED_IN
  предлог «для/про/без»    -> ABOUT
  ИМ.п. в именной части    -> IS_ROLE     («Вася — инженер»)
  Инфинитив после глагола -> PURPOSE   («пришёл увидеть»)

ОМОНИМИЯ СНИМАЕТСЯ СОГЛАСОВАНИЕМ, А НЕ ЧАСТОТОЙ
  У «грозы» словарь ставит выше «мн.ч.им.», но в «кошка боится грозы» нужен
  родительный единственного числа: глагол «боится» требует КОГО/ЧЕГО. Аналогично
  «хвост собаки мокрый»: сказуемое «мокрый» в ЕД.числе, значит «собаки» — не
  им.падеж множественного, а родительный единственного. Если бы я брал
  parse()[0] (самый вероятный вариант), эти два предложения дали бы неверные
  связи. Отсюда правило: варианты фильтруются синтаксисом (morph.candidates),
  а при нескольких подходящих вариантах связь помечается НЕУВЕРЕННОЙ (ambiguity).

ЧЕГО ЗДЕСЬ НЕТ — И ПОЧЕМУ
  Зависимого синтаксического разбора нет: Natasha/udapi — это ~200 МБ моделей,
  деградация которых молчалива, а нужны они пока только для придаточных и
  причастных оборотов. Правила выше проверяются по одному предложению, и ошибка
  видна сразу. Придаточные («кошка, которая спит, пушистая») метод держит плохо —
  это фиксируется заметкой, а не маскируется.

СТАТИКА vs СОБЫТИЯ (задача 32)
  Экстрактор выдаёт не только факты, но и СОБЫТИЯ. Граница — ГРАММАТИЧЕСКАЯ:
    * настоящее время / связка / модальность -> ФАКТ («кошки едят птиц»);
    * личный глагол в ПРОШЕДШЕМ времени     -> СОБЫТИЕ («кошка съела канарейку»).
  У события есть порядок (номер предложения) и время — это ДИНАМИКА рассказа,
  она идёт в отдельную секцию мира "events", а не в онтологию. Иначе один
  случай («съела») стал бы вечным утверждением («кошка ест канареек»).
  Признак финитности — наличие ВРЕМЕНИ, а не лица: pymorphy3 у глаголов
  прошедшего времени лицо НЕ выдаёт, а выдаёт род (замерено).
"""
if __package__ in (None, ""):        # прямой запуск: python core/extract.py
    # Тогда в sys.path лежит папка core/, а не корень проекта — без этого
    # собственный импорт core.* не найдётся.
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from core.morph import Morph, split_sentences, analyze, CONTENT_POS

# ── управление предлогов: какой падеж они требуют ──────────
# Без этой таблицы «на широком подоконнике» превратилось бы в «кошка =
# подоконник»: предлог меняет роль слова на обстоятельство.
PREP_CASES: Dict[str, Tuple[Tuple[str, ...], str]] = {
    # prep: (допустимые падежи, отношение)
    "в":       (("acc", "loc"), "LOCATED_IN"),
    "во":      (("acc", "loc"), "LOCATED_IN"),
    "на":      (("acc", "loc"), "LOCATED_IN"),
    "у":       (("gen",), "LOCATED_IN"),
    "о":       (("loc",), "ABOUT"),
    "об":      (("loc",), "ABOUT"),
    "от":      (("gen",), "LOCATED_IN"),
    "до":      (("gen",), "LOCATED_IN"),
    "из":      (("gen",), "LOCATED_IN"),
    "изо":     (("gen",), "LOCATED_IN"),
    "из-за":   (("gen",), "LOCATED_IN"),
    "к":       (("dat",), "LOCATED_IN"),
    "ко":      (("dat",), "LOCATED_IN"),
    "над":     (("ins",), "LOCATED_IN"),
    "под":     (("ins", "dat"), "LOCATED_IN"),
    "по":      (("dat",), "ABOUT"),
    "за":      (("ins", "acc"), "LOCATED_IN"),
    "перед":   (("ins",), "LOCATED_IN"),
    "пред":    (("ins",), "LOCATED_IN"),
    "между":   (("ins",), "LOCATED_IN"),
    "при":     (("loc",), "LOCATED_IN"),
    "про":     (("acc",), "ABOUT"),
    "для":     (("gen",), "ABOUT"),
    "без":     (("gen",), "ABOUT"),
    "возле":   (("gen",), "LOCATED_IN"),
    "около":   (("gen",), "LOCATED_IN"),
    "вокруг":  (("gen",), "LOCATED_IN"),
    "после":   (("gen",), "LOCATED_IN"),
    "через":   (("acc",), "ABOUT"),
    "внутри":  (("gen",), "LOCATED_IN"),
    "при":     (("loc", "dat"), "CONDITION"),
}
ALL_PREPS = set(PREP_CASES)

# Глаголы-связки: значения не несут, соединяют подлежащее с именной частью.
# «кошка есть животное» — это IS_A, а не действие «есть».
COPULAS = {"быть", "являться", "представлять", "казаться", "стать", "становиться",
           "называться", "считаться", "иметь", "иметься", "статься", "нравиться"}

NEGATIONS = {"не", "ни"}
VERB_POS = ("VERB", "INFN", "PRTF", "GRND")

# Обстоятельственные существительные. Их НЕЛЬЗЯ выдавать за инструмент:
# «солнце светит днём» — «днём» не орудие свечения (я именно так и получал
# INSTRUMENT день). Это отдельный класс: время и мера отвечают на «когда» и
# «сколько», а не «чем».
TIME_LEMMAS = {"день", "ночь", "утро", "вечер", "час", "минута", "секунда",
               "год", "месяц", "неделя", "зима", "весна", "лето", "осень",
               "время", "пора", "утра", "полдень", "полночь", "vek", "век",
               "сезон", "период", "момент", "сутки"}
MEASURE_LEMMAS = {"грамм", "килограмм", "тонна", "метр", "сантиметр", "километр",
                  "литр", "градус", "процент", "рубль", "копейка", "доллар",
                  "евро", "цент", "киловатт", "вольт", "ампер", "герц", "паскаль"}
MONEY_LEMMAS = {"рубль", "копейка", "доллар", "евро", "цент", "цена", "стоимость",
                "деньги", "тысяча", "миллион", "миллиард"}

# ── ЧИСЛИТЕЛЬНЫЕ И МЕРЫ (очередь, п.17) ─────────────────────────────────────
# Зачем. ЗАМЕРЕНО: «Шкаф стоит пять тысяч рублей» уходило в примечание
# «значение цены не сохранено (нет разбора числительных)», а «Кошка весит три
# килограмма» рождало ЛОЖНУЮ валентность: CAN_DO(кошка, весить) +
# OBJECT_OF(весить, килограмм). То есть мера записывалась как умение и объект,
# а само число терялось. Здесь словарь ЕДИНИЦ и МНОЖИТЕЛЕЙ: «пять тысяч» =
# 5 * 1000. Без множителей «пять тысяч рублей» дало бы 5 или 1000, а не 5000.
NUM_ONES = {"ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2,
            "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
            "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
            "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
            "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
            "восемнадцать": 18, "девятнадцать": 19}
NUM_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
            "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80,
            "девяносто": 90}
NUM_HUNDREDS = {"сто": 100, "двести": 200, "триста": 300,
                "четыреста": 400, "пятьсот": 500, "шестьсот": 600,
                "семьсот": 700, "восемьсот": 800, "девятьсот": 900}
# Множители: «тысяча»/«миллион» — это И существительные, И разряды числа.
NUM_MULTIPLIERS = {"тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
                   "миллион": 10 ** 6, "миллиона": 10 ** 6, "миллионов": 10 ** 6,
                   "миллиард": 10 ** 9, "миллиарда": 10 ** 9,
                   "миллиардов": 10 ** 9}
# Глаголы, после которых число/мера — это ЗНАЧЕНИЕ, а не умение.
VALUE_VERBS = {"стоить", "весить", "обойтись", "потянуть", "равняться",
               "составлять", "достигать", "насчитывать"}

# Модальные глаголы: модальность -> HAS_PROPERTY, инфинитив -> CAN_DO
# (наблюдаемое действие уже даёт CAN_DO через основную логику)
MODAL_VERBS = {
    "уметь", "мочь", "хотеть", "пытаться",
    "смочь", "захотеть", "захотеться",
    "знать", "успеть", "начать", "перестать", "продолжать",
    "должен", "надо", "нужно", "следует",
}

# Глаголы ПРОСЬБЫ/ЗАПРЕТА. Конструкция «X попросил Y <инфинитив>» задаёт
# ОЖИДАНИЕ: X ждёт, что Y сделает (или не сделает) инфинитивное действие.
# Это не догадка о смысле — это устойчивый синтаксис, проверяемый падежом:
# субъект в ИМ.п., адресат в ВИН.п., зависимый инфинитив.
REQUEST_VERBS = {
    "просить", "попросить", "умолять", "велеть", "приказать",
    "запретить", "запрещать", "требовать", "потребовать",
    "уговаривать", "уговорить",
}

# Глаголы ЗАПРЕТА: ожидание уже СОДЕРЖИТ отрицание («запретил разговаривать»
# = ждёт, что НЕ будет разговаривать) без частицы «не» в предложении.
FORBID_VERBS = {"запретить", "запрещать"}

@dataclass
class Fact:
    head: str
    relation: str
    tail: str
    evidence: str = ""
    negated: bool = False
    confidence: float = 0.6
    ambiguous: bool = False       # Suitable variants > 1: связь правдоподобна, но не доказана
    # СТАТИКА или ДИНАМИКА. "fact" — «что есть» (связка, свойство, наст. время);
    # "event" — «что произошло» (личный глагол в прошедшем времени). Нужно,
    # чтобы событие рассказа можно было отличить от знания и не свалить его в
    # онтологию (см. experiments/EVENTS_TASK.md).
    predicate: str = "fact"

    def key(self) -> Tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    @property
    def is_event(self) -> bool:
        return self.predicate == "event"

    def __repr__(self) -> str:
        flags = "".join([" /НЕ/" if self.negated else "",
                         " /?" if self.ambiguous else "",
                         " /СОБЫТИЕ/" if self.is_event else ""])
        return f"{self.head} --{self.relation}{flags}--> {self.tail}"


@dataclass
class EventRecord:
    """
    СОБЫТИЕ рассказа: единица ДИНАМИКИ, а не онтологии.

    Почему отдельно от Fact: «кошка СЪЕЛА канарейку» — не знание о кошке, а
    происшествие во времени. Смешать их — значит научить базу утверждать
    «кошка ест канареек» из одного случая, а симулятор оставить без порядка
    событий (он и рисовался руками: sim.add_event).

    Поля: субъект/действие/объект (леммы), tense (past/pres/иной),
    order — номер предложения в рассказе, evidence — исходная фраза.
    """
    subject: str
    action: str
    obj: str = ""
    tense: str = ""
    order: int = 0
    evidence: str = ""
    negated: bool = False

    def describe(self) -> str:
        tail = f" -> {self.obj}" if self.obj else ""
        return f"{self.subject} {self.action}{tail}"


@dataclass
class ExpectationRecord:
    """
    ОЖИДАНИЕ, выведенное ИЗ ТЕКСТА (не заданное директивой).

    Фраза «Кошка попросила канарейку не рассказывать тайну» сообщает
    ожидание: holder=кошка, target=канарейка, action=рассказывать,
    negated=True. Все четыре части уже есть в разборе предложения:
    субъект просьбы, её объект, зависимый инфинитив и частица «не».

    ЗАЧЕМ отдельная запись, а не директива expect в рассказе: рассказ — это
    ТЕКСТ, и ожидание должно читаться из него, иначе модель «знает» то,
    чего в тексте не было, а человек видит выдумку.
    """
    holder: str
    target: str
    action: str
    obj: str = ""
    negated: bool = True
    order: int = 0
    evidence: str = ""
    source: str = ""

    def describe(self) -> str:
        neg = "не " if self.negated else ""
        tail = f" {self.obj}" if self.obj else ""
        return f"{self.holder} ждёт, что {self.target} {neg}{self.action}{tail}"


# Часть речи, означающая ДЕЙСТВИЕ. ИНФИНИТИВ — отдельная часть речи (INFN),
# и именно им записана ЛЕММА любого глагола: «гасит» -> лемма «гасить».
# Проверять только startswith("VERB") значит потерять инфинитивы: на
# «Нужно гасить пожар» глагол не попадал в действие, оставался в словаре как
# обычное слово и отправлялся учиться КАК СУЩНОСТЬ — ровно тот дефект, от
# которого action_lemmas и был заведён.
VERB_POS = ("VERB", "INFN")


def _is_verb_pos(pos) -> bool:
    return str(pos or "").startswith(VERB_POS)


@dataclass
class SentenceFacts:
    """Разбор одного предложения: факты, значимые леммы, замечания о неуверенности."""
    sentence: str
    facts: List[Fact] = field(default_factory=list)
    events: List[EventRecord] = field(default_factory=list)
    expectations: List[ExpectationRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    tokens: List[dict] = field(default_factory=list)

    @property
    def content_lemmas(self) -> List[str]:
        """Значимые леммы без повторов, в порядке появления."""
        seen, out = set(), []
        for t in self.tokens:
            if t["content"] and t["lemma"] not in seen:
                seen.add(t["lemma"])
                out.append(t["lemma"])
        return out

    @property
    def min_confidence(self) -> float:
        return round(min((f.confidence for f in self.facts), default=0.0), 3)

    @property
    def action_lemmas(self) -> List[str]:
        """
        Глаголы предложения, отдельно от остальных значимых слов.

        Нужны, чтобы ОТДЕЛИТЬ глагол от существительного. Глагол — не «ещё
        одна сущность с иерархией и свойствами», а отношение: узел ACTION со
        связями CAN_DO / OBJECT_OF / DIRECTION. ЗАМЕРЕНО: «катиться» и «гасить»
        гнались через Википедию и LLM как существительные; статью-онтологию на
        глагол Википедия не даёт, слово получало asked_empty и БОЛЬШЕ не
        спрашивалось — честный вид отказа, наложенный на неуместный вопрос,
        навсегда снимал слово с изучения.
        """
        seen, out = set(), []
        for t in self.tokens:
            if t["content"] and _is_verb_pos(t.get("pos")) and t["lemma"] not in seen:
                seen.add(t["lemma"])
                out.append(t["lemma"])
        return out


class SimpleExtractor:
    """
    Морфологический экстрактор: детерминирован, без LLM и без обучаемых весов.
    Каждый метод — один шаг разбора, поэтому проверяется изолированно.
    """

    def __init__(self, known_lemmas: Optional[set] = None):
        # Леммы, которые ядро уже знает. Нужны только для снятия омонимии
        # при равных вероятностях (см. _pick).
        self.known = set(known_lemmas or ())

    def set_known(self, lemmas: Iterable[str]) -> None:
        self.known = set(lemmas)

    # ── публичный вход ─────────────────────────────────────

    def extract_text(self, text: str) -> List[SentenceFacts]:
        out = []
        # ПОРЯДОК ВО ВРЕМЕНИ = ПОРЯДОК ПРЕДЛОЖЕНИЙ. Рассказ читается слева
        # направо; событию присваивается индекс предложения. Это единственный
        # доступный признак времени без разметки дат (см. EVENTS_TASK.md).
        for idx, s in enumerate(split_sentences(text)):
            sf = self.extract(s, order=idx)
            if sf.facts or sf.content_lemmas or sf.events:
                out.append(sf)
        return out

    def extract(self, sentence: str, order: int = 0) -> SentenceFacts:
        res = SentenceFacts(sentence=sentence.strip())
        toks = self._tag_prepositions(analyze(sentence))
        res.tokens = toks

        content = [t for t in toks if t["content"]]
        if not content:
            res.notes.append("нет значимых слов")
            return res

        neg = any(t["word"] in NEGATIONS for t in toks)
        # Ценовая конструкция: глагол «стоить» + деньги. Отличать надо от
        # действия, иначе «шкаф» получит умение «стоить».
        money_sentence = self._is_price(content)
        # Мера вообще (вес/длина/объём/цена): глагол меры + число или единица.
        # ЗАЧЕМ ОТДЕЛЬНО: «Кошка весит три килограмма» НЕ ценовое, но тоже
        # рождало ложное CAN_DO(кошка, весить) — мера должна давать VALUE,
        # а не умение, и для этого нужен свой флаг.
        value_sentence = self._is_value_sentence(content)
        # Именная связка без глагола: «Вася — это инженер», «кошка это животное».
        # Частицу «это» и тире мы ловим ЯВНО, потому что морфология тут бессильна:
        # оба существительных стоят в ИМ.п. и согласуются в числе, так что по
        # падежу подлежащее не выбирается вообще. Выбор делает связка.
        copula_marker = ("—" in sentence or " - " in sentence
                         or any(t["word"] in ("это", "есть") or t["lemma"] == "это"
                                for t in toks))

        # Подлежащее и сказуемое ВЫБИРАЮТСЯ ДРУГ ПО ДРУГУ и оба требуют ИМ.п.,
        # поэтому порядок важен. В «хвост собаки мокрый» без числа сказуемого
        # «хвост» и «собаки» неразличимы; со «мокрый» (ед.ч.) подлежащее
        # однозначно. Отсюда: грубое сказуемое -> подлежащее -> сказуемое с учётом
        # подлежащего. Если второй шаг не нашёл сказуемого, подлежащее не
        # согласовалось с ним — оставляем пустым, а не цепляем первый попавшийся
        # ИМ.п.: ложная связь стоит дороже отсутствующей.
        pred = self._predicate(content, None)
        subj = self._subject(content, pred)
        pred = self._predicate(content, subj) or pred
        if subj is None and pred is not None and pred["role"] != "property":
            # подлежащего нет, а сказуемое — личный глагол: актор неизвестен,
            # приписывать действие первому существительному нельзя
            pred = None

        if subj is None and pred is None:
            res.notes.append("нет ни подлежащего (ИМ.п.), ни сказуемого")
            return res

        # Актор может быть выражен МЕСТОИМЕНИЕМ («он купил автомобиль»), которое
        # узлом графа не является. Тогда подлежащее из существительных искать
        # НЕЛЬЗЯ: «автомобиль» формально стоит в ИМ.п. (0.57 против 0.43 вин.)
        # и стал бы субъектом — то есть мы записали бы, что автомобиль умеет
        # покупать. Инверсия агент/объект — самый дорогой тип ошибки: она
        # воспроизводится во всех последующих ответах.
        pronoun_actor = self._pronoun_actor(toks, pred)
        if pronoun_actor is not None:
            subj = None
            res.notes.append("актор — местоимение: действие записано без субъекта")

        subj_lemma = subj["lemma"] if subj else ""
        subj_taken = id(subj)
        num = self._predicate_number(pred)

        # 1) подлежащее -> сказуемое (действие или свойство)
        if pred is not None:
            kind = self._clause_pred_role(pred)
            if kind == "action" and subj is not None and (money_sentence
                                                          or value_sentence):
                # «шкаф стоит 5000 рублей», «кошка весит три килограмма»:
                # глагол меры — это ЗНАЧЕНИЕ, а не умение субъекта. Умения в
                # графе используются для ответов «что умеет X», и цена/вес там
                # как умение выглядят абсурдно (ЗАМЕРЕНО: «кошка CAN_DO
                # весить» + «весить OBJECT_OF килограмм»).
                res.notes.append("цена/мера, а не действие")
            elif kind == "action" and subj is not None:
                res.facts.append(self._fact(subj, pred, "CAN_DO", res.sentence, neg))
                if pred.get("prep"):
                    res.facts.append(Fact(subj["lemma"], "ABOUT",
                                          pred.get("prep") + " " + pred["lemma"],
                                          evidence=res.sentence, negated=neg,
                                          confidence=0.4))
            elif kind == "property" and subj is not None:
                res.facts.append(self._fact(subj, pred, "HAS_PROPERTY", res.sentence, neg))
            # именная часть при связке («Вася — это инженер»)
            if pred is not None and self._is_copula_form(pred):
                # «Вася является инженером»: Тв.п. после СВЯЗКИ — именная часть,
                # а не орудие. Тот же Тв.п. после смыслового глагола («открыла
                # мышью») остаётся инструментом, поэтому проверять надо здесь.
                rest = [t for t in content if id(t) not in (subj_taken, id(pred))]
                for t in rest:
                    v = self._pick(t, ("NOUN", "ADJF"), case="ins") or \
                        self._pick(t, ("NOUN", "ADJF"), case="nom")
                    if v is not None:
                        rel = "IS_ROLE" if v["pos"] == "NOUN" else "HAS_PROPERTY"
                        res.facts.append(self._fact(subj or pred, t, rel,
                                                    res.sentence, neg, variant=v))
                        break

        # Владение («у кошки есть хвост») разбирается ДО всего: оно меняет и
        # субъекта, и отношение.
        owner, thing = self._possession(toks)
        if owner is not None and thing is not None:
            ov = self._pick(owner, ("NOUN",), case="gen") or owner["variants"][0]
            tv = self._pick(thing, ("NOUN",), case="nom") or thing["variants"][0]
            res.facts.append(self._fact_from_variant(
                ov["lemma"], thing, "HAS_PART", res.sentence, neg, tv))
            used = {id(owner), id(thing), id(pred)}
            self._attach_adjectives(res, content, used, neg)
            res.facts = self._dedup(res.facts)
            return res

        # 2) зависимые при сказуемом: объект / инструмент / получатель / место
        used = {subj_taken, id(pred)}

        # ОБРАБОТКА МОДАЛЬНЫХ ГЛАГОЛОВ: если сказуемое — модальный глагол
        # (уметь, мочь, хотеть, пытаться...) и есть инфинитив,
        # заменяем CAN_DO(субъект, модальность) на:
        #   HAS_PROPERTY(субъект, модальность_инфинитив) + CAN_DO(субъект, инфинитив)
        # Это отражает: «уметь X» = свойство, а «X» = умение.
        if (pred is not None and subj is not None
                and self._clause_pred_role(pred) == "action"
                and pred["lemma"] in MODAL_VERBS):
            inf = self._find_infinitive_dep(content, pred, used)
            if inf is not None:
                # Удаляем CAN_DO(субъект, модальность), созданный выше
                res.facts = [f for f in res.facts
                             if not (f.head == subj["lemma"]
                                     and f.relation == "CAN_DO"
                                     and f.tail == pred["lemma"])]
                # Добавляем HAS_PROPERTY(субъект, модальность_инфинитив)
                mod_variant = {
                    "lemma": f"{pred['lemma']}_{inf['lemma']}",
                    "pos": "VERB",
                    "case": "",
                    "number": "",
                    "gender": "",
                    "score": 0.9,
                }
                res.facts.append(self._fact_from_variant(
                    subj["lemma"], pred, "HAS_PROPERTY",
                    res.sentence, neg, mod_variant))
                # Добавляем CAN_DO(субъект, инфинитив)
                res.facts.append(self._fact(subj, inf, "CAN_DO",
                                            res.sentence, neg, variant=inf))
                used.add(id(inf))

        governed = [t for t in content if id(t) not in used and t["role"] != "property"]
        # ИМЕННОЙ режим = сказуемое НЕ смысловой глагол (связка или
        # прилагательное): «хвост собаки мокрый» — «собаки» определитель при
        # «хвост» (PART_OF), объекта действия тут просто нет.
        nominal = not (pred is not None and self._clause_pred_role(pred) == "action")
        head = subj_lemma
        if pred is not None and not nominal:
            # при местоимённом актёре вершиной обязан быть ГЛАГОЛ: запись
            # «купить OBJECT_OF автомобиль» — это знание о валентности глагола
            # и оно верно; ложью было бы приписать покупку автомобилю.
            head = pred["lemma"]

        # ПЕРВЫЙ ПРОХОД: чем токен является. Порядок зондирования падежей в
        # _case_role даёт ЕДИНСТВЕННЫЙ вариант, поэтому «кем является слово»
        # решается до того, как мы решаем, к кому оно относится. Иначе
        # «кошка боится грозы» давало PART_OF: вариант вин.п. у «грозы»
        # существует, и наивная проверка «е ли винительные» срабатывала на нём,
        # хотя фактически «грозы» здесь родительный (боится ЧЕГО).
        resolved = []
        for t in governed:
            rel, variant = self._case_role(t, nominal=nominal)
            if rel is not None and variant is not None:
                resolved.append((t, rel, variant))

        # РОДИТЕЛЬНЫЙ без предлога двусмысленен: «боится грозы» — объект
        # глагола, «крышка гроба» — определитель существительного. Различает их
        # то, забрал ли глагол уже прямой объект.
        verbs_present = any(t["role"] == "action" for t in content)
        acc_assigned = any(v.get("case") == "acc" and not t.get("prep")
                           for t, _r, v in resolved)
        noun_chain = [t for t in content if t["role"] == "subject"]

        # Вершина для зависимого — БЛИЖАЙШИЙ ГЛАГОЛ (перед или после слова),
        # а не всегда главный. В «Вася пришёл увидеть Петю» Петя — объект
        # «увидеть», а не «пришёл»; при одном главном глаголе объекты
        # многословных сказуемых прилипали бы не к тому действию.
        verbs_idx = [i for i, t in enumerate(content) if t["role"] == "action"]

        def verb_head(pos: int) -> str:
            if not verbs_idx:
                return subj_lemma
            best = min(verbs_idx, key=lambda i: (abs(i - pos), 0 if i < pos else 1))
            if content[best]["lemma"] in COPULAS:
                return subj_lemma
            return content[best]["lemma"]

        if money_sentence or value_sentence:
            # ЦЕНА / МЕРА. До п.17 здесь было «ничего не записать и сказать
            # почему»: «пять тысяч рублей» свелось бы к узлам «тысяча» и
            # «рубль», связанным со «стоить». Теперь число РАЗБИРАЕТСЯ
            # (_parse_amount), и знание сохраняется как VALUE, а не теряется.
            # Субъект обязателен: без него VALUE вешать не на что, и
            # записать меру в граф было бы некуда — тогда честный отказ.
            amount = self._amount_in(content)
            if amount is not None and (subj is not None or head):
                anchor = subj_lemma or head or ""
                if anchor:
                    res.facts.append(Fact(
                        head=anchor, relation="VALUE", tail=amount["phrase"],
                        evidence=res.sentence, negated=neg,
                        confidence=0.7))
                    res.notes.append("цена/мера записана как VALUE: %s"
                                     % amount["phrase"])
                else:
                    res.notes.append("цена/мера есть, но субъект не найден — "
                                     "VALUE не записан")
            else:
                res.notes.append("ценовое/мерное предложение: число не разобрано — "
                                 "VALUE не записан")
            res.facts = self._dedup(res.facts)
            return res

        for t, rel, variant in resolved:
            # «кошка это животное»: именная часть стоит в ИМ./Вин.п., но НИКАКОГО
            # действия в предложении нет — называть её OBJECT_OF значит
            # изобрести действие, которого не было.
            if (copula_marker and pred is None and not verbs_present
                    and variant.get("case") in ("nom", "acc")
                    and rel in ("OBJECT_OF", "IS_ROLE")):
                rel = "IS_ROLE"
            if (variant.get("case") == "gen" and not t.get("prep")
                    and (nominal or acc_assigned)):
                owner = next((n for n in reversed(noun_chain)
                              if content.index(n) < content.index(t)), None)
                if owner is not None:
                    rel, t = "PART_OF", owner
            res.facts.append(self._fact_from_variant(
                verb_head(content.index(t)) or head or subj_lemma,
                t, rel, res.sentence, neg, variant))

        # Инфинитив после личного глагола = ЦЕЛЬ, а не второе умение:
        # «кошка пришла поспать» — одна связь PURPOSE, а не «пришла» + «умеет
        # спать». Иначе умение приписывается акту, который к нему отношения не
        # имеет, а различить их в графе потом нельзя.
        if (pred is not None and self._clause_pred_role(pred) == "action"
                and any(v.get("person") for v in pred["variants"])):
            for t in content:
                if id(t) in used or t["role"] != "action":
                    continue
                inf = self._pick(t, ("INFN",))
                if inf is not None:
                    res.facts.append(self._fact_from_variant(
                        pred["lemma"], t, "PURPOSE", res.sentence, neg, inf))
                    used.add(id(t))

        # 3) прилагательные без явного сказуемого-прилагательного
        self._attach_adjectives(res, content, used, neg)

        # «Вася — это инженер»: инженер тоже в ИМ.п. и согласуется в числе, но
        # это именная часть, а не второй кандидат в подлежащие. Считать его
        # конфликтом — значит кормить граф ложными предупреждениями.
        if not self._is_copula_form(pred) and not copula_marker:
            # Считаем спорными только ИМ.п., которые НЕ заняты другой ролью:
            # в «Витя дал кошке корм» «корм» формально может быть и ИМ.п., но он
            # уже разобран винительным объектом. Предупреждать о конфликте там,
            # где конфликта нет, — обесценивать предупреждение.
            assigned = {id(t) for t, _r, v in resolved if v.get("case") != "nom"}
            if sum(1 for t in content
                   if t["role"] == "subject" and self._agrees_number(t, num)
                   and id(t) not in assigned and id(t) != subj_taken
                   and self._has_case(t, ("NOUN",), "nom")) > 0:
                res.notes.append("несколько существительных в ИМ.п. согласуются "
                                 "со сказуемым — субъект спорный")
        if self._has_subordinate(toks):
            res.notes.append("придаточное: связи могут быть разорваны")

        # СТАТИКА vs ДИНАМИКА: пометить и вынести СОБЫТИЯ рассказа.
        # Делается ПОСЛЕ сборки фактов и ДО dedup — метка predicate входит в
        # вывод, но dedup ключуется по (head,relation,tail) и её сохранит.
        self._collect_events(res, subj, pred, order, neg)
        # ОЖИДАНИЕ из конструкции «X попросил Y <инфинитив>» — тоже динамика.
        self._collect_expectations(res, content, subj, pred, order, neg)

        # Одно и то же отношение может родиться из разных ветвей (например,
        # «кошка пушистая» — и из сказуемого, и из привязки прилагательного).
        # Оставляем самое уверенное: дубли в графе выглядели бы как подтверждение
        # факта, которого не было.
        res.facts = self._dedup(res.facts)
        return res

    # ── СТАТИКА / ДИНАМИКА ─────────────────────────────────

    @staticmethod
    def _predicate_tense(pred: Optional[dict]) -> str:
        """
        Время личного глагола-сказуемого: past / pres / fut / "".

        pymorphy3 даёт граммему tense. Нам нужно РАЗЛИЧИТЬ прошедшее от
        настоящего: «кошка СЪЕЛА канарейку» — происшествие, «кошка ЕСТ птиц» —
        общее знание о виде. Без этого различия рассказ целиком осядет в
        онтологию как вечные истины.
        """
        if pred is None:
            return ""
        for v in pred["variants"]:
            if v["pos"] in ("VERB", "PRTF", "GRND") and v.get("tense"):
                return v["tense"]
        return ""

    def _collect_events(self, res: SentenceFacts, subj: Optional[dict],
                        pred: Optional[dict], order: int, neg: bool) -> None:
        """
        Если предложение описывает ПРОИСШЕСТВИЕ — создать EventRecord и
        пометить его факты как события.

        ПРИЗНАК СОБЫТИЯ (грамматический, не смысловой):
          * сказуемое — личный глагол (есть лицо), НЕ связка, НЕ модальный;
          * время — ПРОШЕДШЕЕ («съела», «полил», «остановился»).
        Инфинитив и связка («является») событий не задают: это описание, а не
        случай. Модальный глагол («умеет») — свойство, а не происшествие.
        """
        if pred is None or self._is_copula_form(pred):
            return
        if self._clause_pred_role(pred) != "action":
            return
        if pred["lemma"] in MODAL_VERBS:
            return
        # ЗАМЕР: у глаголов ПРОШЕДШЕГО времени pymorphy3 НЕ ВЫДАЁТ лицо
        # (см. /tmp/t_m.py: «разозлилась» -> person='', tense='past'), а выдаёт
        # род. Поэтому фильтр «есть лицо» отсекал РОВНО те глаголы, которые
        # нам нужны, и событий не находилось ни одного. Признак финитности —
        # наличие ВРЕМЕНИ (инфинитив его не имеет), а не лица.
        tense = self._predicate_tense(pred)
        if not tense:
            # инфинитив / причастие без времени — не происшествие
            return
        if tense != "past":
            # Событие — то, что УЖЕ произошло. Настоящее и будущее описывают
            # закон или план, а не рассказанное происшествие.
            return

        subject = subj["lemma"] if subj is not None else "?"
        action = pred["lemma"]
        obj = ""
        # Объект события — ПЕРВЫЙ зависимый от действия, кроме CAN_DO (это
        # связь «кто способен», а не участник): «нарушила доверие» -> OBJECT_OF,
        # «напала на канарейку» -> LOCATED_IN (предлог «на»), «перешёл в
        # агрессию» -> LOCATED_IN. Ждать только OBJECT_OF — значит терять цель
        # события из-за предлога.
        for f in res.facts:
            if f.head == action and f.relation != "CAN_DO":
                obj = f.tail
                break
        res.events.append(EventRecord(
            subject=subject, action=action, obj=obj, tense=tense,
            order=order, evidence=res.sentence,
            negated=neg))
        # Факты этого предложения — ДИНАМИКА, а не онтология.
        for f in res.facts:
            f.predicate = "event"

    def _collect_expectations(self, res: SentenceFacts, content: List[dict],
                              subj: Optional[dict], pred: Optional[dict],
                              order: int, neg: bool) -> None:
        """
        Вывести ОЖИДАНИЕ из конструкции просьбы/запрета.

        «Кошка попросила канарейку не рассказывать тайну»:
          holder = подлежащее просьбы (кошка),
          target = винительный объект просьбы (канарейка),
          action = зависимый инфинитив (рассказывать),
          negated = частица «не» в предложении.

        ЗАМЕР: все четыре части уже даёт разбор (см. EXPECT_TASK.md). Не
        хватало только склейки. Без зависимого инфинитива ожидание НЕ
        выводится: «кошка попросила канарейку» без действия — неизвестно,
        чего именно ждёт кошка, и выдумывать это запрещено.
        """
        if pred is None or pred["lemma"] not in REQUEST_VERBS:
            return
        if subj is None:
            res.notes.append("просьба без подлежащего: ожидание не выведено")
            return
        # Адресат — объект просьбы. Падеж ЗАВИСИТ от глагола: «попросил
        # канарейку» — ВИН.п., «велел сыну» — ДАТ.п. Требовать только ВИН.п.
        # значит терять просьбы к людям («велел сыну вернуться»), а это самый
        # частый вид просьбы. ЗАМЕР: на «Мать велела сыну вернуться домой»
        # фильтр только-ВИН давал «просьба без адресата».
        target_tok = None
        for t in content:
            if t is subj or t is pred:
                continue
            if t["role"] == "action":
                continue
            if (self._has_case(t, ("NOUN",), "acc")
                    or self._has_case(t, ("NOUN",), "dat")):
                target_tok = t
                break
        if target_tok is None:
            res.notes.append("просьба без адресата (ВИН./ДАТ.п.): ожидание не выведено")
            return
        # Действие ожидания — зависимый ИНФИНИТИВ (не подлежащее, не сказуемое).
        inf_tok = None
        for t in content:
            if t is pred or t is subj or t is target_tok:
                continue
            if t["role"] == "action" and self._pick(t, ("INFN",)):
                inf_tok = t
                break
        if inf_tok is None:
            res.notes.append("просьба без инфинитива: ожидание не выведено")
            return
        # Объект ИНФИНИТИВА — то, чего ожидание касается («тайну»). Нужен,
        # чтобы нарушение находилось БЕЗ сопоставления видов глагола: в
        # «не рассказывать ТАЙНУ» и «рассказала ТАЙНУ» объект — одна лемма.
        # Факт уже собран: рассказывать --OBJECT_OF--> тайна.
        inf_obj = ""
        for f in res.facts:
            if f.head == inf_tok["lemma"] and f.relation != "CAN_DO":
                inf_obj = f.tail
                break
        res.expectations.append(ExpectationRecord(
            holder=subj["lemma"], target=target_tok["lemma"],
            action=inf_tok["lemma"], obj=inf_obj,
            negated=bool(neg) or pred["lemma"] in FORBID_VERBS,
            order=order, evidence=res.sentence,
            source=f"просьба ({pred['lemma']})"))

    @staticmethod
    def _dedup(facts: List[Fact]) -> List[Fact]:
        best: Dict[Tuple[str, str, str], Fact] = {}
        for f in facts:
            cur = best.get(f.key())
            if cur is None or f.confidence > cur.confidence:
                best[f.key()] = f
        return sorted(best.values(), key=lambda f: -f.confidence)

    # ── предлоги ───────────────────────────────────────────

    @staticmethod
    def _tag_prepositions(toks: List[dict]) -> List[dict]:
        """
        Предлог узлом не станет, но именно он отличает «кошка НА подоконнике»
        (место) от «кошка ПОДОКОННИК» (тождество). Пишем его в токен — держать
        таблицу по id() нельзя, id переиспользуются.

        Перенос через прилагательное разрешён («на широком подоконнике»), но
        ОБЯЗАТЕЛЬНО сгорает на существительном и не перелетает через глагол:
        иначе «говорил о погоде, и пошел дождь» сделал бы дождь местоположением.
        """
        pending, budget = "", 0
        for t in toks:
            w = t["word"]
            if w in ALL_PREPS:
                pending, budget, t["prep"] = w, 4, ""
                continue
            if pending and budget > 0:
                t["prep"] = pending
                budget -= 1
                # Расходуется ТОЛЬКО на существительное. Вариант с «любой
                # значимой лексеме» сжигал предлог на числительном и терял его
                # для следующего слова: в «кипит при ста градусах» «при»
                # досталось «ста», а «градусах» стало безпредложным и
                # превратилось в LOCATED_IN вместо CONDITION.
                if t["pos"] == "NOUN":
                    pending, budget = "", 0
            else:
                t["prep"] = ""
                if t["role"] == "action":
                    pending, budget = "", 0
        return toks

    # ── выбор вариантов по синтаксису ──────────────────────

    def _pick(self, tok: dict, pos: Tuple[str, ...], case: str = "",
              number: str = "") -> Optional[dict]:
        """
        Разбор, подходящий под роль. Среди ПОДХОДЯЩИХ с равной вероятностью
        предпочитаем лемму, которую ядро УЖЕ ЗНАЕТ (self.known), а не первый
        из списка.

        Почему это важно: «стоит» даёт 0.33 стоить / 0.33 стоять — частотный
        словарь не выбирает. Но если в графе есть «стоять» и нет «стоить»,
        выбор однозначен. Это не подгонка: знание о том, что система уже знает,
        — легитимный и очень сильный сигнал снятия омонимии.
        """
        cands = [v for v in tok["variants"]
                 if (not pos or v["pos"] in pos)
                 and (not case or v["case"] == case)
                 and (not number or not v["number"] or v["number"] == number)]
        if not cands:
            return None
        if self.known:
            best = max(cands, key=lambda v: (v["lemma"] in self.known, v["score"]))
        else:
            best = max(cands, key=lambda v: v["score"])
        # вернули НЕ самый вероятный вариант — омонимия разрешена по известному,
        # честнее это пометить, чем выдать за 100% разбор
        best = dict(best)
        best["tie_broken_by_known"] = (len({v["lemma"] for v in cands}) > 1)
        return best

    @staticmethod
    def _n_variants(tok: dict, pos: Tuple[str, ...], case: str = "") -> int:
        return sum(1 for v in tok["variants"]
                   if (not pos or v["pos"] in pos) and (not case or v["case"] == case))

    def _has_case(self, tok: dict, pos: Tuple[str, ...], case: str) -> bool:
        return self._pick(tok, pos, case=case) is not None

    @staticmethod
    def _is_copula_form(tok: Optional[dict]) -> bool:
        return bool(tok) and tok["lemma"] in COPULAS

    def _clause_pred_role(self, tok: dict) -> str:
        """
        action | property | other — чем является сказуемое для связи с подлежащим.

        Прилагательное-СКАЗУЕМОЕ («хвост мокрый») — это property; относить его
        к прочим нельзя, иначе теряется единственный факт бесглагольного
        предложения. Глагол с предлогом («кошка залезла на шкаф») — other:
        направление в этой версии не выражается, а повесить CAN_DO «залезть»
        без цели — это записать «кошка умеет залезать» и потерять, КУДА.
        """
        if tok["role"] == "property":
            return "property"
        if tok.get("prep") or self._is_copula_form(tok):
            return "other"
        return "action"

    def _find_infinitive_dep(self, content: List[dict], pred: dict,
                               used: set) -> Optional[dict]:
        """
        Находит инфинитив, зависящий от модального глагола-сказуемого.
        Возвращает токен инфинитива или None.
        """
        for t in content:
            if id(t) in used or t is pred or t["role"] != "action":
                continue
            inf = self._pick(t, ("INFN",))
            if inf is not None:
                return t
        return None

    # ── актёр и владение ───────────────────────────────────

    @staticmethod
    def _pronoun_actor(toks: List[dict], pred: Optional[dict]) -> Optional[dict]:
        """
        Личное местоимение в ИМ.п. перед сказуемым: «ОН купил», «ОНА гладила».
        Существительного-актёра в таком предложении нет, и выдумывать его из
        винительного — значит переворачивать смысл.
        """
        if pred is None:
            return None
        ip = toks.index(pred) if pred in toks else -1
        for t in toks[:max(ip, 0)]:
            if t["word"] in ("он", "она", "оно", "они", "я", "ты", "мы", "вы",
                             "кто", "кто-то", "кто-нибудь"):
                return t
        return None

    @staticmethod
    def _is_price(content: List[dict]) -> bool:
        """Глагол «стоить» рядом с деньгами/числами = цена, а не действие."""
        verbs = {t["lemma"] for t in content if t["role"] == "action"}
        if not (verbs & {"стоить", "обойтись", "потянуть"}):
            return False
        return (any(t["lemma"] in MONEY_LEMMAS for t in content)
                or any(t["pos"] == "NUMR" for t in content)
                or any(any(v["pos"] == "NUMR" for v in t["variants"]) for t in content))

    @staticmethod
    def _is_value_sentence(content: List[dict]) -> bool:
        """
        Предложение-ЗНАЧЕНИЕ: глагол меры (весить/стоить/равняться...) плюс
        число или единица измерения. «Кошка весит три килограмма»,
        «шкаф стоит пять тысяч рублей», «путь равняется ста метрам».

        Это НЕ то же, что _is_price (только деньги): вес и длина тоже
        значения, и без своего флага они бы падали в ложную валентность
        CAN_DO (ЗАМЕРЕНО до п.17).
        """
        verbs = {t["lemma"] for t in content if t["role"] == "action"}
        if not (verbs & VALUE_VERBS):
            return False
        has_num = any(t["pos"] == "NUMR"
                      or any(v["pos"] == "NUMR" for v in t["variants"])
                      for t in content)
        has_unit = any(t["lemma"] in MEASURE_LEMMAS or t["lemma"] in MONEY_LEMMAS
                       for t in content)
        return has_num or has_unit

    @staticmethod
    def _num_word(lemma: str) -> Optional[int]:
        """Слово-числительное -> число. 'пять'->5, 'сорок'->40, 'сто'->100."""
        lw = (lemma or "").lower()
        for table in (NUM_ONES, NUM_TENS, NUM_HUNDREDS):
            if lw in table:
                return table[lw]
        return None

    @staticmethod
    def _num_token_value(tok: dict) -> Optional[int]:
        """
        Число из токена: цифрами ('5000') или словом ('пять').

        ЦИФРОВАЯ запись читается из САМОГО СЛОВА, а не из разбора: pymorphy3
        даёт для «5000» лемму «5000», и терять её нельзя. Словесная —
        через лемму варианта NUMR (лемма уже нормализует 'пять/пяти' -> 'пять').
        """
        w = (tok.get("word") or "").strip()
        digits = w.replace(" ", "").replace("\u00a0", "")
        if digits.isdigit():
            return int(digits)
        for v in tok.get("variants", []):
            if v.get("pos") == "NUMR":
                val = SimpleExtractor._num_word(v.get("lemma") or "")
                if val is not None:
                    return val
        return SimpleExtractor._num_word(w)

    def _parse_amount(self, content: List[dict], start: int) -> Optional[dict]:
        """
        Разобрать число с мерой начиная с позиции start.

        Возвращает {'value': int, 'measure': str|None, 'phrase': str,
                   'end': int} либо None, если числа/меры здесь нет.

        ПОЧЕМУ ручной разбор, а не сумма всех NUMR подряд: «два миллиона
        триста тысяч» — это 2_300_000, и множители работают как разряды.
        Сложить 2+300 = 302 (а тем более 2+1000000) — значит записать в граф
        цену, которой никто не называл.
        """
        i = start
        total = 0
        have = False
        # Разбор в два прохода: единицы/десятки/сотни накапливаются, а
        # множитель («тысяча»/«миллион») умножает накопленное.
        pending = 0
        while i < len(content):
            t = content[i]
            lw = (t.get("lemma") or t.get("word") or "").lower()
            mult = NUM_MULTIPLIERS.get(lw)
            if mult is not None and (have or pending):
                base = pending if pending else 1
                total += base * mult
                pending = 0
                have = True
                i += 1
                continue
            if t.get("pos") == "NUMR" or any(v.get("pos") == "NUMR"
                                              for v in t.get("variants", [])):
                val = self._num_token_value(t)
                if val is None:
                    break
                pending += val
                have = True
                i += 1
                continue
            break
        if not have:
            return None
        total += pending
        # Мера — существительное сразу после числа ('рублей', 'килограмма').
        # Предлог между числом и мерой допустим ('на пять тысяч рублей'),
        # но не глагол: «пять кошек спят» — это НЕ мера «кошек»
        # при числе, а подлежащее, и VALUE тут был бы выдумкой.
        measure = None
        measure_word = ""
        end = i
        j = i
        while j < len(content) and j < i + 2:
            t = content[j]
            if t.get("pos") == "NOUN" or any(v.get("pos") == "NOUN"
                                               for v in t.get("variants", [])):
                measure = (t.get("lemma") or t.get("word") or "").lower()
                measure_word = (t.get("word") or measure).lower()
                end = j + 1
                break
            j += 1
        if measure is None:
            end = i
        # В ФРАЗЕ — поверхностная форма («5000 рублей»), а НЕ лемма
        # («5000 рубль»): это то, что человек произнёс, и именно это
        # читается в ответе «что известно про шкаф».
        phrase = str(total)
        if measure_word:
            phrase = "%d %s" % (total, measure_word)
        return {"value": total, "measure": measure, "phrase": phrase, "end": end}

    def _amount_in(self, content: List[dict]) -> Optional[dict]:
        """
        Найти число с мерой в ЛЮБОМ месте предложения.

        Почему не всегда с начала: «Шкаф стоит пять тысяч рублей» и «Пять
        тысяч рублей стоит шкаф» — одно и то же знание; искать только в позиции
        0 значит терять второй порядок слов. Перебираем все NUMR-позиции и
        берём ПЕРВЫЙ удачный разбор (первое число — главное в мере).
        """
        for idx, t in enumerate(content):
            is_num = (t.get("pos") == "NUMR"
                      or any(v.get("pos") == "NUMR"
                             for v in t.get("variants", [])))
            if not is_num:
                continue
            got = self._parse_amount(content, idx)
            if got is not None:
                return got
        return None

    @staticmethod
    def _possession(toks: List[dict]) -> Tuple[Optional[dict], Optional[dict]]:
        """
        «У кошки ЕСТЬ хвост» / «у дерева много веток»: конструкция владения.
        Возвращает (владелец, владение). Падежная схема устойчивая: «у» + род.п.
        плюс бытийное слово; это надо ловить отдельно, иначе «у» даёт
        LOCATED_IN и связь теряет смысл владения.
        """
        has_be = any(t["word"] in ("есть", "имеется", "имеются", "было", "были")
                     or t["lemma"] in ("быть", "иметься", "иметь", "существовать")
                     for t in toks)
        if not has_be:
            return None, None
        owner = None
        for t in toks:
            if t.get("prep") == "u" or (t.get("prep") == "у" and t["role"] == "subject"):
                owner = t
                break
        if owner is None:
            for i, t in enumerate(toks):
                if t["word"] == "у" and i + 1 < len(toks) and toks[i + 1]["role"] == "subject":
                    owner = toks[i + 1]
                    break
        if owner is None:
            return None, None
        thing = next((t for t in toks
                      if t is not owner and t["role"] == "subject"
                      and not t.get("prep")), None)
        return owner, thing

    # ── сказуемое ──────────────────────────────────────────

    def _predicate(self, content: List[dict],
                   subj: Optional[dict]) -> Optional[dict]:
        """
        Сказуемое — ГЛАВНЫЙ глагол. ИЗ ЦЕПОЧКИ ГЛАГОЛОВ берётся ПЕРВЫЙ ЛИЧНЫЙ:
        в «кошка пришла поспать» сказуемое «пришла», а инфинитив «поспать» —
        обстоятельство цели. Если взять последний смысловой глагол (как я сделал
        сначала), Петьа стал объектом «увидеть», а «пришёл» пропал: грамматически
        перевёрнутая связь, которую потом не отличить от настоящей.
        Личность глагола даёт pymorphy3 (граммема person); инфинитив её не имеет.
        """
        verbs = [t for t in content if t["role"] == "action"
                 and any(v["pos"] in VERB_POS for v in t["variants"])
                 and t["lemma"] not in COPULAS]
        if not verbs:
            # бесглагольное предложение: сказуемое — прилагательное
            props = [t for t in content if t["role"] == "property"]
            return props[0] if props else None
        if subj is not None:
            after = [v for v in verbs if content.index(v) > content.index(subj)]
            verbs = after or verbs
        for v in verbs:                      # первый личный = сказуемое
            if any(x.get("person") for x in v["variants"]):
                return v
        return verbs[0]

    # ── подлежащее ─────────────────────────────────────────

    def _subject(self, content: List[dict],
                 pred: Optional[dict]) -> Optional[dict]:
        """
        Подлежащее = существительное в ИМ.П., согласованное с сказуемым в ЧИСЛЕ.
        Согласование — не косметика: в «хвост собаки мокрый» именно число
        сказуемого («мокрый», ед.ч.) говорит, что подлежащее «хвост», а не
        «собаки». Без этого шага два омонима неразличимы.
        При нескольких подходящих берём ближайшее ПЕРЕД сказуемым (прямой
        порядок слов): «инженер проверил чертёж» — субъект инженер.
        """
        num = self._predicate_number(pred)
        # Существительное ПОСЛЕ ПРЕДЛОГА подлежащим быть не может в принципе:
        # без этой проверки «у кошки есть хвост» делало подлежащую «кошку»,
        # а связь переворачивалась.
        cands = [t for t in content
                 if t["role"] == "subject" and not t.get("prep")
                 and self._has_case(t, ("NOUN",), "nom")
                 and self._agrees_number(t, num)]
        if not cands:
            return None
        if pred is not None:
            before = [t for t in cands if content.index(t) < content.index(pred)]
            if before:
                return before[0]
        return cands[0]

    @staticmethod
    def _predicate_number(pred: Optional[dict]) -> str:
        """Число сказуемого, если оно выражено (инфинитив числа не имеет)."""
        if pred is None:
            return ""
        for v in pred["variants"]:
            if v["pos"] in ("VERB", "ADJF", "ADJS") and v["number"]:
                return v["number"]
        return ""

    def _agrees_number(self, tok: dict, num: str) -> bool:
        if not num:
            return True
        return self._pick(tok, ("NOUN",), case="nom", number=num) is not None

    # ── падеж зависимого -> отношение ──────────────────────

    def _case_role(self, tok: dict, nominal: bool = False
                   ) -> Tuple[Optional[str], Optional[dict]]:
        """
        Возвращает (отношение, выбранный ВАРИАНТ разбора) либо (None, None).
        Возврат варианта обязателен: из него берутся лемма и уверенность,
        иначе омонимия («кошки» = им.мн. или род.ед.) просочится в граф шумом.
        """
        prep = tok.get("prep") or ""
        if prep:
            cases, rel = PREP_CASES.get(prep, ((), "LOCATED_IN"))
            for c in cases:
                v = self._pick(tok, ("NOUN", "ADJF"), case=c)
                if v is not None:
                    return rel, v
            return None, None

        # Порядок важен: вин. закрывает цепочку, иначе омоним («хвост»:
        # вин. 0.57 / им. 0.43) утащит связь не туда.
        probe = [("acc", "OBJECT_OF"), ("gen", "PART_OF" if nominal else "OBJECT_OF"),
                 ("ins", "INSTRUMENT"), ("dat", "RECIPIENT"),
                 ("loc", "LOCATED_IN"), ("nom", "IS_ROLE" if nominal else None)]
        for case, rel in probe:
            if case == "nom" and rel is None:
                continue
            v = self._pick(tok, ("NOUN",), case=case)
            if v is None:
                continue
            if case == "ins":
                # Тв.п. отвечает на «чем?» ТОЛЬКО для орудия. «Солнце светит
                # днём», «вода кипит пятью градусами» — это время и мера:
                # записать их инструментом значит утверждать, что солнце
                # светит с помощью дня.
                if v["lemma"] in TIME_LEMMAS:
                    return "CIRCUMSTANCE", v
                if v["lemma"] in MEASURE_LEMMAS or v["lemma"] in MONEY_LEMMAS:
                    return "VALUE", v
            if case == "dat" and v.get("animacy") != "anim":
                # Дательный неодушевлённого = адресат/состояние («кошке холодно»),
                # одушевлённого = ПОЛУЧАТЕЛЬ («Витя дал кошке»). Называть
                # получателя «местоположением» — врать графу.
                rel = "LOCATED_IN"
            return rel, v
        return None, None

    # ── прилагательные ─────────────────────────────────────

    def _attach_adjectives(self, res: SentenceFacts, content: List[dict],
                           used_ids: set, neg: bool) -> None:
        """
        Свойство цепляется к существительному, с которым СОГЛАСОВАНО в числе и
        падеже; из согласованных — к ближайшему. В «большой дом и высокая
        башня» без согласования свойства разъедутся не туда.
        """
        nouns = [t for t in content if t["role"] == "subject"]
        if not nouns:
            return
        for adj in [a for a in content if a["role"] == "property"
                    and id(a) not in used_ids]:
            best, best_score = None, -1e9
            for n in nouns:
                score = -abs(content.index(n) - content.index(adj)) * 2
                av = self._pick(adj, ("ADJF", "ADJS", "COMP"))
                nv = None
                if av is not None:
                    # сущ. в том же падеже и числе — почти наверняка это пара
                    nv = self._pick(n, ("NOUN",), case=av["case"], number=av["number"])
                    if nv is not None:
                        score += 6
                    else:
                        # разное число — плохо; тот же падеж — сильнее близости,
                        # чем случайное соседство слов («на широком подоконнике»:
                        # предлог у прилагательного потерян, падеж всё-таки совпадает)
                        nv = self._pick(n, ("NOUN",), number=av["number"])
                        score += 2 if nv is not None else -5
                    if av.get("prep") and n.get("prep") == av["prep"]:
                        score += 3
                else:
                    score -= 5
                if score > best_score:
                    best, best_score = n, score
            if best is None:
                res.notes.append("прилагательное не согласуется ни с одним существительным")
                continue
            res.facts.append(self._fact(best, adj, "HAS_PROPERTY", res.sentence, neg))

    # ── конструкторы фактов ────────────────────────────────

    def _fact(self, head_tok: dict, tail_tok: dict, rel: str,
              evidence: str, neg: bool, variant: Optional[dict] = None) -> Fact:
        """Факт из двух токенов. variant задаётся, когда роль уже известна
        (именная часть связки), иначе её вычисляет _case_role."""
        if variant is None:
            _rel, variant = self._case_role(tail_tok)
            if variant is None:
                variant = tail_tok["variants"][0]
        return self._fact_from_variant(head_tok["lemma"], tail_tok, rel,
                                       evidence, neg, variant)

    def _fact_from_variant(self, head_lemma: str, tail_tok: dict, rel: str,
                           evidence: str, neg: bool, variant: dict) -> Fact:
        n_ok = self._n_variants(tail_tok, ("NOUN", "ADJF", "VERB"))
        ambiguous = n_ok > 1
        return Fact(
            head=head_lemma, relation=rel, tail=variant["lemma"],
            evidence=evidence, negated=neg,
            confidence=self._conf(variant, ambiguous),
            ambiguous=ambiguous,
        )

    @staticmethod
    def _conf(variant: dict, ambiguous: bool) -> float:
        """
        Уверенность = вероятность разбора с одним моральным ограничением:
        никогда не выдаём 1.0. Извлечение из текста — эвристика, и граф должен
        помнить разницу между «кошка» (вне сомнений) и «кошки» (омоним), иначе
        следующей версии нельзя будет сказать, на что можно опираться.
        """
        base = float(variant.get("score", 0.5))
        c = 0.45 + 0.4 * base
        if ambiguous:
            c *= 0.8
        return round(min(0.95, max(0.25, c)), 2)

    @staticmethod
    def _has_subordinate(toks: List[dict]) -> bool:
        return any(t["word"] in ("который", "которая", "которые", "которого",
                                 "когда", "пока", "после", "перед", "чтобы")
                   and t["content"] for t in toks)


if __name__ == "__main__":
    import sys
    text = ("Кошка сидит на широком подоконнике. Пушистая кошка боится грозы. "
            "Вася — это инженер. Хвост собаки мокрый. Кошка открыла мышь. "
            "Рыбу съела кошка. Кошка не боится воды. Кошки пушистые. "
            "Вася пришёл увидеть Петю. Витя дал кошке корм.")
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    ex = SimpleExtractor()
    for sf in ex.extract_text(text):
        print(f"\n«{sf.sentence}»")
        for f in sf.facts:
            print(f"   {f}   conf={f.confidence}")
        for n in sf.notes:
            print(f"   ! {n}")
