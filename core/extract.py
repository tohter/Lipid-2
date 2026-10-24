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

from core.morph import (Morph, split_sentences, analyze, split_clauses,
                       is_subordinate_marker, CONTENT_POS)

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
# Нулевая связка: «Кит не рыба» — сказуемого нет, именованная часть держится
# без него. Относится к связкам ТЕХНИЧЕСКИ (именное предложение, не действие),
# а не по смыслу: это способ сказать «сказуемого нет», не вводя в граф узел
# «быть». Символ в тексте встречается нулю раз (замер по корпусу experiments),
# поэтому не смешивается с настоящим словом.
ZERO_COPULA = "\u2205"

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

# Синонимия ВЫВОДИТСЯ ИЗ СИНТАКСИСА, а не угадывается по смыслу.
# ЗАМЕР (experiments/MERGE_TASK.md): косинус эмбеддингов даёт
#   бегемот/гиппопотам 0.218, тогда как стол/небо 0.134 — диапазоны
#   перекрываются, порогом не разъединить; морфовариант бегемот/бегемота
#   вообще 0.482. Значит близость слов НЕ признак тождества, и слияние по
#   косинусу запрещено: ложное объединение двух сущностей необратимо.
# Работает только порядок слов (замер на 6 предложениях):
#   «X, или Y, <сказуемое>»  — Y называет X, то же самое в ИМ.п. ДО глагола;
#   «выбрал стол, или стул»  — перечисление объектов, оба ПОСЛЕ глагола.
# Отсюда признак: союз «или» МЕЖДУ двумя существительными в ИМ.п., стоящими
# ДО сказуемого. NOTHING more — «Вася, или Петя, пришли» читается как два
# имени одного человека только потому, что так устроена русская запятая с
# «или»; это грамматика, а не догадка о словах.
APPOSITION_CONJ = "или"
# Слово, названное ВТОРЫМ, и есть вариант имени первого.

# Противопоставительные союзы: они ОБРЫВАЮТ область действия «не».
ADVERSATIVE_CONJ = ("а", "но")

# Союзы придаточного, которые морфология сама называет CONJ/ADVB (замер:
# «когда» -> CONJ|ADVB, «где» -> ADVB|CONJ, «чтобы» -> CONJ). Слова с падежными
# формами («который») тут не перечисляются — они ловятся граммемами, см.
# morph.RELATIVE_GRAMMEMES: список обязан был бы помнить все восемь падежей.
# «после»/«перед» ИСКЛЮЧЕНЫ намеренно: как предлоги они стоят в любом
# «после обеда», и объявлять там придаточное — обесценивать предупреждение
# (тот же приём, что отказ печатать ложные «спорные ИМ.п.»).
SUBORD_CONJ = ("когда", "пока", "чтобы", "где", "если", "хотя",
               "поскольку", "будто", "словно", "чем", "как")


class NegationScope:
    """
    Что ИМЕННО отрицает частица «не».

    ЗАМЕР ДО ПРАВКИ (experiments/NEGATION_TASK.md): «Кит — это не рыба,
    а млекопитающее» записывало ДВА факта, и ОБА с negated=True:

        кит --IS_ROLE /НЕ/--> рыба
        кит --IS_ROLE /НЕ/--> млекопитающее     <-- «кит не млекопитающее»: ЛОЖЬ

    Причина в одной строке: отрицание считалось по всему предложению
    (`any(t["word"] in NEGATIONS)`) и предъявлялось каждой связи одинаково.
    Одно «не» умело отрицать всё сказанное сразу.

    ДВЕ ОБЛАСТИ, выбор — фактом наличия противопоставительного союза:

      а) СОЮЗА НЕТ («кошка не ест рыбу») — область = всё предложение, как
         раньше. Это не консерватизм, а замер: «не» при сказуемом отрицает и
         действие, и его объект; разделения здесь нет.

      б) СОЮЗ ЕСТЬ («не A, а B»; «X, а не Y, ...») — противопоставление
         ПОТЕРМИНАЛЬНОЕ, каждая частица держит только свой член. Область
         каждой «не» = слова ОТ неё до границы (см. в). Почему с привязкой к
         самой частице, а не к «части между союзами»: ЗАМЕРЕНО на «Столица
         Франции — не Париж, а Лион», что отсечь только по союзу — оставить
         «не» властью над всем прологом, и PART_OF(столица, Франция) тоже
         стал бы отрицанием, которого в предложении нет.

      в) ГРАНИЦА области — не запятая (морфология пунктуации в токенах НЕ
         отдаёт, замер), а два грамматических признака:
         · частица СРАЗУ ПОСЛЕ «а»/«но» («кошка, а не собака, ловит мышь»):
           это ВСТАВНОЙ противопоставительный член, он кончается БЛИЖАЙШИМ
           личным глаголом — дальше идёт основная часть предложения. Без это-
           го правила CAN_DO(кошка, ловить) стало бы отрицанием, то есть
           «кошка не ловит мышь» — ровно наоборот сказанного.
         · иначе («не кошка поймала мышь, а собака») частица стоит в своей
           части и держит всё до ближайшего противопоставительного союза:
           событие «кошка поймала мышь» здесь ОТРИЦАЕТСЯ ЦЕЛИКОМ — иначе в
           граф попало бы «кошка поймала мышь», чего в тексте нет.

    ПЕРЕНОС отрицания на зависимых. В «кошка не ест рыбу, а собака ест»
    частица стоит при глаголе, а объект стоит за ним: OBJECT_OF(есть, рыба)
    обязана остаться отрицанной, иначе в граф попадёт «кошка ест рыбу».
    Поэтому связь отрицается и когда в области стоит ЕЁ ГЛАГОЛ (head), а не
    только хвост. Держать это в одном месте важно: два места проверки
    разносятся неизбежно.

    Это позиция частицы, союза и финитности глагола — не список слов и не
    догадка о смысле.
    """

    def __init__(self, toks: List[dict]):
        self.toks = toks
        # есть ли отрицание в предложении вообще (для отчётов и мер)
        self.any = any(t.get("word") in NEGATIONS for t in toks)
        adversative = [i for i, t in enumerate(toks)
                       if t.get("pos") == "CONJ"
                       and str(t.get("lemma") or "").lower() in ADVERSATIVE_CONJ]
        # узкая область имеет смысл ТОЛЬКО при противопоставлении
        self.narrow = bool(self.any and adversative)
        self._at = {id(t): i for i, t in enumerate(toks)}
        # индексы слов, попадающих под каждую частицу
        self.scopes: List[List[int]] = []
        if self.narrow:
            for i, t in enumerate(toks):
                if t.get("word") not in NEGATIONS:
                    continue
                prev_adv = [a for a in adversative if a < i]
                inserted = bool(prev_adv) and (max(prev_adv) == i - 1)
                stop = len(toks)
                for j in range(i + 1, len(toks)):
                    if j in adversative:
                        stop = j
                        break
                    # вставной член «а не X» закрыт: дальше основная часть
                    if inserted and self._is_finite(toks[j]):
                        stop = j
                        break
                self.scopes.append([j for j in range(i + 1, stop)
                                    if toks[j].get("content")])
        self._scope_idx = set().union(*[set(s) for s in self.scopes]) \
            if self.scopes else set()
        self._scope_lemmas = {str(toks[j].get("lemma") or "").lower()
                             for j in self._scope_idx}
        # что именно отрицается — для ГРОМКОЙ заметки в разборе
        self.denied: List[str] = [str(toks[j].get("lemma") or toks[j].get("word")).lower()
                                  for j in sorted(self._scope_idx)]
        # сколько содержательных слов СПАСЛОСЬ от отрицания благодаря границе.
        # Нужно, чтобы правка была видна числом, а не «должно стать лучше».
        total = sum(1 for t in toks if t.get("content"))
        self.rescued = (total - len(self._scope_idx)
                        if self.narrow and total > len(self._scope_idx) else 0)

    @staticmethod
    def _is_finite(tok: dict) -> bool:
        """Личный глагол (есть время ИЛИ лицо) — граница основной части."""
        if not tok.get("content") or tok.get("role") != "action":
            return False
        return any(v.get("tense") or v.get("person")
                   for v in tok.get("variants", [])
                   if v.get("pos") in ("VERB", "PRTF"))

    def _in_scope(self, tok: Optional[dict]) -> bool:
        if not self.narrow or tok is None:
            return False
        i = self._at.get(id(tok))
        return i is not None and i in self._scope_idx

    def negates(self, tok: Optional[dict]) -> bool:
        """Отрицается ли этот токен.

        Вне узкой области (союза нет) — да, если «не» в предложении есть:
        это ровно прежняя область «всё предложение», замеренная на
        «кошка не ест рыбу». Токена нет (синтетическая запись) — не выдумываем.
        """
        if not self.any:
            return False
        if not self.narrow:
            return True
        if tok is None:
            return False
        return self._in_scope(tok)

    def negates_pair(self, head_lemma: str, tail_tok: Optional[dict]) -> bool:
        """Отрицается ли связь head -> tail.

        Отрицание глагола отрицает и его зависимых: иначе из «кошка не ест
        рыбу, а собака ест» в граф попало бы «кошка ест рыбу».
        """
        if self.negates(tail_tok):
            return True
        if not self.any or not self.narrow:
            return bool(self.any)
        return str(head_lemma or "").lower() in self._scope_lemmas

    def note(self) -> str:
        """Заметка об уточнённой области — только если она что-то изменила."""
        if not self.rescued:
            return ""
        return ("отрицание «не» относится только к «%s»; ещё %d значимых слов "
                "в предложении УТВЕРЖДАЮТСЯ (противопоставление «а»/«но»)"
                % ("», «".join(self.denied) or "?", self.rescued))


def apposition_aliases(tokens: List[dict], sentence: str = "") -> List[tuple]:
    """
    Найти пары «X, или Y» — одно и то же, названное двумя именами.
    Возвращает [(канон, вариант)].

    Проверки порядком слов, а не значениями:
      1) оба токена — существительные в ИМ.п.;
      2) между ними союз «или»;
      3) глагола-сказуемого ещё НЕ было: после глагола тот же союз даёт
         перечисление объектов («выбрал стол, или стул» — не тождество).

    СОЮЗ БЕРЁТСЯ ИЗ ЛЕММЫ, а не из surface-формы: морфология приводит слово
    к «или» в любом регистре, а сравнение по «word» молча потеряло бы союз,
    записанный с другой формы.

    Связка «— это» здесь СОЗНАТЕЛЬНО не разбирается: «Гиппопотам — это
    бегемот» и «Кит — это млекопитающее» синтаксически неразличимы (замер на
    реальных разборах), а «бегемот» — не роль гиппопотама. Придумывать там
    отношение нельзя, поэтому остаётся то, что разбор действительно видит:
    IS_ROLE.
    """
    out: List[tuple] = []
    pred_idx = next((i for i, t in enumerate(tokens)
                     if t.get("content") and _is_verb_pos(t.get("pos"))),
                    None)
    pred_at = pred_idx if pred_idx is not None else len(tokens)
    # ЗАМЕР, ИЗМЕНИВШИЙ ПРАВИЛО: на уровне разбора «Кот, или собака, спят»
    # неотличимо от «Бегемот, или гиппопотам, живёт» — те же ИМ.п., тот же
    # союз, то же положение до глагола. Различают их ДВА грамматических
    # признака, ни одного списка слов:
    #   а) настоящая аппозиция ОБОСОБЛЕНА: «, или » в тексте; выбор «либо»
    #      запятой не выделяется («кот или собака спят»);
    #   б) у настоящей аппозиции сказуемое в ЕД. числе (оно про один предмет),
    #      у перечисления - во МН. числе («кот или собака спят»).
    # Принять «кот = собака» было бы необратимой ошибкой, поэтому требуются
    # ОБА признака сразу: одной запятой мало (сравните «взял стол, или тумбу»
    # с винительным), одного единственного числа тоже мало.
    set_off = (", " + APPOSITION_CONJ + " ") in (sentence or "").lower()
    plural_verb = bool(pred_idx is not None and
                       str(tokens[pred_idx].get("number") or "").lower() == "pl")
    if not set_off or plural_verb:
        return out

    def nom_noun(tok):
        if not tok or not tok.get("content"):
            return None
        if tok.get("pos") != "NOUN" or tok.get("case") != "nom":
            return None
        # берём лемму, а не surface-форму: леммой ключуется сущность
        lem = str(tok.get("lemma") or "").strip().lower()
        return lem or None

    for i in range(1, min(len(tokens) - 1, pred_at)):
        if str(tokens[i].get("lemma") or "").strip().lower() != APPOSITION_CONJ:
            continue
        first = nom_noun(tokens[i - 1])
        second = nom_noun(tokens[i + 1])
        if not first or not second or first == second:
            continue
        out.append((first, second))
    return out


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
    # Пары «канон, вариант» из аппозиции («бегемот, или гиппопотам»). Едут
    # ОТДЕЛЬНЫМ списком, а не только внутри facts: обучение обязано успеть
    # ЗАПОМНИТЬ вариант имени ДО того, как начнёт разрешать слова текста, иначе
    # второе имя успеет родить вторую сущность.
    aliases: List[tuple] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    tokens: List[dict] = field(default_factory=list)
    # Сказуемое ГЛАВНОЙ части (лемма) — нужно придаточному цели: «спрятала,
    # чтобы ловить» связывает «ловить» с «спрятала», а не с подлежащим.
    # ЗАМЕРЕНО до правки: часть «чтобы не ловить её» не давала НИ ОДНОЙ связи,
    # потому что личный глагол в ней один — главный, а он в другой части.
    main_action: str = ""

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
        """
        Разбор предложения. Сложноподчинённое разбирается ПО ЧАСТЯМ.

        ЗАЧЕМ (замер, п.18). Плоский разбор ищет ОДИН субъект и ОДНО сказуемое
        на весь текст, и второе сказуемое остаётся без пары:
          «Мяч, который бросил мальчик, укатился»  -> «укатился» потеряно,
                                                      записано «мяч CAN_DO
                                                      бросить» (инверсия)
          «Дом, где жил пёс, сгорел»              -> «дом CAN_DO жила»
        Придаточное вырезается из главного предложения (СКЕЛЕТ), поэтому у
        главного остаётся своё сказуемое, а придаточное разбирается отдельно.
        Союзное слово при этом не просто выбрасывается: его падеж говорит, КТО
        что делал, — см. _relative_part.
        """
        sentence = (sentence or "").strip()
        if not sentence:
            return self._extract_flat(sentence, order)
        parts = split_clauses(sentence)
        if len(parts) <= 1 or not any(p["sub"] for p in parts):
            return self._extract_flat(sentence, order)

        subs = [p for p in parts if p["sub"]]
        # СКЕЛЕТ главного предложения: части придаточного ВЫРЕЗАНЫ целиком, а не
        # заменены пустотой — иначе «Дом, где жил пёс, сгорел» несло бы «жила»
        # и в главном.
        skeleton = " ".join(p["text"] for p in parts if not p["sub"]).strip()
        res = self._extract_flat(skeleton, order)
        # evidence обязано остаться ИСХОДНЫМ предложением: по evidence ищут,
        # откуда взялась связь, а скелет — служебная строка, которой в тексте нет.
        for f in res.facts:
            f.evidence = sentence
        for e in res.events:
            e.evidence = sentence

        for part in subs:
            if self._relative_part(part["text"], res, sentence, order):
                continue
            if self._purpose_part(part["text"], res, sentence):
                continue
            # Придаточное без союзного слова («когда играл дождь»): порядок слов
            # там обычный, общий разбор с ним справляется. NegationScope берётся
            # СВОЙ, по этой части — иначе «не» из придаточного цели отрицает
            # главное (замер: «кошка спрятала мышь, чтобы НЕ есть её» записывало
            # «кошка НЕ прятала мышь»). Тот же приём, что в NEGATION_TASK.md,
            # только граница — СЕГМЕНТ, а не часть простого предложения.
            sub = self._extract_flat(part["text"], order)
            for f in sub.facts:
                f.evidence = sentence
                res.facts.append(f)
            for e in sub.events:
                e.evidence = sentence
                res.events.append(e)
            # Общую заметку о разрыве связей из ЧАСТИ не тащим: про неё уже
            # известно, что она придаточное, и она разобрана. Два предупреждения
            # об одном и том же обесценивают оба.
            res.notes.extend(n for n in sub.notes
                             if "связи могут быть разорваны" not in n)

        # Заметка обязана остаться: без строки в отчёте нельзя отличить
        # «придаточного не было» от «придаточное размечено, но не разобрано».
        res.notes.append("придаточное: разобрано по частям (%d ч.)" % len(parts))
        res.facts = self._dedup(res.facts)
        return res

    # ── ПРИДАТОЧНОЕ С СОЮЗНЫМ СЛОВОМ (п.18) ────────────────────

    def _relative_part(self, clause: str, res: SentenceFacts,
                       evidence: str, order: int) -> bool:
        """
        Придаточное с союзным словом: «который бросил мальчик»,
        «берега которой заросли».

        Возвращает True, если часть разобрана (пусть и с пометкой о неполноте),
        False — если её следует отдать общему разбору.

        ПОЧЕМУ НЕ ПОДСТАНОВКА АНТЕЦЕДЕНТА. Соблазн велик: заменить «который»
        словом, к которому оно относится, и позвать обычный разбор. ЗАМЕРЕНО,
        что обычный разбор на порядок слов опереться не может: «Мяч бросил
        мальчик» и без всякого придаточного даёт «мяч CAN_DO бросить», потому
        что словарь у «мяч» ставит вин.п. 0.67 против им.п. 0.33 (замер), а у
        «мальчик» им.п. единственный. Подстановка унаследовала бы ровно ту
        инверсию агент/объект, ради исправления которой всё затеяно. Поэтому
        роль здесь берётся из ПАДЕЖА союзного слова.

        Падеж союзного слова — это ответ на вопрос «кто кого»:
          им.п.  -> антецедент сам делает («пёс, который охранял дом»)
          вин.п. -> антецедент испытывает («мяч, который бросил мальчик»)
          род.п. -> принадлежность («река, берега которой заросли»)
        Двусмысленность им./вин. снимается НЕ частотой (замер: у «который»
        им.п. 0.83 против вин. 0.17 — частота всегда тянула бы им.п.), а
        тем, МОЖЕТ ЛИ вообще слово стоять в винительном: у «мальчик» вин.п. не
        существует, у «дом» существует.
        """
        toks = self._tag_prepositions(analyze(clause))
        rel = next((t for t in toks if t.get("relative")), None)
        if rel is None:
            return False

        others = [t for t in toks if t.get("content") and t is not rel
                  and not t.get("prep")
                  and t.get("role") in ("subject", "object")]

        antecedent = self._antecedent(evidence, rel, clause)
        if not antecedent:
            res.notes.append(
                "придаточное с «%s»: определяемое слово не найдено — связи "
                "придаточного НЕ записаны" % rel["word"])
            return True

        case = self._relative_case(rel, others)
        verbs = [t for t in toks if t.get("content")
                 and t.get("role") == "action"
                 and any(v["pos"] in VERB_POS for v in t["variants"])
                 and t["lemma"] not in COPULAS]
        predicate = next((v for v in verbs
                          if any(x.get("person") or x.get("tense")
                                 for x in v["variants"])), None)
        if predicate is None:
            res.notes.append("придаточное «%s» без личного глагола — связи не "
                             "записаны" % clause)
            return True
        # ХВОСТ ГЛАГОЛА ОБЯЗАТЕЛЬНО ЛЕММА. Замер до правки: «дом --CAN_DO-->
        # жила»: в граф легла ФОРМА, и знание об одном глаголе разъехалось бы
        # по двум узлам («жить» и «жила») навсегда.
        verb = self._pick(predicate, VERB_POS) or predicate["variants"][0]
        action = verb["lemma"]

        # Отрицание — по ОБЛАСТИ ЭТОЙ ЧАСТИ. «не» из придаточного цели не имеет
        # власти над главным (замер: «кошка спрятала мышь, чтобы НЕ есть её»
        # отрицало «спрятала»). Граница — сегмент, а не часть простого
        # предложения, как в NEGATION_TASK.md.
        negs = NegationScope(toks)
        made = False
        mark_from = len(res.facts)

        if case in ("nom", "voc"):
            # антецедент — агент
            res.facts.append(self._fact_from_variant(
                antecedent, predicate, "CAN_DO", evidence, negs, verb))
            made = True
            obj = next((t for t in others
                        if self._pick(t, ("NOUN", "ADJF"), case="acc")
                        or self._pick(t, ("NOUN", "ADJF"), case="gen")), None)
            if obj is not None:
                ov = (self._pick(obj, ("NOUN", "ADJF"), case="acc")
                      or self._pick(obj, ("NOUN", "ADJF"), case="gen"))
                res.facts.append(self._fact_from_variant(
                    action, obj, "OBJECT_OF", evidence, negs, ov))

        elif case == "acc":
            # антецедент — пациент. Агент обязан стоять в им.п.; приоритет у
            # слова, которое винительным быть НЕ может (оно однозначно).
            agent = next((t for t in others
                          if self._has_case(t, ("NOUN", "ADJF"), "nom")
                          and not self._can_be_accusative(t)), None)
            if agent is None:
                agent = next((t for t in others
                              if self._has_case(t, ("NOUN", "ADJF"), "nom")),
                             None)
            if agent is None:
                res.notes.append(
                    "придаточное «%s»: антецедент в вин.п., агент в им.п. не "
                    "найден — действие записано БЕЗ субъекта" % clause)
            else:
                av = (self._pick(agent, ("NOUN", "ADJF"), case="nom")
                      or agent["variants"][0])
                res.facts.append(Fact(
                    head=av["lemma"], relation="CAN_DO", tail=action,
                    evidence=evidence,
                    negated=negs.negates_pair(action, predicate),
                    confidence=self._conf(verb, True), ambiguous=True))
                made = True
            res.facts.append(Fact(
                head=action, relation="OBJECT_OF", tail=antecedent,
                evidence=evidence,
                negated=negs.negates_pair(action, predicate),
                confidence=self._conf(verb, True), ambiguous=True))
            made = True
            self._relative_event(res, negs, predicate,
                                 agent["lemma"] if agent else "?",
                                 action, antecedent, order, evidence)

        else:
            # род./дат./твор./предл. — придаточное о принадлежности:
            # «река, берега которой заросли» = берега ПРИНАДЛЕЖАТ реке.
            owner = next((t for t in others
                          if self._pick(t, ("NOUN",), case="gen") is not None
                          or self._pick(t, ("NOUN",), case="acc") is not None),
                         None)
            if owner is not None:
                ov = (self._pick(owner, ("NOUN",), case="gen")
                      or self._pick(owner, ("NOUN",), case="acc")
                      or owner["variants"][0])
                res.facts.append(Fact(
                    head=ov["lemma"], relation="PART_OF", tail=antecedent,
                    evidence=evidence,
                    negated=negs.negates_pair(ov["lemma"], owner),
                    confidence=self._conf(ov, True), ambiguous=True))
                res.facts.append(self._fact_from_variant(
                    ov["lemma"], predicate, "CAN_DO", evidence, negs, verb))
                made = True

        if made and case in ("nom", "voc"):
            self._relative_event(res, negs, predicate, antecedent, action,
                                 self._relative_object(res, action, evidence),
                                 order, evidence)
        # ДИНАМИКА, а не онтология. Плоский разбор помечает факты события
        # целиком (_collect_events), и факты придаточного обязаны помечаться
        # так же. ЗАМЕРЕНО сразу после первой правки: «пёс CAN_DO охранять»
        # осталось predicate=fact, то есть единичное происхождение рассказом
        # осело бы как вечное знание о виде — ровно тот дефект, из-за которого
        # события и вынесли в отдельную секцию (EVENTS_TASK.md).
        if made and self._predicate_tense(predicate) == "past":
            for f in res.facts[mark_from:]:
                f.predicate = "event"
        return True

    def _relative_event(self, res: SentenceFacts, negs, predicate, subject,
                        action, obj, order, evidence) -> None:
        """
        Событие придаточного. Собирается здесь, а не в _collect_events:
        тот метод берёт субъект из плоского разбора, а у придаточного субъект
        определён падежом союзного слова.
        """
        tense = self._predicate_tense(predicate)
        if tense != "past":
            return
        res.events.append(EventRecord(
            subject=subject or "?", action=action, obj=obj or "",
            tense=tense, order=order, evidence=evidence,
            negated=negs.negates_pair(action, predicate)))

    @staticmethod
    def _relative_object(res: SentenceFacts, action: str, evidence: str) -> str:
        for f in reversed(res.facts):
            if f.head == action and f.relation == "OBJECT_OF":
                return f.tail
        return ""

    def _purpose_part(self, clause: str, res: SentenceFacts,
                      evidence: str) -> bool:
        """
        Придаточное ЦЕЛИ: «спрятала мышь, чтобы не ловить её».

        ЗАМЕРЕНО до правки: такая часть не давала НИ ОДНОЙ связи — личный
        глагол в ней единственный, и он стоит в главной части, а общий разбор
        без сказуемого возвращает пустоту. Отношение PURPOSE для этого случая
        в реестре ЕСТЬ (core/storage.py, «пришёл -> увидеть»), просто
        придаточное цели до него не доходило.

        Отрицание берётся из области САМОЙ части, поэтому «чтобы НЕ ловить»
        отрицает цель, а не главное действие — ровно то, что было перевёрнуто.
        """
        toks = self._tag_prepositions(analyze(clause))
        if any(t.get("relative") for t in toks):
            return False        # это придаточное определения, не цели
        verbs = [t for t in toks if t.get("content")
                 and t.get("role") == "action"
                 and any(v["pos"] in VERB_POS for v in t["variants"])
                 and t["lemma"] not in COPULAS]
        if not verbs:
            return False
        if any(any(x.get("person") or x.get("tense") for x in v["variants"])
               for v in verbs):
            return False        # там есть своё сказуемое — справится общий разбор
        inf = self._pick(verbs[0], ("INFN",))
        if inf is None:
            return False
        head = res.main_action
        negs = NegationScope(toks)
        if not head:
            res.notes.append("придаточное цели «%s»: главное действие не найдено, "
                             "цель НЕ записана" % clause)
            return True
        res.facts.append(Fact(
            head=head, relation="PURPOSE", tail=inf["lemma"],
            evidence=evidence,
            negated=negs.negates_pair(head, verbs[0]),
            confidence=self._conf(inf, False)))
        return True

    # ── Падеж союзного слова и антецедент ──────────────────

    def _relative_case(self, rel: dict, others: List[dict]) -> str:
        """
        Падеж союзного слова — его решает СОСТАВ придаточного, а не словарь.

        Двусмысленны только им.п. и вин.п. (замер: «который» -> [им. 0.83,
        вин. 0.17], «которая» -> только им.п.), прочие падежи однозначны.

        Им./вин. разрешаются по тому, есть ли в придаточном слово, которое
        винительным быть НЕ может: такое слово и есть подлежащее, значит
        субъект занят КЕМ-ТО ДРУГИМ, и союзному слову остаётся объект.
        Проверено на обеих парных конструкция:
          «который бросил мальчик» — «мальчик» только им.п. -> который=вин.
          «который охранял дом»   — «дом» может быть вин.п., обязательного
                                    подлежащего нет -> который=им.
        Частота тут бессильна: она всегда даёт им.п. (0.83), то есть всегда
       агентом антецедент, и одна из двух конструкций была бы неверной.
        """
        cases = [v["case"] for v in rel["variants"] if v["pos"] == "ADJF"]
        if not cases:
            return ""
        core = {c for c in cases if c in ("nom", "acc")}
        if core == {"nom"}:
            return "nom"
        if core == {"acc"}:
            return "acc"
        if core == {"nom", "acc"}:
            obliged = [t for t in others
                       if self._has_case(t, ("NOUN", "ADJF"), "nom")
                       and not self._can_be_accusative(t)]
            return "acc" if obliged else "nom"
        for want in ("gen", "dat", "ins", "loc"):
            if want in cases:
                return want
        return cases[0]

    @staticmethod
    def _can_be_accusative(tok: dict) -> bool:
        """
        Может ли слово ВООБЩЕ стоять в винительном падеже.

        Тот признак, который снимает омонимию им./вин. там, где частота
        бессильна. ЗАМЕРЕНО: «мальчик» и «река» — только им.п.; «дом», «мяч»,
        «мышь» — им.п./вин.п. Значит в «который бросил мальчик» агент —
        мальчик, больше некому; частотный же разбор всегда выбирал бы им.п. у
        «который» и делал агентом мяч.
        """
        return any(v["case"] == "acc"
                   for v in tok.get("variants", [])
                   if v["pos"] in ("NOUN", "ADJF", "NUMR"))

    @staticmethod
    def _antecedent(sentence: str, rel: dict, clause: str) -> str:
        """
        Определяемое слово — ближайшее существительное ЛЕВЕЕ придаточного.

        Позиция, а не догадка о смысле: придаточное стоит при определяемом
        слове. Именно существительное: местоимение узлом графа не является, и
        «он, который...» породило бы узел «он».
        """
        head = sentence
        if clause and clause in head:
            head = head[:head.index(clause)]
        elif clause:
            low = head.lower()
            cut = low.find(str(rel.get("word", "")).lower())
            if cut > 0:
                head = head[:cut]
        for t in reversed(analyze(head)):
            if not t.get("content") or t.get("role") != "subject":
                continue
            if t.get("prep"):
                continue
            variants = Morph.parse_all(t["word"])
            if variants and variants[0]["pos"] == "NOUN":
                return t["lemma"]
        return ""

    def _extract_flat(self, sentence: str, order: int = 0) -> SentenceFacts:
        res = SentenceFacts(sentence=sentence.strip())
        toks = self._tag_prepositions(analyze(sentence))
        res.tokens = toks

        content = [t for t in toks if t["content"]]
        if not content:
            res.notes.append("нет значимых слов")
            return res

        # ОТРИЦАНИЕ СЧИТАЕТСЯ ПО ОБЛАСТИ ДЕЙСТВИЯ, а не по предложению целиком:
        # «не A, а B» отрицает только A (см. NegationScope и NEGATION_TASK.md).
        negs = NegationScope(toks)
        neg = negs.any            # «в предложении есть отрицание» — для отчётов
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

        # НУЛЕВАЯ СВЯЗКА (замер, изменивший отношение). Русское определение
        # обходится без глагола: «Кит не рыба, а млекопитающее» — связки нет
        # вовсе. ЗАМЕРЕНО: без сказуемого вершиной становилось подлежащее, и
        # именная часть уезжала как OBJECT_OF:
        #     кит --OBJECT_OF--> млекопитающее   («кит — объект действия»)
        # хотя с тире та же связь выходила IS_ROLE. Отношение выбирал ТИРЕ,
        # а не разбор. Где признаков именного предложения два (нет ни глагола,
        # ни модальности/меры + ИМ.п. после подлежащего, отделённого частицей
        # «не» или связочным словом) — ставится НУЛЕВАЯ СВЯЗКА быть/nom, и
        # именная часть идёт общей ветвью связки (IS_ROLE), как в «Вася — это
        # инженер». Придумано не слово, а отсутствие сказуемого: узла «быть»
        # в графе не появляется, потому что имя нулевой связки не используется
        # как head ни одной связи.
        if (pred is None and subj is not None
                and not any(t["role"] == "action" for t in content)
                and not money_sentence and not value_sentence):
            zero_named = copula_marker or negs.denied
            if zero_named and any(
                    t is not subj and t["role"] == "subject"
                    # БЕЗ ПРЕДЛОГА. ЗАМЕРЕНО на «У кошки есть хвост»: «есть» —
                    # связочное слово, предлог «у» морфологией к слову не
                    # приклеен, и «кошки» (разобрано ИМ.п.) стало именной
                    # частью: хвост --IS_ROLE--> кошка рядом с верным
                    # HAS_PART. Предложное слово — косвенный член, именной
                    # частью оно быть не может.
                    and not t.get("prep")
                    and self._has_case(t, ("NOUN", "ADJF"), "nom")
                    and self._agrees_number(t, self._predicate_number(subj))
                    for t in content):
                pred = {
                    "word": ZERO_COPULA, "lemma": ZERO_COPULA, "pos": "COP0",
                    "content": True,
                    "case": "nom", "number": subj.get("number", ""),
                    "role": "property", "score": 1.0, "variants": [
                        {"word": ZERO_COPULA, "lemma": ZERO_COPULA, "pos": "VERB",
                         "case": "nom",
                         "number": subj.get("number", ""), "gender": "",
                         "tense": "pres", "person": "3", "score": 1.0,
                         "grammemes": frozenset()}],
                }
                # «не» при нулевой связке отрицает ИМЕННУЮ ЧАСТЬ, а не субъ-
                # ект: в «Кит не рыба» частица стоит перед сказуемым, а не
                # перед «кит». Отдельная строка вместо мутации отрицания
                # внутри самого предложения — так её видно в отчёте.
                if negs.denied and subj.get("word") in negs.denied:
                    negs.denied.remove(subj["word"].lower())

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
                res.facts.append(self._fact(subj, pred, "CAN_DO", res.sentence, negs))
                if pred.get("prep"):
                    res.facts.append(Fact(subj["lemma"], "ABOUT",
                                          pred.get("prep") + " " + pred["lemma"],
                                          evidence=res.sentence,
                                          negated=negs.negates_pair(
                                              subj["lemma"], pred),
                                          confidence=0.4))
            elif kind == "property" and subj is not None \
                    and not self._is_zero_copula(pred):
                # Нулевая связка — НЕ свойство: её лемма служебный символ, а
                # не слово языка. Именную часть собирает ветвь связки ниже.
                res.facts.append(self._fact(subj, pred, "HAS_PROPERTY",
                                            res.sentence, negs))
            # именная часть при связке («Вася — это инженер», «кит не рыба»)
            if pred is not None and (self._is_copula_form(pred)
                                     or self._is_zero_copula(pred)):
                # «Вася является инженером»: Тв.п. после СВЯЗКИ — именная часть,
                # а не орудие. Тот же Тв.п. после смыслового глагола («открыла
                # мышью») остаётся инструментом, поэтому проверять надо здесь.
                rest = [t for t in content if id(t) not in (subj_taken, id(pred))
                        and not t.get("prep")]
                for t in rest:
                    v = self._pick(t, ("NOUN", "ADJF"), case="ins") or \
                        self._pick(t, ("NOUN", "ADJF"), case="nom")
                    if v is not None:
                        rel = "IS_ROLE" if v["pos"] == "NOUN" else "HAS_PROPERTY"
                        res.facts.append(self._fact(subj or pred, t, rel,
                                                    res.sentence, negs, variant=v))
                        break

        # Владение («у кошки есть хвост») разбирается ДО всего: оно меняет и
        # субъекта, и отношение.
        owner, thing = self._possession(toks)
        if owner is not None and thing is not None:
            ov = self._pick(owner, ("NOUN",), case="gen") or owner["variants"][0]
            tv = self._pick(thing, ("NOUN",), case="nom") or thing["variants"][0]
            res.facts.append(self._fact_from_variant(
                ov["lemma"], thing, "HAS_PART", res.sentence, negs, tv))
            used = {id(owner), id(thing), id(pred)}
            self._attach_adjectives(res, content, used, negs)
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
                    res.sentence, negs, mod_variant))
                # Добавляем CAN_DO(субъект, инфинитив)
                res.facts.append(self._fact(subj, inf, "CAN_DO",
                                            res.sentence, negs, variant=inf))
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
                        evidence=res.sentence,
                        negated=negs.negates_pair(anchor, pred),
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
            # «кошка это животное» и «кит не рыба»: именная часть стоит в
            # ИМ./Вин.п., но НИКАКОГО действия в предложении нет — называть её
            # OBJECT_OF значит изобрести действие, которого не было.
            naming = ((pred is not None and self._is_zero_copula(pred))
                      or (pred is None and copula_marker))
            if (naming and not verbs_present
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
                t, rel, res.sentence, negs, variant))

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
                        pred["lemma"], t, "PURPOSE", res.sentence, negs, inf))
                    used.add(id(t))

        # 3) прилагательные без явного сказуемого-прилагательного
        self._attach_adjectives(res, content, used, negs)

        # «Вася — это инженер»: инженер тоже в ИМ.п. и согласуется в числе, но
        # это именная часть, а не второй кандидат в подлежащие. Считать его
        # конфликтом — значит кормить граф ложными предупреждениями.
        if not self._is_copula_form(pred) and not self._is_zero_copula(pred) \
                and not copula_marker:
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
        self._collect_events(res, subj, pred, order, negs)
        # ОЖИДАНИЕ из конструкции «X попросил Y <инфинитив>» — тоже динамика.
        self._collect_expectations(res, content, subj, pred, order, negs)
        # ГДЕ КОНЧИЛОСЬ «НЕ» — громко, если правка что-то изменила: без строки
        # в отчёте нельзя отличить «отрицания не было» от «область не определили».
        if negs.note():
            res.notes.append(negs.note())
        # Синоним по аппозиции. Роль становится известна ДО связывания имён,
        # поэтому возвращается отдельно (см. SentenceFacts.aliases).
        res.aliases = apposition_aliases(res.tokens, res.sentence)

        # Одно и то же отношение может родиться из разных ветвей (например,
        # «кошка пушистая» — и из сказуемого, и из привязки прилагательного).
        # Оставляем самое уверенное: дубли в графе выглядели бы как подтверждение
        # факта, которого не было.
        res.facts = self._dedup(res.facts)
        # Сказуемое ГЛАВНОЙ части нужно придаточному цели, которое разбирается
        # отдельно (см. _purpose_part). Нулевая связка — не действие: её «лемма»
        # служебный символ, и вешать на него PURPOSE значило бы изобрести глагол.
        if pred is not None and not self._is_zero_copula(pred):
            res.main_action = pred["lemma"]
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
                        pred: Optional[dict], order: int, neg) -> None:
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
            negated=(neg.negates(pred) if isinstance(neg, NegationScope)
                     else bool(neg))))
        # Факты этого предложения — ДИНАМИКА, а не онтология.
        for f in res.facts:
            f.predicate = "event"

    def _collect_expectations(self, res: SentenceFacts, content: List[dict],
                              subj: Optional[dict], pred: Optional[dict],
                              order: int, neg) -> None:
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
            # Отрицание ожидания смотрим У ИНФИНИТИВА, а не у глагола просьбы:
            # «попросила не рассказывать» отрицает действие, а в «не A, а B»
            # частица может стоять в другой части предложения.
            negated=(bool(neg.negates(inf_tok))
                     if isinstance(neg, NegationScope) else bool(neg))
            or pred["lemma"] in FORBID_VERBS,
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

    @staticmethod
    def _is_zero_copula(tok: Optional[dict]) -> bool:
        """Сказуемое НЕ ВЫРАЖЕНО словом (наша нулевая связка)."""
        return bool(tok) and tok.get("lemma") == ZERO_COPULA

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

        ПОЗИЦИЯ ПРОИГРЫВАЕТ МОРФОЛОГИИ, КОГДА ТА СВИДЕТЕЛЬСТВУЕТ (п.A1).
        Позиционная эвристика работает ровно на ПРЯМОМ порядке слов, а обратный
        порядок она ломает молча и инвертирует смысл. Замер 24.10 (14 пар
        «прямой/обратный», tests/test_wordorder_contract.py): 11/14, провалено
            «Огонь гасит вода.»      -> огонь CAN_DO гасить, гасить OBJECT_OF «вод»
            «Мяч бросил мальчик.»    -> мяч CAN_DO бросить   (агент потерян)
            «Чертёж проверил инженер.» -> чертёж CAN_DO проверить
        Дальше по цепочке зависимое слово становится субъектом, а настоящее
        подлежащее скатывается в дополнение, где первым в probe стоит вин.п.,
        и «вода» проигрывает «вод» (у «огонь» вин.п. вероятнее им.: 0.63/0.37).

        Снимаем омонимию ТЕМ ЖЕ признаком, что уже пойман в _relative_case
        (п.18): слово, которое ВООБЩЕ не может стоять в винительном, обязано
        быть подлежащим — кому-то действовать надо. «вода» не может (замер:
        parse_all даёт nom 0.99 / gen 0.01, аккузатива ровно ноль), «огонь»
        может — значит субъект «вода». Ровно ОДИН такой кандидат — иначе
        морфология молчит и право выбора честно остаётся у позиции.
        Замер условия «ровно один» (мутация len(obliged)==1 -> obliged):
        различимых предложений на корпусе 440 фраз найдено РОВНО ОДНО, и то
        мета-фраза про это самое правило, где оба варианта одинаково неверны.
        То есть ветка «двух обязательных им.п.» на реальных текстах сейчас
        НЕДОСТИЖИМА и тестом не покрывается; оставляю её намеренно — это
        требование не угадывать, когда морфология молчит, а не оптимизация.

        Guard по _clause_pred_role обязателен, это не украшение: в ИМЕННОМ
        предложении винительного нет в природе, и без него «Кит не рыба, а
        млекопитающее» терял субъекта (замерено — три теста
        test_negation_contract падали). Расхождение с прежним поведением на
        корпусе 342 фраз — ровно 4 предложения, и все 4 это ИСПРАВЛЕНИЕ
        («отрицание утверждает» -> «онтология утверждает»)."""
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
        # Обязательный им.п. сильнее позиции — НО ТОЛЬКО при переходном
        # сказуемом: в именном предложении винительного нет, и там этот признак
        # ничего не утверждает (см. docstring).
        if pred is not None and self._clause_pred_role(pred) == "action":
            obliged = [t for t in cands if not self._can_be_accusative(t)]
            if len(obliged) == 1:
                return obliged[0]
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
                           used_ids: set, neg) -> None:
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
              evidence: str, neg, variant: Optional[dict] = None) -> Fact:
        """Факт из двух токенов. variant задаётся, когда роль уже известна
        (именная часть связки), иначе её вычисляет _case_role.

        neg — либо NegationScope (тогда отрицание берётся ПО ОБЛАСТИ ДЕЙСТВИЯ
        частицы, по хвостовому токену), либо старый булев флаг "в предложении
        есть «не»". Оба законны: вызывающий может точно знать, что связь
        отрицается целиком (объект при отрицанном действии)."""
        if variant is None:
            if tail_tok.get("role") == "action":
                # Хвост — ГЛАГОЛ. Брать вариант у _case_role нельзя: тот ищет
                # существительные, а у глагольной формы словарь обязан дать
                # омонима. ЗАМЕРЕНО на «Дом, где жил пёс, сгорел»: для «жил»
                # словарь даёт ГЛАГОЛ жить 0.96 и СУЩЕСТВИТЕЛЬНОЕ жила 0.01,
                # и в граф легла «жила» — форма вместо леммы. Два узла («жить»
                # и «жила») разъезжаются по графу навсегда, и собрать их
                # обратно уже нельзя.
                variant = (self._pick(tail_tok, VERB_POS)
                           or tail_tok["variants"][0])
            else:
                _rel, variant = self._case_role(tail_tok)
                if variant is None:
                    variant = tail_tok["variants"][0]
        return self._fact_from_variant(head_tok["lemma"], tail_tok, rel,
                                       evidence, neg, variant)

    def _fact_from_variant(self, head_lemma: str, tail_tok: dict, rel: str,
                           evidence: str, neg, variant: dict) -> Fact:
        n_ok = self._n_variants(tail_tok, ("NOUN", "ADJF", "VERB"))
        ambiguous = n_ok > 1
        return Fact(
            head=head_lemma, relation=rel, tail=variant["lemma"],
            evidence=evidence,
            negated=(neg.negates_pair(head_lemma, tail_tok)
                     if isinstance(neg, NegationScope) else bool(neg)),
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
        """
        Есть ли в предложении придаточное — по РАЗМЕТКЕ, а не по списку слов.

        Союзное слово придаточного больше НЕ значимое (content=False), поэтому
        проверять content на нём бессмысленно: признак живёт в ключе "relative".
        Забыть об этом — значит потерять заметку на придаточном, а заметка есть
        единственное, чем экстрактор честно признаётся в неумении.
        """
        return any(t.get("relative")
                   or (t.get("word") in SUBORD_CONJ
                       and t.get("pos") in ("CONJ", "ADVB"))
                   for t in toks)


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
