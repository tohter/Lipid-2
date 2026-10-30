# -*- coding: utf-8 -*-
"""
Русский лемматизатор, КОТОРЫЙ ПРИНОСЯТ ядру (шаг A2.2 границы языка).

Почему файл живёт здесь, а не в core/. До этого шага в core/records.py сидел
`import pymorphy3`: ядро само звало русский морфологический движок, то есть
знало язык и само решало, что с ним делать. По закону
(experiments/IR_BOUNDARY.md §1) ядро обязано знать только IR; язык обязан жить
в lang/<xx> или приходить оттуда. Здесь — именно «оттуда».

ПОЧЕМУ core.morph.Morph, а не pymorphy3 напрямую. Два ответа, оба измерены:

  1. КАНАЛ. Ядру всё равно, как зовут движок. Ему нужна функция
     слово -> лемма. Morph.lemma уже даёт её и уже умеет деградировать без
     pymorphy3, таскать второй экземпляр движка в процесс — второй независимый
     кэш и второй способ молча разойтись с первым.

  2. ЭКВИВАЛЕНТНОСТЬ. ЗАМЕРЕНО на 1048 словах (сущности и наблюдения живого
     data/memory/lipid_world.json + весь корпус corpus_dev/corpus_heldout +
     слова, на которых эвристика известна как плохая: сеть/сети, кошка/кошачьи,
     пчёлы, флюгер/флюгера): расхождений Morph.lemma с pymorphy3 normal_form —
     0 из 1048. Замена шва НЕ меняет качество извлечения; это проверено до
     правки, а не после того, как тесты позеленели.

ВНЕДРИТЬ ОБЯЗАНЫ. Ядро внедрить себе язык не может (потолок
C_core_reaches_lang = 0 вечен), поэтому место установки — этот слой или
production-вход. Кто пишет мир и забыл вызвать install(), тот получает
эвристику; молча — нельзя, records орёт в stderr, а tests
/test_lemma_service_contract.py падает: в наборе есть проверка, что КАЖДЫЙ
производственный вход, сводящий формы слов, службу принёс.
"""
from __future__ import annotations

from typing import Callable, Optional

_installed = False
_provider: Optional[Callable[[str], str]] = None


def _lazy_lemma(word: str) -> str:
    """
    Лемма через Morph, НО pymorphy3 поднимается только на СЛОВЕ, не на импорте.

    У учителей (teachers/) лёгкий канал знания обязан работать на любом python3
    без тяжёлого стека — тот же порядок, что в core/encoder.py. Поэтому здесь
    отсроченная отсылка, а не готовая Morph.lemma: install() стоит двух
    присваиваний, а движок включается, когда лемму ПРАВДИВО попросили.
    """
    from core.morph import Morph

    return Morph.lemma(word)


def provider() -> Callable[[str], str]:
    """
    Русский лемматизатор: слово -> лемма, "" — если формы нет.
    """
    global _provider
    if _provider is None:
        _provider = _lazy_lemma
    return _provider


def _lazy_pos(word: str) -> str:
    """
    Часть речи через Morph, pymorphy3 поднимается на СЛОВЕ, а не на импорте.

    Та же отсрочка, что у _lazy_lemma: лёгкий канал знания обязан работать на
    любом python3 без тяжёлого стека.
    """
    from core.morph import Morph

    return Morph.pos(word)


def pos_provider() -> Callable[[str], str]:
    """Русский распознаватель части речи: слово -> «VERB»/«INFN»/«NOUN»/…"""
    return _lazy_pos


def install():
    """
    Один раз на процесс: отдать ядру русскую морфологию ЦЕЛИКОМ.

    Внедряет ОБЕ службы — сведение форм (core.records, шаг A2.2) и часть речи
    (core.actions, шаг A2.3). Одна точка входа намеренна: два отдельных
    установщика — это два способа внедрить половину и получить процесс, где
    леммы сведены, а глаголы распознаются справочником. Цена такой половины
    измерена: неверный диагноз asked_empty на живых глаголах памяти.

    Возвращает True, если установка произошла СЕЙЧАС, False если уже стояла.
    Повторный вызов безопасен, но число установок должно быть видно: это
    единственный способ отличить «службу принесли» от «её никто не просил».
    """
    global _installed
    import core.records as R
    import core.actions as A
    R.set_lemma_service(provider(), name="lang.ru.lemma:core.morph.Morph.lemma")
    A.set_pos_service(pos_provider(), name="lang.ru.lemma:core.morph.Morph.pos")
    first = not _installed
    _installed = True
    return first


def is_installed() -> bool:
    """Ставили ли СЛУЖБУ ЭТИМ модулем. Состояние ядра — в R.lemma_service_state()."""
    return _installed
