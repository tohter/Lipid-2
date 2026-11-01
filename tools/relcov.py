#!/usr/bin/env python3
"""Покрытие реестра отношений: ДВА числа, а не одно (замер, а не утверждение).

    LIPID2_STATUS=0 .venv/bin/python tools/relcov.py

ПОЧЕМУ ЧИСЕЛ ДВА. Прежняя мера называлась «отношения, которые в принципе
доезжают до мира», а смотрела ТОЛЬКО в sf.facts и sf.aliases экстрактора —
то есть меряла одно и называла другим. Витрина (tools/presnum.py) продавала
это число как состояние мира. Замер 01.11 на ОДНОМ списке фраз:

    имён у экстрактора ....... 16 из 17
    имён дошло до мира ....... 15 из 17

Разрыв — не шумы, а две разные вещи, и терять любую из них нельзя:

  (1) ЧТО УМЕЕТ ЭКСТРАКТОР. Верхняя граница: сколько имён вообще рождается из
      русской фразы. Инструмент языковой границы.
  (2) ЧТО ОСЕЛО В МИРЕ. нижняя граница: сколько доехало до наблюдений/событий
      после brain/learner.py. Здесь видна ЦЕНА КОНВЕЙЕРА: политика «событие не
      становится знанием» (_apply_fact: is_event -> skipped) съедает падежные
      роли, и INSTRUMENT с RECIPIENT до мира не доходят вовсе.

МНОЖЕСТВА НЕ ВЛОЖЕНЫ ДРУГ В ДРУГА — и это главное, что нельзя спрятать за
двумя числами. «Отношение дошло, хотя экстрактор его не порождал» — нормальный
случай: IS_A рождает сам learner, закрепляя глагол как действие
(_store_action: «ловить» hierarchy [«действие»]), без Учителя и без LLM. Поэтому
заперто не «у экстрактора >= до мира как включение множеств» (это было бы
враньём: см. tests/test_relcov_contract.py), а НАЗВАНЫ оба расхождения.

CASES намеренно бедный и открытый: это НЕ тест на качество, это граница —
какие отношения в принципе недостижимы из русской фразы. Поэтому список
предложений можно дополнять, но «НЕ ПОРОЖДАЕТСЯ» имеет смысл только как
«не порождается НИ ОДНИМ из известных нам способов».
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extract import SimpleExtractor          # noqa: E402
from core.storage import RelationType             # noqa: E402

# По одному образцу на отношение. Пометки справа — НАЗНАЧЕНИЕ пробы, а не её
# результат: «дыра» тут написана один раз наугад и три раза оказывалась дырой
# не отношений, а формулировки пробы (ABOUT, SYNONYM_OF, IS_A — см. докстринг
# меры). Настоящий результат печатает main(), и только ему можно верить.
CASES = [
    "Кошка пушистая.",                                  # HAS_PROPERTY
    "Кошка ловит мышь.",                                # CAN_DO, OBJECT_OF
    "Вася — инженер.",                                  # IS_ROLE
    "Пожар погас вследствие дождя.",                    # CAUSES
    "Пёс спит на подоконнике.",                         # LOCATED_IN
    "Витя дал кошке корм.",                             # RECIPIENT
    "Кошка открыла мышь лапой.",                        # INSTRUMENT
    "Кошка спрятала мышь, чтобы не есть её.",           # PURPOSE
    "Пёс думает о погоде.",                             # ABOUT (прошей: с
    #                                                   # местоимением ABOUT
    #                                                   # не проверяется — там
    #                                                   # актор-политика, см.
    #                                                   # test_alias_contract)
    "Солнце светит днём.",                              # CIRCUMSTANCE
    "Вода кипит при ста градусах.",                     # CONDITION
    "Кошка весит три килограмма.",                      # VALUE
    "Хвост собаки мокрый.",                             # PART_OF
    "У кошки есть хвост.",                              # HAS_PART
    "Гиппопотам, или бегемот, живёт в реке.",           # SYNONYM_OF
    "Кошка — млекопитающее.",                           # IS_ROLE (связка X — Y)
]


def extracted_relations() -> set:
    """(1) Что УМЕЕТ экстрактор: имена, рождающиеся из русской фразы.

    Прежнее имя этой функции — produced_relations — врало: docstring обещал
    «доезжают до мира», а код смотрел только в sf.facts. Теперь имя говорит
    ровно то, что меряется.

    ЗАМЕР, ИЗМЕНИВШИЙ МЕРУ (31.10). Два прежних искажения:
      (а) ABOUT числился «дырой», хотя рождается: проба стояла «Он думает о
          погоде», а там актор — местоимение, и CAN_DO снимается политикой
          «не приписывать действие первому существительному». ABOUT тут ни
          при чём; с существительным («Пёс думает о погоде») отношение на
          месте. Мера принимала ОТКАЗ ПО ДРУГОЙ ПРИЧИНЕ за свою дыру.
      (б) SYNONYM_OF числился недостижимым, потому что смотрели только в
          .facts. А пара «X, или Y» приезжает полем sf.aliases и в мир её
          несёт brain/learner._record_aliases — отношение ДОСТИЖИМО, просто
          не фактом. Вопрос «достижимо ли из фразы» шире вопроса «лежит ли в
          .facts», поэтому канал aliases считается.
    """
    ex = SimpleExtractor()
    got, via_alias = set(), set()
    for s in CASES:
        sf = ex.extract(s, 1)
        for f in sf.facts:
            got.add(f.relation)
        if getattr(sf, "aliases", None):
            via_alias.add("SYNONYM_OF")
    return got | via_alias


def world_relations() -> set:
    """(2) Что ОСЕЛО В МИРЕ: имена отношений после brain/learner.py.

    Это ДРУГОЙ измеритель, а не тот же список с другим фильтром: здесь
    реально прогоняются те же CASES через Learner и считаются имена из
    наблюдений, полей hierarchy сущностей и событий (cause).

    ТРИ ОБСТОЯТЕЛЬСТВА, без которых число стало бы враньём:

      * УЧИТЕЛЬ ВЫКЛЮЧЕН (teacher=None, allow_teacher=False). Иначе часть
        имён принесла бы не фраза, а LLM, и мера языковой границы меряла бы
        сеть. Побочный эффект: незнакомые слова не изучаются и часть фактов
        скипается. На покрытие это НЕ влияет — проверено тремя прогонами
        подряд (1-й, 2-й и 3-й дали одно и то же: INSTRUMENT и RECIPIENT
        отсутствуют). Но знать про скипы надо, поэтому они не молчат:
        main() печатает счётчик.
      * ПАМЯТЬ ИЗОЛИРОВАНА. Мир живёт во временном файле, живая Вселенная не
        трогается (правило проекта: измеритель не имеет права писать в память;
        заперто tests/test_memory_isolation_contract.py).
      * СОБЫТИЕ — НЕ ЗНАНИЕ. _apply_fact: is_event -> skipped, поэтому у
        фразы в прошедшем времени падежные роли доходят как OBJECT_OF события
        или не доходят вовсе. Это не поломка, а запертая политика
        («кошка съела канарейку» не должна становиться «кошка ест канареек»),
        но именно она и есть цена конвейера, которую второе число и показывает.
    """
    import tempfile

    from brain.learner import Learner
    from core.worldfile import read_world, write_world

    tmp = tempfile.mkdtemp(prefix="l2_relcov_")
    path = os.path.join(tmp, "lipid_world.json")
    write_world(path, {"entities": {}, "observations": {}, "unknown_words": {}})

    def _read():
        return read_world(path)

    def _write(w):
        write_world(path, w)

    lrn = Learner(load_world=_read, save_world=_write,
                  teacher=lambda w: None, core=None, max_teacher_calls=0)
    for s in CASES:
        lrn.learn_from_text(s, source="relcov", allow_teacher=False)

    w = _read()
    got = {str(o.get("relation", "")).upper()
           for o in (w.get("observations") or {}).values()}
    # Иерархия сущностей — это IS_A, записанный НЕ фактом (поле, а не
    # наблюдение). Рождается learner'ом: _store_action закрепляет глагол как
    # действие, «ловить» -> hierarchy [«действие»]. Именно поэтому множества
    # экстрактора и мира НЕ вложены друг в друга.
    for ent in (w.get("entities") or {}).values():
        if isinstance(ent, dict) and (ent.get("hierarchy") or []):
            got.add("IS_A")
        if isinstance(ent, dict) and (ent.get("aliases") or []):
            got.add("SYNONYM_OF")
    # Причина события живёт в событии и остаётся причиной: это CAUSES,
    # доехавшее до мира не наблюдением.
    for ev in (w.get("events") or []):
        if ev.get("cause"):
            got.add("CAUSES")
        if ev.get("purpose"):
            got.add("PURPOSE")
    return {r for r in got if r}


def coverage() -> tuple:
    """(всего_в_реестре, у_экстрактора, до_мира) — три числа из одного места.

    Единая точка: и main(), и tools/presnum.py вызывают ЭТУ функцию. Второй
    пересказ арифметики был бы вторым измерителем одной границы.
    """
    total = len([e.name for e in RelationType])
    return total, extracted_relations(), world_relations()


def main() -> int:
    total, ext, world = coverage()
    reg = {e.name for e in RelationType}
    miss_ext = sorted(reg - ext)
    miss_world = sorted(reg - world)
    only_world = sorted(world - ext)
    lost = sorted(ext - world)
    # АРИФМЕТИКА СПИСКОВ. На этой же мере я уже успела построить враньё:
    # отношение убрали из «НЕ ПОРОЖДАЕТСЯ» справочно, счётчик не тронули — и
    # строка «ИТОГО 17 ПОРОЖДАЕТСЯ 15» обещала два имени в списке, где лежало
    # одно. Числа печатаются вместе, значит обязаны сходиться.
    assert total - len(ext) == len(miss_ext), (
        f"расхождение меры (экстрактор): всего {total}, порождено {len(ext)}, "
        f"в списке недостижимых {len(miss_ext)}")
    assert total - len(world) == len(miss_world), (
        f"расхождение меры (мир): всего {total}, дошло {len(world)}, "
        f"в списке недостижимых {len(miss_world)}")
    print(f"ИТОГО ОТНОШЕНИЙ={total}")
    print(f"У ЭКСТРАКТОРА   ={len(ext)}  (умеет извлечь из фразы)")
    print(f"ДОШЛО ДО МИРА   ={len(world)}  (осело после learner'а)")
    print("УМЕЕТ, НЕ ДОХОДИТ:", " ".join(lost) if lost else "(нет)")
    print("В МИРЕ БЕЗ ЭКСТРАКТОРА:", " ".join(only_world) if only_world else "(нет)",
          "  <- рождает сам learner, не фраза")
    print("НЕ ДОСТИЖИМО ВОВСЕ:", " ".join(miss_world) if miss_world else "(нет)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
