"""Замер раздувания достоверности: сколько ПОДТВЕРЖДЕНИЙ видит система там,
где человек видит одно основание знания.

Делается ДО и ПОСЛЕ закона происхождения (core/records.py::add_witness).
Отвечает на один вопрос: сколько раз одна и та же мысль засчитывается как
новое подтверждение, если её пересказали другими словами.

    LIPID2_STATUS=0 .venv/bin/python experiments/measure_blowup.py

Замер «до» снимается ТЕМ ЖЕ прогоном на другом дереве, а не по памяти:

    git worktree add -q /tmp/head_wt HEAD
    LIPID2_STATUS=0 .venv/bin/python /tmp/measure_chain.py /tmp/head_wt HEAD
    LIPID2_STATUS=0 .venv/bin/python /tmp/measure_chain.py . РАБОТ

Никакого ответа про дождь и пожар здесь нет и быть не должно — замеряется
аппарат, а не знание.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lang.ru.lemma import install                                  # noqa: E402
install()

from core.records import (add_witness, fact_weight,                 # noqa: E402
                          independent_bases, lineage_of, observation_weight)


def _rec(source: str, evidence: str, document: str = "",
         confidence: float = 0.55) -> dict:
    """Запись так, как её создаёт писатель: один свидетель = support 1.

    Вес НЕ пишется с головы: он считается той же fact_weight, что и у
    настоящих писателей. Сочинённое число веса — ровно тот грех, против
    которого восставал урок про «14 из 20 отношений».
    """
    return {
        "head": "дождь", "relation": "CAN_DO", "tail": "гасить",
        "confidence": confidence, "negated": False, "support": 1,
        "weight": fact_weight(confidence, False, "CAN_DO", 1),
        "ground": "testimony",
        "source": source, "evidence": evidence,
        "lineage": lineage_of(source, document),
    }


def show(title: str, obs: dict) -> tuple:
    """Печатает и число подтверждений, и ВЕС — вес всегда рядом.

    Причина, почему вес печатается НАПРОТЯЖКУ, а не только support: этот
    же замер показал, что закон остановил рост support, а вес всё равно
    уполз 170 -> 207 от четырёх безымянных пересказов. Дыра НЕ в support:
    уверенность записи берётся max-ом из самоотчётов каналов, а weight —
    линейная функция уверенности (90 + 130*conf для CAN_DO). Модели
    называют о себе 0.75...0.90, поэтому самый громкий выигрывает.
    Замерено до правки веса (ядро шага 2.2/2.3 босса — впереди).
    """
    bases = independent_bases(obs)
    support = int(obs.get("support", 0))
    rows = len(obs.get("provenance") or [])
    print(f"  {title}")
    print(f"    support (число подтверждений) = {support}")
    print(f"    отдельных оснований           = {bases}")
    print(f"    строк в журнале свидетелей    = {rows}")
    print(f"    вес связи                     = {observation_weight(obs)} "
          f"(уверенность записи {float(obs.get('confidence', 0.0)):.2f})")
    # Отношение ПЕРЕДАЁТСЯ везде и всегда. Замер без него выглядел бы как
    # падение веса при росте support: fact_weight считает CAN_DO по
    # «уверенной» формуле (90 + 130*conf), а пустое отношение — по обычной
    # (60 + 100*conf). Настоящие писатели отношение передают; замер, который
    # их не имитирует, врал бы о весе.
    if not str(obs.get("relation") or ""):
        print("    (внимание: relation не передан — вес занижен формулой)")
    return support, bases


def case_seven_phrases() -> tuple:
    """Семь формулировок одной мысли из трёх названных мест.

    Честный ответ — ТРИ основания (статья, книга, слово хозяина), а не
    семь. Модель, пересказывающая прочитанное, нового знания не приносит,
    и слово хозяина — своё собственное основание, а не эхо статьи.
    """
    print("\n[1] Семь формулировок, три названных источника")
    obs = _rec("wiki", "дождь помогает тушить пожары", "статья-о-дожде")
    print(f"    вес до:                       = {observation_weight(obs)} "
          f"(уверенность записи 0.55)")
    входящие_свидетельства = [
        ("model", "вода из дождя гасит огонь", ""),                    # пересказ
        ("text", "дождь гасит пламя", "книга-о-лесах"),
        ("gemma", "дождевая вода используется при тушении", ""),        # пересказ
        ("user", "дождь помогает тушить лесные пожары", ""),            # слово хозяина
        ("llm", "дождь способен потушить пожар", ""),                   # пересказ
        ("text", "сильный дождь снижает опасность пожара", ""),         # безымянно
    ]
    for source, evidence, document in входящие_свидетельства:
        add_witness(obs, source=source, evidence=evidence,
                    confidence=0.55, relation="CAN_DO", document=document)
    return show("после семи формулировок:", obs)


def case_retelling_chain() -> tuple:
    """Цепочка пересказов ОДНОЙ базы:article -> model -> model -> model.

    Здесь человек видит ОДНО основание. На HEAD (без закона происхождения)
    этот же прогон даёт support 5 и вес 247 — из ничего, одним пересказом.

    Уверенности здесь НЕ одинаковые: это те числа, которые каналы ВЫДАЮТ О
    САМОМ СЕБЕ в живом прогоне. Разные, а не одинаковые, — и ровно поэтому
    max-уверенность оказывается самоотчётом самой громкой модели.
    """
    print("\n[2] Четыре пересказа одной и той же статьи")
    obs = _rec("wiki", "дождь помогает гасить пожар", "статья-о-дожде")
    for source, evidence, уверенность in [
            ("model", "дождь гасит пожар", 0.75),
            ("gemma", "дождевая вода тушит огонь", 0.80),
            ("llm", "дождь способен гасить пламя", 0.90),
            ("model", "огонь гасится дождевой водой", 0.85)]:
        add_witness(obs, source=source, evidence=evidence,
                    confidence=уверенность, relation="CAN_DO")
    return show("после четырёх пересказов:", obs)


def case_same_article_two_channels() -> tuple:
    """Та же статья, названная двумя каналами, — ОДНО основание.

    Цена, принятая сознательно: теряем подтверждение, никогда не
    приобретаем ложную уверенность.
    """
    print("\n[3] Одна статья из двух каналов")
    obs = _rec("wiki", "дождь помогает гасить пожар", "статья-о-дожде")
    add_witness(obs, source="model", evidence="дождь гасит пожар",
                confidence=0.55, relation="CAN_DO", document="статья-о-дожде")
    return show("после пересказа с названным заголовком:", obs)


def case_two_documents() -> tuple:
    """Две названные книги — ДВА основания: закон не впадает в паралич."""
    print("\n[4] Две разные названные книги")
    obs = _rec("text", "дождь гасит пожар", "книга-о-лесах")
    add_witness(obs, source="text",
                evidence="дождевая вода снижает пожарную опасность",
                confidence=0.55, relation="CAN_DO", document="энциклопедия-огня")
    return show("после второй книги:", obs)


if __name__ == "__main__":
    print("=" * 72)
    print("РАЗДУВАНИЕ ДОСТОВЕРНОСТИ: support против числа оснований")
    print("=" * 72)
    a = case_seven_phrases()
    b = case_retelling_chain()
    c = case_same_article_two_channels()
    d = case_two_documents()
    print("\n" + "=" * 72)
    print("ИТОГ: support никогда не должен превышать число отдельных оснований")
    for name, (support, bases) in (("7 формулировок", a), ("цепочка пересказов", b),
                                   ("одна статья, два канала", c), ("две книги", d)):
        метка = "OK" if support <= bases else "РАЗДУВАНИЕ"
        print(f"  {метка:11} {name:26} support={support} оснований={bases}")
