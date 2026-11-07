"""
ЛЕСТНИЦА «ДОЖДЬ И ПОЖАР» — измеритель того, думает ли модель.

ЗАЧЕМ ФАЙЛ. Хозяин велел: «выжечь из модели хотя бы очень простую думалку»,
и запретил вписывать правильный ответ в код. Значит единственный честный
измеритель — список вопросов, у которых правильный ответ ЗНАЕТ ХОЗЯИН, но не
знает программа, и который она обязана собирать сама из базы.

ЛЕСТНИЦА ПОДНЯТА ОТ ПРОСТОГО К СЛОЖНОМУ, каждый шаг требует ровно одного
нового механизма, и отказ обязан сказать, какого:

  1. что такое дождь            — одно поле записи
  2. где бывает дождь           — одна связь LOCATED_IN
  3. что общего у дождя и росы  — пересечение, и узел «роса», которого нет
  4. может ли вода потушить огонь — CAN_DO + OBJECT_OF, без события
  5. может ли дождь потушить лесной пожар — ТРИ шага + масштаб
  6. а морось? а ливень?        — перенос способности на брата по классу

ЧТО СЧИТАЕТСЯ ОТВЕТОМ, а что отказом. Ответ — только когда программа НАЗВАЛА
цепочку, по которой она идёт. «Не знаю» с именем причины — честный отказ, он
тоже результат замера: он говорит, какой механизм следующий. Пустой ответ или
ответ без цепочки считается отказом всегда, даже если слово совпало.

ЗАПРЕТ: в этом файле нет ни одного правильного ответа. Тут только вопросы.
Сравнивает с ожиданием хозяин (или контракт, где ожидание — структура ответа,
а не слово).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LADDER = [
    "что такое дождь",
    "где бывает дождь",
    "что общего у дождя и росы",
    "может ли вода потушить огонь",
    "может ли дождь потушить лесной пожар",
    "может ли морось потушить лесной пожар",
    "может ли ливень потушить лесной пожар",
]


def run(world_path: str) -> int:
    from brain.lipid2_brain import Lipid2Brain
    import tempfile
    d = tempfile.mkdtemp()
    world = json.load(open(world_path, encoding="utf-8"))
    json.dump(world, open(os.path.join(d, "lipid_world.json"), "w"),
              ensure_ascii=False)
    brain = Lipid2Brain(memory_dir=d)
    # УЧИТЕЛЯ ВЫКЛЮЧЕНЫ НАМЕРЕННО. Если ответ получится с живым Gemma, мы не
    # узнаем, думает ли модель или пересказывает большую языковую модель.
    brain.use_llm_teacher = False
    brain.use_wiki_teacher = False
    brain.ask_user_enabled = False

    print(f"=== {os.path.basename(world_path)}: сущностей "
          f"{len(world.get('entities') or {})}, наблюдений "
          f"{len(world.get('observations') or {})} ===\n")
    answered = 0
    for q in LADDER:
        try:
            ans = brain.process_input(q)
        except Exception as exc:                       # noqa: BLE001
            ans = f"💥 {type(exc).__name__}: {exc}"
        ans = " ".join(str(ans).replace("[Lipid-2]:", "").split())
        # Мера обязана ловить СОБСТВЕННЫЕ отказы. Замер 07.11: первая
        # редакция считала ответом всё, где нет слов «Интент»/«не понял»,
        # и отказ думалки «не могу рассуждать: не знаю слов» засчитывался
        # как ✅ — лестница показывала успех там, где модель только что
        # призналась, что не знает слов. Честный признак — ЛЮБОЕ отрицание
        # в ответе, а не список заранее угаданных формулировок.
        low = ans.lower()
        bad = ("интент" in low or "не понял" in low or "не содержит" in low
               or "источники" in low or "💥" in low or len(ans) < 12
               or low.startswith("не ") or " не могу" in low or " не знаю"
               in low or " не нашёл" in low or "не хватает" in low
               or "нет связи" in low or "не могу утверждать" in low)
        if not bad:
            answered += 1
        print(f"  {'✅' if not bad else '❌'} {q}")
        print(f"       {ans[:170]}")
    print(f"\nотвечено {answered} из {len(LADDER)}")
    # Лестница проверяет не только «ответила ли», но и ЧТО ОСТАЛОСЬ ПОСЛЕ.
    # Слово хозяина: высокая тревога = поиск не завершён. Если после прогона
    # отвеченных меньше, чем вопросов, а кортизол спокойный — флаг врёт:
    # система «успокоилась», не найдя ответа.
    gate = getattr(brain, "hormone_gate", None)
    if gate is not None:
        cur = getattr(brain, "curiosity", None)
        goals = len(cur.open_goals()) if cur is not None else -1
        print(f"тревога {gate.cortisol} (ищем={gate.must_search()}), "
              f"открытых целей {goals}, кандидатов в карантине "
              f"{len(cur.staged()) if cur is not None else -1}")
        if answered < len(LADDER) and not gate.must_search():
            print("⚠️ вопрос(ы) не закрыты, а тревога СПОКОЙНАЯ — флаг враньёт")
    return answered


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "experiments", "fire_rain_world.json")
    run(path)
