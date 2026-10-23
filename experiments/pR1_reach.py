"""
ЗАМЕР п.A-остаток1, шаг 1: А ДОСТИЖИМА ЛИ ВЕТКА ВООБЩЕ?

Очередь обещает: «РОВНО ОДНА категория от Вики — 7 слов из 22 — отвечает как
род, хотя это алфавитная позиция». Но это вывод из СНИМКА категорий, а не из
ответа мозга. Ведущий путь ответа — definition, а Вики отдаёт его всегда.
Значит ветка «одна категория = род» могла стать недостижимой ещё до этой
правки, и чинить её — значит правать недоступный код.

Проверяется четырьмя путями, которыми категория вообще попадает в память:
  1. Вики-канал живой (то, что очередь называет виновником);
  2. текст с ИС_A («Бегемот — это млекопитающее») — тут одиночная категория
     есть настоящее мнение, и ответ «X — это Y» ПРАВИЛЬНЫЙ;
  3. ответ пользователя — то же самое;
  4. старая запись без definition (формат до починки) — вот где ветка живёт.

Категория ищется не по тексту ответа, а по факту: что именно мозг поставил
на место «рода». Сличение с сырыми категориями Вики даёт ответ честно.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def ask(memory_dir, text, wiki=False, llm=False, user=False):
    """Одиночный вопрос в изолированной памяти. Отдельным процессом, чтобы
   состояние одного пути не влияло на другое."""
    code = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "from brain.lipid2_brain import Lipid2Brain\n"
        "b = Lipid2Brain(memory_dir=%r)\n"
        "b.use_llm_teacher = %s; b.use_wiki_teacher = %s; b.ask_enabled = %s\n"
        "b.learn_experience = False\n"
        "print('ОТВЕТ::' + b.process_input(%r))\n"
        % (ROOT, memory_dir, llm, wiki, user, text)
    )
    env = dict(os.environ, LIPID2_STATUS="0", LIPID2_MEMORY_DIR=memory_dir)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=600)
    out = [l for l in (r.stdout + r.stderr).splitlines() if "ОТВЕТ::" in l]
    return out[0].split("ОТВЕТ::", 1)[1] if out else f"ВЫХОД rc={r.returncode}"


def main():
    print("──────── ПУТЬ 1: ЖИВОЙ ВИКИ-КАНАЛ (7 слов из замера) ────────")
    d = tempfile.mkdtemp(prefix="pr1_wiki_")
    single = ["книга", "река", "город", "экосистема", "луна", "валенок", "рыбы"]
    from brain.lipid2_brain import Lipid2Brain
    from core.records import meaningful_categories
    b = Lipid2Brain(memory_dir=d)
    b.use_llm_teacher = False
    b.use_wiki_teacher = True
    b.ask_enabled = False
    b.learn_experience = False
    hit_genus = 0
    for w in single:
        b._trigger_curiosity_and_learn(w)
        ans = b.process_input(f"что такое {w}")
        ent = (b.find_in_memory(w) or {}).get("base_entity", {})
        hier = meaningful_categories(w, ent.get("hierarchy") or [])
        genus = f"{w.capitalize()} — это " in ans
        hit_genus += genus
        print(f"  {'ЭТО РОД' if genus else 'не род '} {w:11s} категорий={len(hier)} "
              f"{str(hier)[:40]:40s} | {ans[:70]}")
    print(f"  → из 7 слов ответило «X — это <категория>»: {hit_genus}")
    shutil.rmtree(d, ignore_errors=True)

    print("\n──────── ПУТЬ 2: ТЕКСТ (одиночная категория = настоящее мнение) ────")
    d2 = tempfile.mkdtemp(prefix="pr1_txt_")
    b2 = Lipid2Brain(memory_dir=d2)
    b2.use_llm_teacher = b2.use_wiki_teacher = b2.ask_enabled = False
    b2.learn_experience = False
    b2.learn_from_text("Бегемот — это млекопитающее.")
    print("  " + b2.process_input("что такое бегемот")[:110])
    ent = (b2.find_in_memory("бегемот") or {}).get("base_entity", {})
    print(f"  definition={str(ent.get('definition'))[:40]!r} hierarchy={ent.get('hierarchy')}")
    shutil.rmtree(d2, ignore_errors=True)

    print("\n──────── ПУТЬ 4: СТАРАЯ ЗАПИСЬ БЕЗ definition ────────")
    d4 = tempfile.mkdtemp(prefix="pr1_old_")
    world = {"entities": {
        "экосистема": {"concept": "экосистема", "definition": "",
                       "hierarchy": ["экологические термины"],
                       "properties": [], "actions": [],
                       "provenance": [{"source": "wiki", "evidence": "Экосистема"}]}},
        "relations": [], "observations": {}, "unknown_words": {}, "transitions": {}}
    with open(os.path.join(d4, "lipid_world.json"), "w", encoding="utf-8") as fh:
        json.dump(world, fh, ensure_ascii=False)
    print("  " + ask(d4, "что такое экосистема"))
    shutil.rmtree(d4, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
