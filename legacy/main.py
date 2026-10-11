import sys
from sdr import SDR1024
from graph import GraphMemory
from teacher import LLMTeacher
from cortex import PredictiveCortex
from narrator import NarrativeParser

def run_lipid2_narrative_test():
    print("=" * 65)
    print("       НЕЙРОМОРФНОЕ ЯДРО «ЛИПИД-2» (CORTEX-1024) — NARRATIVE TEST")
    print("=" * 65)

    LLM_HOST = "http://localhost:8080"

    # 1. Инициализация компонентов
    graph = GraphMemory()
    teacher = LLMTeacher(host=LLM_HOST)
    cortex = PredictiveCortex(graph=graph, teacher=teacher)
    narrator = NarrativeParser(llm_host=LLM_HOST, graph=graph)

    # 2. Тестовый нарратив
    sample_story = (
        "Кошка поделилась секретом с канарейкой. "
        "Канарейка рассказала секрет всей улице. "
        "Кошка узнала о предательстве, разозлилась и сожрала канарейку."
    )

    print("\n📝 ИСХОДНЫЙ ТЕКСТ ДЛЯ АНАЛИЗА:")
    print(f"«{sample_story}»\n")

    # 3. Запуск 2-проходного нарративного парсера
    success = narrator.parse(sample_story)

    if not success:
        print("❌ Не удалось распарсить нарратив. Проверьте запуск LLM на localhost:8080.")
        return

    # 4. Запуск симуляции аффектов и каузальной цепочки
    narrator.simulate()

    # 5. Ответы на вопросы "Почему?" по графу
    print("\n" + "═" * 60)
    print("❓ ПРОВЕРКА КАУЗАЛЬНОГО ПОИСКА (answer_why)")
    print("═" * 60)
    print(narrator.answer_why("сожрала"))
    print(narrator.answer_why("разозлилась"))

    # 6. Обучение предиктивного ядра (Cortex) на потоке текста нарратива
    print("\n" + "═" * 60)
    print("🧠 ОБУЧЕНИЕ ПРЕДИКТИВНОГО КОРТИКАЛЬНОГО ЯДРА (Семантический поток)")
    print("═" * 60)
    
    import re

    def clean_token(text: str) -> str:
        # Убираем все в скобках, например "рассказала(канарейка)" -> "рассказала"
        return re.sub(r'\(.*?\)', '', text).strip()

    semantic_tokens = []
    for ev in sorted(narrator.events.values(), key=lambda x: x["tick"]):
        actor  = clean_token(narrator._name(ev["actor"]))
        action = clean_token(ev["action"])
        target = clean_token(narrator._name(ev.get("target") or ""))

        # Добавляем актора, только если он не совпадает с предыдущей целью (избегаем дублей)
        if not semantic_tokens or semantic_tokens[-1] != actor:
            if actor:
                semantic_tokens.append(actor)
        if action:
            semantic_tokens.append(action)
        if target and target != "—":
            semantic_tokens.append(target)

    semantic_stream = " ".join(semantic_tokens)
    print(f"Чистый поток для Кортекса: «{semantic_stream}»\n")
    
    cortex.process_stream(semantic_stream, verbose=True)


    # 7. Интерактивный режим Q&A после обучения
    print("\n" + "═" * 60)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ ПРОВЕРКИ ПАМЯТИ")
    print("=" * 60)
    print("Введите слово/действие (например, 'кошка', 'сожрала') или 'exit' для выхода.")

    while True:
        try:
            user_input = input("\n[Липид-2 Q&A] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "выход"]:
                print("\nЗавершение теста. Сеанс сохранен.")
                break

            # Быстрый запрос причин и SDR-ассоциаций
            if "почему" in user_input.lower():
                target = user_input.lower().replace("почему", "").strip()
                print(narrator.answer_why(target))
            else:
                cortex.query(user_input)

        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            break
        except Exception as e:
            print(f"\n[Ошибка цикла]: {e}")

if __name__ == "__main__":
    run_lipid2_narrative_test()
