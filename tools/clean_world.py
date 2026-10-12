"""
tools/clean_world.py — привести lipid_world.json к правилам записи.

Что и почему чинит (см. core/records.py, meaningful_categories):
  1. из hierarchy выбрасываются формы самого концепта и служебные категории
     Википедии — источник ответов вида «экосистема — это экосистемы»;
  2. пустые записи (только название) НЕ удаляются молча: они переводятся в
     unknown_words, потому что «искали и нашли дырку» — это факт о НЕЗНАНИИ,
     а unknown_words — ровно то место, где такие факты живут.

Работает идемпотентно, пишет атомарно, исходник копирует рядом. Запуск:
    .venv/bin/python tools/clean_world.py            # только показать
    .venv/bin/python tools/clean_world.py --write    # показать и записать
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.records import is_empty_entity, meaningful_categories
from core.worldfile import read_world, write_world

MEMORY_DIR = os.environ.get("LIPID2_MEMORY_DIR",
                            os.path.join(ROOT, "data", "memory"))


def keeps_content_without_definition(rec: dict) -> bool:
    """Останется ли в записи что-нибудь после удаления обрывка.

    Если нет — удалять нечего: пустая запись и так отвечает «знания нет»,
    а выкидывать запись целиком миграция не имеет права (потеря provenance).
    """
    return not is_empty_entity({k: v for k, v in rec.items()
                                if k != "definition"})


def plan(world: dict) -> tuple:
    """Возвращает (исправленный мир, список изменений). Сам мир НЕ меняет."""
    changes = []
    out = {"entities": {}, "observations": dict(world.get("observations") or {}),
           "unknown_words": dict(world.get("unknown_words") or {}),
           "relations": list(world.get("relations") or [])}
    for name, rec in (world.get("entities") or {}).items():
        if not isinstance(rec, dict):
            changes.append(f"{name}: запись не-объект -> unknown_words")
            out["unknown_words"][name] = {"reason": "битая запись",
                                          "source": "clean_world"}
            continue
        new = dict(rec)

        # Шаг 1. definition-обрывок. ЗАМЕРЕНО в живом файле: у «пчела»
        # definition = «Пчёлы (лат» — хвост, который новый _first_sentence
        # выдать уже не может (он следит за скобками), но в файле он пережил
        # правку: запись определений появилась позже самих определений.
        # Срезать его надо ДО проверки пустоты: огрызок — не знание, и запись
        # из одного огрыска обязана уехать в unknown_words, а не притворяться
        # ответом.
        #
        # КРИТЕРИЙ ПО ЗАМЕРУ (реальный lipid_world.json, 3 определения):
        #   «Пчёлы (лат»                     10 симв, скобка не закрыта  -> мусор
        #   «Термин ввод обозначает ...»     122 симв                     -> знание
        #   «Экосистема, или ... систему»    366 симв                     -> знание
        # Требование «кончаться точкой» краснит два последних: определения
        # вырезаны по первому предложению, и точки в конце может не быть.
        # Значит: незакрытая скобка ИЛИ короче 15 символов (тот же порог, что
        # в _definition_phrase — короче не бывает фразой).
        d = str(new.get("definition") or "")
        if d and (d.count("(") != d.count(")") or len(d.strip()) < 15):
            changes.append(f"{name}: definition-обрывок {d[:40]!r} -> убрать "
                           f"(ветхая вырезка; лучше «знания нет», чем огрызок)")
            new.pop("definition", None)

        # Шаг 2. тавтологичные и служебные категории.
        hier = meaningful_categories(new.get("concept") or name, new.get("hierarchy"))
        if hier != (rec.get("hierarchy") or []):
            removed = [c for c in (rec.get("hierarchy") or []) if c not in hier]
            changes.append(f"{name}: иерархия {rec.get('hierarchy')} -> {hier or '[]'} "
                           f"(убрано мусора: {removed})")
            if hier:
                new["hierarchy"] = hier
            else:
                new.pop("hierarchy", None)

        # Шаг 3. пустая запись (после зачистки, а не до неё).
        if is_empty_entity(new):
            changes.append(f"{name}: запись пуста (только название) -> unknown_words")
            known = out["unknown_words"].get(name) or {}
            out["unknown_words"][name] = {
                "reason": "знания нет: все каналы молчали",
                "seen_as": known.get("seen_as", name),
                "moved_from": "entities",
            }
            continue
        out["entities"][name] = new

    # Шаг 4. Противоречие «известно и неизвестно одновременно» (ЗАМЕРЕНО в
    # живом lipid_world.json: «пчела» и «ввод» лежали в обоих списках).
    # Побеждает запись сущности: в ней определение и свойства, а unknown_words —
    # только счётчик обращений. Оставлять оба нельзя: панель памяти показывает
    # «сущностей N · неизвестных M», и обе цифры оказываются про одно слово.
    # Пустая запись сюда уже не доходит (шаг 3), поэтому проверка простая.
    for key in list(out["unknown_words"]):
        ent = out["entities"].get(key)
        if isinstance(ent, dict) and not is_empty_entity(ent):
            changes.append(f"{key}: есть запись со знанием -> убрать из "
                           f"unknown_words (иначе файл враньё про одно слово)")
            del out["unknown_words"][key]

    # Шаг 5. Неверный диагноз на ГЛАГОЛЕ. ЗАМЕРЕНО в живом lipid_world.json:
    # «гасить» стоит в unknown_words со статусом asked_empty. Этот статус
    # значит «мы спросили честным вопросом и не нашли» — но спрашивали его как
    # СУЩНОСТЬ, то есть вопроса как такого не было. Оставить это = навсегда
    # запретить слово к изучению из-за категориальной ошибки прежнего кода.
    #
    # Здесь НЕТ контекста предложения (в отличие от Learner, который видит
    # разбора), поэтому распознавание — по морфологии и словарику профилей.
    from core.actions import looks_like_verb
    for key, u in list(out["unknown_words"].items()):
        if not isinstance(u, dict) or u.get("status") != "asked_empty":
            continue
        if not looks_like_verb(key):
            continue
        changes.append(f"{key}: asked_empty на глаголе выставлен НЕВЕРНО "
                       f"(искали сущность, а это действие) -> снять статус, "
                       f"слово возвращается в очередь уроков")
        rest = {k: v for k, v in u.items() if k != "status"}
        rest["status"] = "not_asked"
        rest["status_note"] = ("диагноз asked_empty снят clean_world: слово "
                               "спрашивали как сущность, а оно действие")
        out["unknown_words"][key] = rest

    # Шаг 6. Записи СТАРЫХ уроков без поля status. Поведенчески они и так
    # трактуются как «не спрашивали» (статус не равен asked_empty -> спрашиваем),
    # но файл при этом НЕ самоописан: GUI и любой читатель видит пустоту там,
    # где должен быть выверенный факт. ЗАМЕРЕНО в живой памяти: 6 из 7 записей
    # ('берег', 'рыбак', 'сеть', 'чинить', 'опылять', 'цветок') рождения поля
    # status не застали.
    for key, u in out["unknown_words"].items():
        if not isinstance(u, dict) or u.get("status"):
            continue
        changes.append(f"{key}: у записи нет status -> not_asked (уроки до "
                       f"появления поля; НЕ спрашивали, значит учить можно)")
        u["status"] = "not_asked"
        u["status_note"] = "статус дописан clean_world: запись старше поля status"
    return out, changes


def main(argv):
    do_write = "--write" in argv
    # Путь обязан читаться и из аргумента командной строки: без этого «почисти вот этот
    # файл» молча чистит ДРУГОЙ (тот, что указан в env), а оператор видит
    # «мусора не найдено» про совершенно другую память. ЗАМЕРЕНО: вызов
    # `clean_world.py /tmp/x/lipid_world.json --show` показал цифры рабочей
    # памяти Вселенной.
    args = [a for a in argv if not a.startswith("--")]
    if len(args) > 1:
        print(f"⚠️ ожидается ОДИН путь, передано {len(args)}: {args} — "
              "беру первый", file=sys.stderr, flush=True)
    path = (args[0] if args
            else os.environ.get("LIPID2_WORLD_FILE")
            or os.path.join(MEMORY_DIR, "lipid_world.json"))
    world = read_world(path)
    clean, changes = plan(world)
    print(f"файл: {path}")
    if os.environ.get("LIPID2_MEMORY_DIR"):
        print("КАТАЛОГ ПАМЯТИ ВЗЯТ ИЗ LIPID2_MEMORY_DIR — это НЕ рабочая память")
    print(f"сущностей: {len(world.get('entities') or {})} -> {len(clean['entities'])}")
    print(f"unknown:   {len(world.get('unknown_words') or {})} -> "
          f"{len(clean['unknown_words'])}")
    for c in changes:
        print("  · " + c)
    if not changes:
        print("мусора не найдено")
        return 0
    if not do_write:
        print("\nэто показ; для записи добавить --write")
        return 0
    backup = path + ".pre-clean"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
        print(f"исходник сохранён: {backup}")
    write_world(path, clean)
    print(f"записано атомарно: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
