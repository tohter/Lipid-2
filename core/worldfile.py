"""
core/worldfile.py — единая точка чтения/записи lipid_world.json.

ПОЧЕМУ ОБЩИЙ МОДУЛЬ, А НЕ «копипаст на месте»
----------------------------------------------
Файл Вселенной пишут НЕЗВИМЫЕ друг для друга места: brain.lipid2_brain,
brain.learner (через _commit), teachers/teacher1 (Учитель-агент),
teachers/teacher_wiki. Каждое имело СВОЮ копию чтения вида

    try: return json.load(f)
    except Exception: pass
    return {"entities": {}, ...}

и СВОЮ запись `open(path, "w")`. Два разных способа потерять знание:

1. ЧТЕНИЕ. Битый/недописанный JSON трактовался как «файла нет» -> пустой мир.
   Дальнейший merge (перечитать и слить) к пустому миру = ЗАМЕНИТЬ содержимое
   своим снимком. То есть механизм, защищающий от гонки writer'ов, при битом
   файле сам становился механизмом стирания памяти.

2. ЗАПИСЬ. Открытие на "w" уссекает файл немедленно; если процесс умрёт в
   момент json.dump (закрытие окна во время обучения — самый вероятный
   сценарий, ctrl+C, OOM, перезагрузка), на диске останется ОБОРВАННЫЙ JSON.
   Через пару секунд json.dump запишет остаток, и следующий запуск получит
   либо битый файл, либо половину знаний. atomic_replace закрывает это:
   читатель видит либо старую, либо новую версию, но никогда — половину.

Оба случая обязаны ГОВОРИТЬ: «молча вернули пустой мир» — тихая деградация,
а тихая деградация в этом проекте считается багом (memory.txt).

Модуль — чистый stdlib: он нужен и учителям, которым torch запрещён.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from typing import Any, Dict, Optional

# Разные писатели живут в разных ПОТОКАХ одного процесса (GUI-обучение и
# Учитель-агент). Операционная система атомарно переименовывает файл, но два
# потока могут пересечься внутри «прочитал -> изменил -> записал», поэтому
# внутри процесса нужен ещё и мьютекс.
_LOCK = threading.RLock()

EMPTY_WORLD: Dict[str, Any] = {"entities": {}, "relations": [], "observations": {},
                               "unknown_words": {}}


def _complain(text: str) -> None:
    sys.stderr.write(text if text.endswith("\n") else text + "\n")
    sys.stderr.flush()


def default_memory_dir(memory_dir: str = None, filename: str = "lipid_world.json"):
    """
    Каталог памяти: явный аргумент -> LIPID2_MEMORY_DIR -> <проект>/data/memory.

    Переменная окружения обязана читаться в ОДНОМ месте. Её отсутствие в brain
    стоило того, что сквозной прогон GUI (с LIPID2_MEMORY_DIR=/tmp/...) писал в
        РАБОЧУЮ память: «учебные» пчела и цветок осели в настоящем
        lipid_world.json, потому что мозг переменную не знал, а тест — знал.
    Возвращает (каталог, путь_к_файлу) — путь считается сразу, чтобы у всех
    писателей он был буквально один и тот же.
    """
    if not memory_dir:
        memory_dir = os.environ.get("LIPID2_MEMORY_DIR")
    if not memory_dir:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        memory_dir = os.path.join(root, "data", "memory")
    os.makedirs(memory_dir, exist_ok=True)
    return memory_dir, os.path.join(memory_dir, filename)


def atomic_write_json(path: str, data: dict) -> None:
    """
    Записать JSON так, чтобы читатель не мог увидеть полупустой файл.

    Схема: временный файл в ТОМ ЖЕ каталоге -> flush + fsync -> os.replace.
    os.replace атомарен в пределах одной файловой системы; отсюда требование
    «тот же каталог» (в /tmp в другой FS — уже не атомарно, а тихо медленнее).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    with _LOCK:
        fd, tmp = tempfile.mkstemp(prefix=".l2world-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            # неудачная запись не должна оставлять мусор в каталоге данных
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def read_world(path: str, *, loud: bool = True) -> Dict[str, Any]:
    """
    Прочитать Вселенную. Отсутствие файла — норма; битый файл — АНОМАЛИЯ.

    Возврат пустого мира при НЕЧИТАЕМОМ файле сохранён (падение приложения из
    читающего пути хуже), но сопровождается явным сообщением в stderr и
    сохранением копии битого файла рядом: «разобраться потом» возможно только
    пока битые байты не перезаписаны.
    """
    if not os.path.exists(path):
        return {"entities": {}, "relations": [], "observations": {},
                "unknown_words": {}}
    try:
        with _LOCK:
            with open(path, "r", encoding="utf-8") as fh:
                world = json.load(fh)
    except Exception as exc:
        if loud:
            backup = _preserve_broken(path)
            _complain(
                f"[world] ⛔ {os.path.basename(path)} НЕ ЧИТАЕТСЯ: "
                f"{type(exc).__name__}: {exc}\n"
                + (f"[world]    битая копия сохранена: {backup}\n" if backup else
                   "[world]    сохранить копию не удалось\n")
                + "[world]    работаю с ПУСТЫМ миром: любое сохранение сейчас "
                  "перезапишет Вселенную. Разберитесь с файлом до обучения.")
        return {"entities": {}, "relations": [], "observations": {},
                "unknown_words": {}}
    if not isinstance(world, dict):
        if loud:
            _complain(f"[world] ⛔ {os.path.basename(path)} содержит {type(world).__name__}, "
                      "а не объект JSON — работаю с пустым миром")
        return {"entities": {}, "relations": [], "observations": {},
                "unknown_words": {}}
    for key, default in EMPTY_WORLD.items():
        world.setdefault(key, dict(default) if isinstance(default, dict) else list(default))
    return world


def write_world(path: str, world: dict) -> None:
    """Атомарная запись (см. atomic_write_json)."""
    atomic_write_json(path, world)


def _preserve_broken(path: str) -> Optional[str]:
    """Скопировать нечитаемый файл, чтобы его можно было починить руками."""
    try:
        base, ext = os.path.splitext(path)
        target = f"{base}.broken{ext}"
        n = 1
        while os.path.exists(target):
            target = f"{base}.broken{n}{ext}"
            n += 1
        with open(path, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
        return target
    except OSError:
        return None
