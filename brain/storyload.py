"""
brain/storyload.py — витрина без имён рассказов в коде.

Зачем отдельный модуль. В окне никогда не было симулятора: вкладка
«Симуляция» ходила в ядро («что такое кошка»), а «почему кошка набросилась»
спросить было нечерез что. Чтобы это починить, нужны три вещи: найти миры на
диске, подобрать для мира класс симулятора и спросить. Все три — обычная
логика, Tk в ней не участвует, поэтому она здесь и поэтому проверяется без
окна.

ЧЕГО ЗДЕСЬ СОЗНАТЕЛЬНО НЕТ: списка рассказов. Ни одного `if имя == "fire"`.
Класс симулятора берётся из схемы мира (handler), а набор миров — из
каталога. Новый рассказ = новый файл, код не меняется.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_MAX_JSON_BYTES = 8 * 1024 * 1024


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def search_dirs(extra: Optional[List[str]] = None) -> List[str]:
    """Каталоги, где лежат миры. По умолчанию — experiments/ проекта."""
    dirs = [os.path.join(project_root(), "experiments")]
    for d in (extra or []):
        if d and d not in dirs:
            dirs.append(os.path.abspath(os.path.expanduser(d)))
    return [d for d in dirs if os.path.isdir(d)]


def _load_json(path: str) -> Optional[dict]:
    """Чтение с двумя guardrail: размер и «это действительно объект»."""
    try:
        if os.path.getsize(path) > _MAX_JSON_BYTES:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


_QUOTED = re.compile("\u00ab([^\u00bb]{2,60})\u00bb")


def _title(data: dict, path: str) -> str:
    """
    Заголовок для СПИСКА, то есть для чтения человеком.

    Порядок: явное имя -> название в кавычках из _comment (миры пишут
    «Эксперимент „дождь и пожар“…») -> имя файла без суффикса. Имя файла
    остаётся последним прибежищем, а не первым: пользователю «cat_canary»
    говорит меньше, чем «кошка и канарейка».
    """
    name = data.get("name") or data.get("scheme") or ""
    if name:
        return str(name)
    quoted = _QUOTED.search(str(data.get("_comment") or ""))
    if quoted:
        return quoted.group(1)
    base = os.path.basename(path)
    for suffix in ("_world.json", "_story.json", ".json"):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


# handler из схемы -> (модуль, класс, КАК загружать рассказ). Третий элемент
# обязателен: миры хранят динамику по-разному — «огонь» в секции events, а
# «социум» в секции script, и у script свой порядок вызовов (ожидания ДО
# сценария, задача 34). Делать это разветвлением по имени рассказа было бы
# возвратом к той самой ошибке, которую исправляет задача 35A.
_HANDLERS: Dict[str, tuple] = {
    "why_stopped": ("brain.simulator", "Simulator", "events"),
    "why_acted": ("brain.social_sim", "SocialSimulator", "script"),
}


def handler_of(world: dict) -> Optional[str]:
    """
    Какой обработчик миру нужен — из ДАННЫХ мира (задача 35A).

    Мир с двумя схемами берёт самую старшую по priority: вопрос всё равно
    разойдётся по стоп-глаголам внутри choose_scheme, а класс симулятора
    обязан быть один и тот же на оба вопроса.
    """
    from core.schemes import load_schemes
    for _name, scheme in sorted(load_schemes(world).items(),
                                key=lambda kv: -int(getattr(kv[1], "priority", 0)
                                                    or 0)):
        h = str(getattr(scheme, "handler", "") or "").strip()
        if h:
            return h
    return None


@dataclass
class StorySource:
    """Один рассказ на диске: мир, события, ожидания — всё парами."""
    key: str
    title: str
    world_path: str
    story_path: Optional[str] = None
    scheme_names: List[str] = field(default_factory=list)
    entities: int = 0
    handlers: List[str] = field(default_factory=list)
    warning: str = ""

    def label(self) -> str:
        base = f"{self.title}  [{', '.join(self.scheme_names) or 'без схем'}]"
        return base + (f"  ⚠ {self.warning}" if self.warning else "")


def discover(extra_dirs: Optional[List[str]] = None) -> List[StorySource]:
    """
    Найти миры по каталогам. Файлом мира считается тот, у которого есть
    entities или schemes; события ищутся по соседству.

    Миры БЕЗ схем в список попадают, но с предупреждением: иначе пользователь
    видит пустой список и не понимает, что файл он нашёл, а схема не описана.
    """
    out: List[StorySource] = []
    seen: set = set()
    for d in search_dirs(extra_dirs):
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(d, fname)
            data = _load_json(path)
            if data is None:
                continue
            if not (data.get("entities") or data.get("schemes")):
                continue
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            key = fname[:-5]
            # Схемы читаются ШТАТНЫМ загрузчиком, а не перебором секции:
            # "schemes" — это словарь {имя: спецификация}, и наивный
            # `for s in schemes` отдаёт КЛЮЧИ-СТРОКИ. Первый вариант этого
            # модуля так и сделал и доложил «миров нет, схем нет» про два
            # мира, в которых схемы есть.
            from core.schemes import load_schemes
            schemes = load_schemes(data)
            handlers = sorted({getattr(s, "handler", "") for s in schemes.values()
                               if getattr(s, "handler", "")})
            story_path = _find_story(d, key, data)
            warning = ""
            if not schemes:
                warning = "в мире нет schemes — спрашивать будет нечего"
            out.append(StorySource(
                key=key, title=_title(data, path), world_path=path,
                story_path=story_path,
                scheme_names=sorted(schemes),
                entities=len(data.get("entities") or {}),
                handlers=handlers, warning=warning))
    return out


def _find_story(directory: str, world_key: str, data: dict) -> Optional[str]:
    """
    Файл событий к миру. Порядок поиска: явная договорённость *_world.json ->
    *_story.json, потом поле "story" внутри мира, потом любой JSON с
    тем же началом имени.
    """
    if world_key.endswith("_world"):
        cand = os.path.join(directory, world_key[:-6] + "_story.json")
        if os.path.isfile(cand):
            return cand
    named = str(data.get("story") or "").strip()
    if named:
        for cand in (named, os.path.join(directory, named),
                     os.path.join(directory, os.path.basename(named))):
            if os.path.isfile(cand):
                return cand
    stem = world_key.split("_")[0]
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json") and stem in fname and fname != world_key:
            data2 = _load_json(os.path.join(directory, fname))
            if data2 and data2.get("events"):
                return os.path.join(directory, fname)
    return None


def events_of(src: StorySource) -> List[dict]:
    if not src.story_path:
        return []
    data = _load_json(src.story_path) or {}
    evs = data.get("events")
    return [e for e in evs if isinstance(e, dict)] if isinstance(evs, list) else []


def build_simulator(src: StorySource,
                    params: Optional[Dict[str, int]] = None):
    """
    Собрать симулятор выбранного рассказа.

    Класс берётся из handler в схеме. Неизвестный handler — НЕ молчаливый
    откат к базовому симулятору: базовый не умеет ни надежд, ни порога, и
    пользователь получил бы «не знаю» вместо «не умею». Возвращаю (None,
    причина).
    """
    world = _load_json(src.world_path)
    if world is None:
        return None, f"мир не читается: {src.world_path}"
    handler = handler_of(world)
    if not handler:
        return None, ("в мире нет schemes[*].handler — я не знаю, КАКИМ "
                      "симулятором его считать")
    spec = _HANDLERS.get(handler)
    if spec is None:
        return None, (f"схема требует обработчик «{handler}», а я умею только "
                      f"{', '.join(sorted(_HANDLERS))}")
    module_name, cls_name, how = spec
    import importlib
    cls = getattr(importlib.import_module(module_name), cls_name)

    if how == "script":
        # СОЦИАЛЬНЫЙ порядок вызовов. Ожидания читаются ДО сценария: просьба
        # обязана родиться раньше нарушения, иначе её нарушитель окажется
        # «родившимся» до самой просьбы и нарушение молча не засчитается.
        sim = cls(world)
        if hasattr(sim, "load_expectations"):
            sim.load_expectations(world)
        script = []
        story = _load_json(src.story_path) if src.story_path else None
        if story:
            script = [s for s in (story.get("script") or [])
                      if isinstance(s, dict)]
        if not script:
            script = [s for s in (world.get("script") or [])
                      if isinstance(s, dict)]
        if not script:
            return sim, ("у социального мира нет секции script — ни просьбы, "
                         "ни нарушения, спрашивать не про что")
        # ЗНАЧЕНИЯ плейсхолдеров. Рассказ пишет «n": "$between_ticks», а число
        # приходит снаружи (так тесты варят выдержку, не трогая данные). Если
        # подстановки нет, берём дефолт из social_sim, а не придумываем своё:
        # иначе окно и тесты симулировали бы разные рассказы молча.
        from brain.social_sim import RIPEN_TICKS
        defaults = {"$between_ticks": RIPEN_TICKS}
        defaults.update(params or {})
        for d in script:
            if isinstance(d.get("n"), str) and d["n"].startswith("$"):
                if d["n"] not in defaults:
                    return sim, (f"в рассказе плейсхолдер {d['n']}, а значения "
                                 f"для него нет")
                d["n"] = int(defaults[d["n"]])
        sim.load_script(script)
        return sim, ""

    if how == "events":
        sim = cls(world)
        if src.story_path:
            n = sim.load_story(src.story_path)
            if not n:
                return sim, (f"файл рассказа {os.path.basename(src.story_path)} "
                             f"не дал ни одного события")
        elif world.get("events"):
            sim.load_events([e for e in world["events"] if isinstance(e, dict)])
        else:
            return sim, ("у мира нет ни файла рассказа, ни секции events — "
                         "динамики нет, спрашивать не про что")
        return sim, ""

    return None, f"неизвестный способ загрузки рассказа: {how!r}"
