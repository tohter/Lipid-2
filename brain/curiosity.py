"""
brain/curiosity.py — КОРТИЗОЛ КАК ФЛАГ ПОИСКА, ДОФАМИН КАК ПРАВО ЗАПИСИ.

ЗАЧЕМ (слова хозяйки, 07.11)
---------------------------
«Пока кортизол высокий, программа должна найти информацию, как связать данные,
либо изучить вопрос до такой степени, чтобы уметь выдать ответ. Тут годятся оба
учителя, и локалка, и интернет. Когда ответ найден, кортизол понижается, и
дофамин может записать данные в базу данных.»

ДО ПРАВКИ ЭТОГО НЕ БЫЛО (замер 07.11)
-------------------------------------
1. Гормоны писались, но НЕ ЧИТАЛИСЬ: ни один if в мозге не зависел от их
   значений. Кортизол горел — поведение не менялось.
2. Учитель писал в lipid_world.json САМ по пути из конструктора, а дофамин
   загорался ПОСЛЕ. Запись опережала разрешение, и снять её было нельзя.

ЧТО ЗДЕСЬ ЖИВЁТ
---------------
HormoneGate  — единственная точка, где значение гормона ПРЕВРАЩАЕТСЯ в
               решение: must_search() («ищи») и dopamine_ok() («пиши»).
               Сам он ничего не ищет и не пишет — только разрешает/запрещает.
Curiosity    — реестр ПРОБЕЛОВ: цель на каждое непонятное слово/несвязанную
               пару, карантин кандидатов и перенос кандидатов во Вселенную
               ТОЛЬКО через шлюз. Цель не закрывается от отказа каналов:
               «искали и не нашли» — это не «вопроса больше нет».

ЗАКОНЫ
------
* Кортизол НЕ снимается неудачей. Снимает его только relief() — найденная
  связь/ответ.
* Права на запись нет без answer_found, какой бы ни был дофамин: число на
  шкале не аргумент, аргумент — закрытый вопрос.
* Кандидат входит в базу, только если ЗАЦЕПИЛСЯ за известное (есть ребро к
  уже знакомому узлу). Иначе это строка, а не знание.
* Состояние (цели, карантин, уровень тревоги) пережидаёт перезапуск: снятый
  при закрытии окна кортизол — забытая недосказанность.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional

from core.records import merge_entity
# СЛУЖБА СВЕДЕНИЯ ФОРМ. Запрет tests/test_lemma_service_contract.py ловит тех,
# кто сводит формы слов молча: без службы merge_entity опирается на эвристику
# по буквам и может не заметить дубль («роса»/«росу»). Любопытство имеет право
# на язык больше, чем ядро (оно в brain/, а не в core/), поэтому приносит
# службу само, а не надеется, что её принесли до него.
from lang.ru.lemma import install as _install_lemma_service

_install_lemma_service()

GATE_FILE = "hormone_gate.json"
CURIOSITY_FILE = "curiosity.json"
STAGING_FILE = "staging_world.json"

#: Кортизол на этом уровне и выше означает «ИЩИ», ниже — «можно отвечать».
#: Выбран не «на глаз»: 128 — середина шкалы 0..255, один всплеск неизвестного
#: (стресс +200) её гарантированно пробивает, а relief() гарантированно
#: опускает ниже (базовый уровень 20).
DEFAULT_SEARCH_THRESHOLD = 128
#: Уровень дофамина, при котором запись разрешена (при answer_found).
DOPAMINE_WRITE_LEVEL = 200
#: Спокойный уровень после успеха. Ноль ставить нельзя: нулевой кортизол —
#: это не «спокойствие», а «перестала реагировать».
CALM_CORTISOL = 20
CALM_DOPAMINE = 255
STRESS_STEP = 200


class HormoneGate:
    """Шлюз: превращает уровни гормонов в два решения — «ищи» и «пиши».

    Работает либо с собственными числами, либо поверх переданного объекта
    гормонов (core.cortex.Hormones), чтобы GUI-шкалы и решения питались ОДНИМИ
    значениями, а не двумя копиями правды.
    """

    def __init__(self, hormones: Any = None,
                 search_threshold: int = DEFAULT_SEARCH_THRESHOLD):
        self._h = hormones
        self.search_threshold = int(search_threshold)
        self._cortisol = 0
        self._dopamine = 0
        self._norepinephrine = 0

    # ── уровни: один источник правды ────────────────────────────────
    def _get(self, name: str) -> int:
        if self._h is not None:
            return int(getattr(self._h, name, 0) or 0)
        return int(getattr(self, f"_{name}", 0))

    def _set(self, name: str, value: int) -> None:
        value = max(0, min(255, int(value)))
        if self._h is not None:
            setattr(self._h, name, value)
        else:
            setattr(self, f"_{name}", value)

    @property
    def cortisol(self) -> int:
        return self._get("cortisol")

    @property
    def dopamine(self) -> int:
        return self._get("dopamine")

    @property
    def norepinephrine(self) -> int:
        return self._get("norepinephrine")

    # ── решения ─────────────────────────────────────────────────────
    def stress(self, amount: int = STRESS_STEP) -> None:
        """Неизвестное: тревога ВВЕРХ, дофамин приземлён (руки заняты поиском)."""
        self._set("cortisol", min(255, self.cortisol + max(0, int(amount))))
        self._set("norepinephrine", min(255, self.norepinephrine + 60))
        if self.dopamine > 40:
            self._set("dopamine", 40)

    def relief(self) -> None:
        """Ответ/связь найдена: кортизол падает, дофамин поднимается.

        Порядок принципиален: без relief() дофамин сам по себе права не даёт
        (dopamine_ok требует спокойного кортизола), поэтому «вырвать» запись
        на пике тревоги нельзя даже вручную.
        """
        self._set("cortisol", CALM_CORTISOL)
        self._set("dopamine", CALM_DOPAMINE)
        self._set("norepinephrine", 20)

    def must_search(self) -> bool:
        """Высокий кортизол = поиск НЕ завершён. Это и есть тот флаг."""
        return self.cortisol >= self.search_threshold

    def dopamine_ok(self, answer_found: bool = True) -> bool:
        """Разрешена ли запись. БЕЗ ответа — никогда, при каких бы ни было числах.

        С ответом — только при спокойном кортизоле и поднятом дофамине: оба
        условия проверяются здесь, а не в месте вызова, чтобы ни один новый
        канал не мог «поставить галочку сам».
        """
        if not answer_found:
            return False
        return (self.cortisol < self.search_threshold
                and self.dopamine >= DOPAMINE_WRITE_LEVEL)

    # ── персистентность ─────────────────────────────────────────────
    def save(self, dirpath: str) -> None:
        os.makedirs(dirpath, exist_ok=True)
        data = {"cortisol": self.cortisol, "dopamine": self.dopamine,
                "norepinephrine": self.norepinephrine,
                "search_threshold": self.search_threshold,
                "saved_at": time.time()}
        with open(os.path.join(dirpath, GATE_FILE), "w",
                  encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def load(self, dirpath: str) -> "HormoneGate":
        path = os.path.join(dirpath, GATE_FILE)
        if not os.path.exists(path):
            return self
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return self                 # битое состояние не должно глушить мозг
        for name in ("cortisol", "dopamine", "norepinephrine"):
            if name in data:
                self._set(name, data[name])
        if "search_threshold" in data:
            self.search_threshold = int(data["search_threshold"])
        return self


class Goal:
    """Один пробел: что не знаю, зачем понадобилось и сколько раз искала."""

    def __init__(self, word: str, why: str = ""):
        self.word = word
        self.why = why
        self.open = True
        self.attempts = 0
        self.last_failure: str = ""
        self.created_at = time.time()
        self.closed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "why": self.why, "open": self.open,
                "attempts": self.attempts, "last_failure": self.last_failure,
                "created_at": self.created_at, "closed_at": self.closed_at}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        g = cls(data.get("word", ""), data.get("why", ""))
        g.open = bool(data.get("open", True))
        g.attempts = int(data.get("attempts", 0))
        g.last_failure = data.get("last_failure", "")
        g.created_at = float(data.get("created_at", 0.0))
        g.closed_at = data.get("closed_at")
        return g


class Curiosity:
    """Реестр пробелов и карантин знаний.

    dirpath — каталог памяти (там же лежит lipid_world.json); свои файлы
    любопытство пишет РЯДОМ и во Вселенную НЕ лезет — в неё переносит только
    promote() и только через разрешение шлюза.
    """

    def __init__(self, dirpath: str, writer: Optional[Callable[[Dict], int]] = None):
        self.dirpath = dirpath
        self.writer = writer            # writer(entities) -> сколько записано
        self._goals: Dict[str, Goal] = {}
        self._staged: List[Dict[str, Any]] = []
        self._load()

    # ── цели ────────────────────────────────────────────────────────
    def note_unknown(self, word: str, why: str = "") -> Goal:
        """Пробел обнаружен. Повтор того же слова НЕ удваивает цель.

        Цель открывается молча и НЕ спрашивает учителей сама: решение «когда
        и у кого спрашивать» остаётся у мозга, иначе любопытство превратится
        во второй канал обучения, конкурирующий с первым.
        """
        word = (word or "").strip().lower()
        if not word:
            raise ValueError("имя цели пустое")
        goal = self._goals.get(word)
        if goal is None:
            goal = Goal(word, why)
            self._goals[word] = goal
        elif not goal.open:
            goal.open = True            # закрытая цель ожила: слово снова
            goal.closed_at = None       # встретилось и снова непонято
        if why and not goal.why:
            goal.why = why
        self._save()
        return goal

    def open_goals(self) -> List[Goal]:
        return [g for g in self._goals.values() if g.open]

    def failed(self, goal: Goal, reason: str = "") -> Goal:
        """Каналы промолчали. Цель ЖИВЁТ, тревога НЕ снимается.

        Это место, где раньше терялся вопрос: «не нашли» закрывало цель, и
        система с пустой головой выглядела спокойно.
        """
        goal.attempts += 1
        goal.last_failure = reason or "каналы не дали знания"
        goal.open = True
        self._save()
        return goal

    def closed(self, goal: Goal) -> Goal:
        goal.open = False
        goal.closed_at = time.time()
        self._save()
        return goal

    # ── карантин кандидатов ─────────────────────────────────────────
    # ── карантинный ФАЙЛ для учителей ───────────────────────────────
    def live_world_path(self) -> str:
        return os.path.join(self.dirpath, "lipid_world.json")

    def staging_path(self) -> str:
        return os.path.join(self.dirpath, STAGING_FILE)

    def begin_search(self) -> str:
        """Подготовить каналу файл, в котором МОЖНО учиться.

        Пока поиск не завершён, учитель не имеет права трогать живую базу
        (замер: вопрос про дождь изменил lipid_world.json на 3 сущности и не
        дал ответа). Здесь учитель получает КОПИЮ живой базы: он видит уже
        известное и не переучивает его, а пишет в копию. Живой файл от этого
        не меняется ни на байт.

        Копия, а не пустой файл — иначе учитель потерял бы контекст и начал
        заново описывать то, что уже есть (та же дыра, что с перезаписью
        сущности вырожденным derived).
        """
        os.makedirs(self.dirpath, exist_ok=True)
        live, stage = self.live_world_path(), self.staging_path()
        if os.path.exists(stage):
            # Рабочее окно УЖЕ открыто. Перезаливать его копией живой базы
            # нельзя: затёрлись бы данные, которые канал принёс в прошлой
            # неудачной попытке (провал поиска не имеет права терять их).
            return stage
        if os.path.exists(live):
            shutil.copyfile(live, stage)
        else:
            with open(stage, "w", encoding="utf-8") as fh:
                json.dump({"entities": {}, "observations": {},
                           "unknown_words": {}}, fh, ensure_ascii=False)
        return stage

    def collect_from_staging(self) -> List[str]:
        """Забрать из карантина ТОЛЬКО новое и убрать карточку-копию.

        Переносится то, чего в живой базе не было вовсе, или что стало БОГАЧЕ
        (учитель добавил ребро к уже знакомому слову). Иначе копия всей
        Вселенной переехала бы в кандидаты, и promote затащил бы в файл чужие
        знания под моей меткой.
        """
        stage = self.staging_path()
        if not os.path.exists(stage):
            return []
        try:
            with open(stage, encoding="utf-8") as fh:
                new_world = json.load(fh)
        except Exception:
            return []
        live_entities: Dict[str, Any] = {}
        try:
            with open(self.live_world_path(), encoding="utf-8") as fh:
                live_entities = json.load(fh).get("entities") or {}
        except Exception:
            live_entities = {}
        taken: List[str] = []
        fresh: Dict[str, Any] = {}
        already = {str(i.get("name", "")).strip().lower() for i in self._staged}
        for key, rec in ((new_world.get("entities") or {}).items()):
            if not isinstance(rec, dict):
                continue
            if key in already:
                # Кандидат уже лежит в карантине (собирали в прошлой попытке).
                # Второй раз ставить его нельзя: promote перенёс бы дубль.
                continue
            old = live_entities.get(key)
            if not isinstance(old, dict):
                fresh[key] = rec
                taken.append(key)
                continue
            gained = {}
            for field in ("hierarchy", "properties", "actions",
                          "delta_properties"):
                now = [str(x) for x in (rec.get(field) or [])]
                was = [str(x) for x in (old.get(field) or [])]
                if now and now != was:
                    gained[field] = rec.get(field) or []
            richer = bool(gained) or (bool(rec.get("definition")) and
                                      rec["definition"] != old.get("definition"))
            if richer:
                merged = merge_entity(old, rec)
                merged.update(gained)
                fresh[key] = merged
                taken.append(key)
        if fresh:
            self.stage(fresh, source="staging")
        # Карточку-копию НЕ удаляем: в ней остаются observations (свидетельства
        # канала) и то, что не подошло по имени. Живая база не изменилась, а
        # данные живы. Окно закрывает clear_staging().
        return taken

    def clear_staging(self) -> bool:
        """Закрыть окно поиска: карточка-копия больше не нужна."""
        try:
            os.remove(self.staging_path())
            return True
        except OSError:
            return False

    def stage(self, entities: Dict[str, Any], source: str = "") -> int:
        """Кандидат от учителя/наблюдателя ДО разрешения записи.

        Хранится НЕ в Вселенной: пока дофамин не разрешил, эти данные не имеют
        права влиять на ответы — иначе «гипотеза» и «знание» сливаются, и
        проверить источник становится нечем.
        """
        count = 0
        for name, rec in (entities or {}).items():
            name = str(name).strip().lower()
            if not name or not isinstance(rec, dict):
                continue
            self._staged.append({"name": name, "record": rec, "source": source,
                                 "staged_at": time.time()})
            count += 1
        self._save()
        return count

    def staged(self) -> List[Dict[str, Any]]:
        return list(self._staged)

    def _hooks(self, record: Dict[str, Any], knower: Any) -> List[str]:
        """За что кандидат цепляется в уже известном графе."""
        hooks: List[str] = []
        if knower is not None:
            names = set()
            for field in ("hierarchy", "properties", "actions", "parts"):
                for value in record.get(field) or []:
                    v = str(value).strip().lower()
                    if not v:
                        continue
                    if knower.knows(v):
                        names.add(v)
            for edge in record.get("relates_to") or []:
                if isinstance(edge, dict):
                    t = str(edge.get("to", "")).strip().lower()
                    if t and knower.knows(t):
                        names.add(t)
            hooks = sorted(names)
        return hooks

    def promote(self, knowledge: Callable[[], Any], gate: HormoneGate,
                answer_found: bool = True) -> int:
        """Перенос кандидатов во Вселенную. Три замка на одном ключе.

        knowledge() — функция, дающая текущее знание (для проверки зацепки);
        записывает writer, переданный в конструктор. Возвращает число вошедших.

        Замки:
          1. gate.dopamine_ok(answer_found) — без ответа не пишем НИКОГДА;
          2. зацепка за известное — без неё это строка, а не знание;
          3. кандидат, не прошедший зацепку, ОСТАЁТСЯ в карантине (его ещё
             можно закрыть позже), а не сжигается.
        """
        if not gate.dopamine_ok(answer_found=answer_found):
            return 0
        if not self._staged or self.writer is None:
            return 0
        knower = knowledge() if knowledge is not None else None
        keep: List[Dict[str, Any]] = []
        ready: Dict[str, Any] = {}
        for item in self._staged:
            hooks = self._hooks(item["record"], knower)
            if hooks:
                ready[item["name"]] = item["record"]
            else:
                keep.append(item)
        if not ready:
            return 0
        moved = self.writer(ready) or 0
        # Ушёл только то, что реально ушло писателю; при отказе писателя
        # карантин сохраняется целиком (лучше повторная попытка, чем потеря).
        if moved:
            self._staged = keep
            self._save()
        return moved

    # ── хранение своего состояния ───────────────────────────────────
    def _path(self) -> str:
        return os.path.join(self.dirpath, CURIOSITY_FILE)

    def _save(self) -> None:
        os.makedirs(self.dirpath, exist_ok=True)
        data = {"goals": [g.to_dict() for g in self._goals.values()],
                "staged": self._staged}
        with open(self._path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)

    def _load(self) -> None:
        path = self._path()
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        for raw in data.get("goals", []):
            g = Goal.from_dict(raw)
            if g.word:
                self._goals[g.word] = g
        self._staged = list(data.get("staged", []))
