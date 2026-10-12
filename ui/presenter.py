"""
ui/presenter.py — логика окна без Tkinter.

РАЗДЕЛЕНИЕ СОЗНАТЕЛЬНОЕ
-----------------------
Проверять «окно нарисовалось» дорого и хрупко (нужен X, цвета, шрифты).
Проверять надо другое, и оно как раз в этом файле:

  * журнал не теряет сообщения и не растёт бесконечно;
  * повторный «Учить», пока предыдущий ещё работает, НЕ запускает второй поток:
    два learner'а пишут в один lipid_world.json, и гонка там уже была реальным
    багом (учитель — отдельный процесс);
  * исключение из рабочего потока доходит до пользователя текстом, а не
    «зависшим» окном (прежний worker без try/except убивал поток молча);
  * обратный вопрос мозгу («что такое X?») реально возвращается в мозг и так же
    реально ОТМЕНЯЕТСЯ закрытием окна, а не вешает поток навсегда.

Здесь нет ни одного импорта tkinter — модуль тестируется в headless.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

# Журнал ограничен: сеанс обучения даёт сотни строк статуса, а Tk-виджет с
# десятками тысяч строк тормозит перерисовку. Старые строки не теряются бесследно:
# счётчик сброшенного передаётся в stats(), и «журнал укоротился» — видно.
JOURNAL_LIMIT = 2000


class Journal:
    """
    Буфер журнала с временем и категорией строки.

    Категория (kind) нужна не для красоты: «error» и «ok» обязаны отличаться
    цветом, иначе пользователь читает простыню одинакового серого и не замечает
    единственную важную строку («битовый кэш ОТВЕРГНУТ», «модель недоступна»).
    """

    def __init__(self, limit: int = JOURNAL_LIMIT):
        self._rows: Deque[Tuple[float, str, str]] = deque(maxlen=limit)
        self._lock = threading.Lock()
        self.dropped = 0
        self.limit = limit

    def add(self, text: str, kind: str = "status") -> Optional[Tuple[float, str, str]]:
        text = str(text).rstrip()
        if not text:
            return None
        row = (time.time(), kind, text)
        with self._lock:
            if len(self._rows) == self._rows.maxlen:
                self.dropped += 1
            self._rows.append(row)
        return row

    def rows(self) -> List[Tuple[float, str, str]]:
        with self._lock:
            return list(self._rows)

    def tail(self, n: int = 1) -> List[Tuple[float, str, str]]:
        with self._lock:
            return list(self._rows)[-n:]

    def last_status(self) -> str:
        """Последняя строка НЕ-отчёта — она идёт в индикатор «чем занята»."""
        with self._lock:
            for ts, kind, text in reversed(self._rows):
                if kind in ("status", "error", "ask"):
                    return text
            return ""

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()
            self.dropped = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


class AnswerBridge:
    """
    Мост «мозг спрашивает — человек отвечает».

    Вызов идёт из РАБОЧЕГО потока, поэтому здесь событие и ручная разблокировка:
    Tk-виджет трогать нельзя, а ждать надо.

    Закрытое окно ОБЯЗАНО освободить ожидание. Иначе рабочий поток висит на
    Event.wait() вечно, процесс не завершается, а пользователь видит «окно
    закрылось, а python живой» — классическое тихое зависание.
    """

    def __init__(self, on_question: Callable[[str], None]):
        self._on_question = on_question       # колбэк в UI-поток
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._answer: Optional[str] = None
        self._question: Optional[str] = None
        self._closed = False
        self.pending = False

    def handler(self, prompt: str) -> Optional[str]:
        """Колбэк для core.logchannel.ask_handler(). Блокирует ВЫЗВАВШИЙ поток."""
        if self._closed:
            return None
        with self._lock:
            self._answer = None
            self._question = prompt
            self._event.clear()
            self.pending = True
        self._on_question(prompt)
        # Опрос вместо бесконечного wait: нужен способ проснуться при закрытии.
        while not self._event.wait(0.1):
            if self._closed:
                with self._lock:
                    self.pending = False
                return None
        with self._lock:
            self.pending = False
            return self._answer

    def submit(self, answer: Optional[str]) -> bool:
        """Ответ пользователя. False — вопроса нет или окно закрывается."""
        if self._closed:
            return False
        with self._lock:
            if not self.pending:
                return False
            self._answer = answer
        self._event.set()
        return True

    def current_question(self) -> Optional[str]:
        with self._lock:
            return self._question if self.pending else None

    def close(self) -> None:
        self._closed = True
        self._event.set()


class JobController:
    """
    Один фоновыйjob за раз + гарантия, что ошибка дошла до пользователя.

    ПОЧЕМУ ОДИН: обучение пишет в lipid_world.json. Второй поток обучения
    поверх первого — это гонка «перечитать и слить» (_commit) с самим собой, и
    потерянные знания неотличимы от «таких не было».
    """

    def __init__(self, journal: Journal,
                 on_state: Callable[[bool], None],
                 on_done: Optional[Callable[[], None]] = None):
        self.journal = journal
        self.on_state = on_state
        self.on_done = on_done
        self._lock = threading.Lock()
        self.busy = False
        self.label = "работа"
        self.thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None
        self.finished = 0

    def start(self, work: Callable[[], None], label: str = "работа") -> bool:
        """
        Запустить работу. False — уже занята, и это СКАЗАНО в журнал, а не
        проигнорировано: молча отклонённая кнопка выглядит как «не нажал».
        """
        with self._lock:
            if self.busy:
                self.journal.add(
                    f"уже работаю — дождись окончания «{self.label}» "
                    f"(второй поток пошёл бы писать в тот же lipid_world.json)",
                    "warn")
                return False
            self.busy = True
            self.label = label
            self.last_error = None
        self.on_state(True)
        self.journal.add(f"начал: {label}", "status")

        def runner():
            t0 = time.time()
            try:
                work()
            except Exception as exc:
                # Без этой ветки исключение убивало поток, и пользователь
                # оставался с «работает» до конца жизни окна.
                self.last_error = f"{type(exc).__name__}: {exc}"
                import traceback
                self.journal.add(f"⛔ {label}: {self.last_error}", "error")
                detail = traceback.format_exc(limit=3).strip().splitlines()
                if detail:
                    self.journal.add(f"   {detail[-1]}", "error")
            finally:
                dt = time.time() - t0
                with self._lock:
                    self.busy = False
                    self.finished += 1
                self.journal.add(f"закончил: {label} за {dt:.1f} с"
                                 + ("  (с ошибкой)" if self.last_error else ""),
                                 "ok" if not self.last_error else "error")
                self.on_state(False)
                if self.on_done:
                    try:
                        self.on_done()
                    except Exception as exc:                # не роняем поток
                        self.journal.add(f"⚠️ on_done: {type(exc).__name__}: {exc}",
                                         "warn")

        self.thread = threading.Thread(target=runner, daemon=True, name="l2-job")
        self.thread.start()
        return True


def format_report(rep) -> str:
    """
    LearningReport -> текст для журнала.

    Цифры обязательны: «выучил» без чисел невозможно проверить. Сюда дополнительно
    выводится то, что обычно молчит: доступна ли морфология и был ли Учитель —
    без них «8 фактов» и «0 фактов» выглядят одинаково правдоподобно.
    """
    lines = [rep.as_text()]
    q = getattr(rep, "quality", {}) or {}
    marks = []
    marks.append("морфология: pymorphy3" if q.get("morph")
                 else "морфология: ЭВРИСТИКА (нет pymorphy3 — разбор грубее)")
    marks.append("учитель: подключён" if q.get("teacher")
                 else "учитель: ОТКЛЮЧЁН (неизвестные слова не будут выяснены)")
    lines.append("  " + "   |   ".join(marks))
    return "\n".join(lines)


def summarize_report(rep) -> str:
    """Одна строка-итог для статус-бара."""
    return (f"фактов +{getattr(rep, 'observations_new', 0)}, "
            f"слов {getattr(rep, 'known_words', 0)}/{getattr(rep, 'content_words', 0)}"
            f" знакомо, учитель {getattr(rep, 'teacher_calls', 0)}×")
