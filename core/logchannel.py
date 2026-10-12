"""
core/logchannel.py — канал статуса: чем система занята ПРЯМО СЕЙЧАС.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ
---------------------
Долгое время единственным выходом обучающего контура был stdout. Для терминала
этого хватало, но GUI не видит stdout: пользователь нажимал «Учить» и получал
неподвижное окно на 30-60 секунд (Учитель — это локальная LLM, шаг агента
длится секунды, шагов до шести). Молчаливое ожидание неотличимо от зависания,
а зависание без ошибок — это тихая деградация интерфейса.

Здесь ОДИН узел, в который ключевые точки кода пишут человекочитаемые шаги:

    status("изучаю слово «джойстик»...")

Сообщение уходит одновременно:
  * всем подписчикам (GUI: журнал + статус-строка) — через их колбэки;
  * в stdout, если LIPID2_STATUS != "0" (терминальный запуск видит то же).

ДВУСТОРОННОСТЬ (request/answer)
-------------------------------
Третий канал знания — «спросить пользователя». Мозг умел это делать только
на словах: без подключённого колбэка в лог уходило «Нет колбэка к пользователю
(GUI не подключён)», а пользователь не видел НИЧЕГО. Молчаливый вопрос — это
тихая деградация того же класса, что и неработающий канал записи учителя.

request(prompt) бросает вопрос слушателю и ЖДЁТ ответа (рабочий поток GUI и
так отдельный, блокировать его можно). Если слушатель вопросов не принимает —
возвращается None немедленно, и вызывающий обязан отнестись к этому как к
«знания нет», а не как к успеху.

ПОЧЕМУ КОЛБЭКИ, А НЕ ОЧЕРЕДЬ ЗДЕСЬ
----------------------------------
Очередь (queue.Queue) — забота слушателя: у GUI свой цикл Tkinter, он сам
опрашивает очередь в удобном ему темпе. Здесь только доставка. Поток, ПИШУЩИЙ
в канал, не имеет права умереть из-за слушателя: подписчик, бросивший
исключение, отключается с явным сообщением в stderr, но отправитель
(обучение, учитель, мозг) продолжает работу. Молча отключать слушателя тоже
нельзя — «журнал перестал писаться» выглядит как «процесс закончился».

ЗАГЛУШКА — НЕ ТИХАЯ. Без подписчиков и с LIPID2_STATUS=0 сообщения никуда не
попадают; это осознанный режим тестов, он включается ЯВНО и только переменной
окружения.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Callable, Dict, List, Optional

_LOCK = threading.RLock()
_SUBSCRIBERS: Dict[int, Callable[[str], None]] = {}
_NEXT_ID = 1
# Подписчики, умершие на нашей стороне, — чтобы «журнал замолчал» было видно.
_DROPPED: List[str] = []

# Обратный канал: слушатель, принимающий вопросы мозга.
_QUESTION_HANDLER: Optional[Callable[[str], Optional[str]]] = None

PREFIX = "[l2]"


def _stdout_enabled() -> bool:
    return os.environ.get("LIPID2_STATUS", "1").strip() != "0"


def subscribe(callback: Callable[[str], None]) -> int:
    """Подписаться на поток статуса. Возвращает id для unsubscribe()."""
    global _NEXT_ID
    with _LOCK:
        sid = _NEXT_ID
        _NEXT_ID += 1
        _SUBSCRIBERS[sid] = callback
    return sid


def unsubscribe(sub_id: int) -> None:
    with _LOCK:
        _SUBSCRIBERS.pop(sub_id, None)


def subscribers() -> int:
    with _LOCK:
        return len(_SUBSCRIBERS)


def dropped() -> List[str]:
    """Кто из подписчиков отвалился и почему — для отчётов и отладки GUI."""
    with _LOCK:
        return list(_DROPPED)


def reset() -> None:
    """Только для тестов."""
    global _DROPPED
    with _LOCK:
        _SUBSCRIBERS.clear()
        _DROPPED = []


def status(message: str, *, echo: Optional[bool] = None) -> None:
    """
    Один шаг работы системы, понятный человеку: «изучаю слово «джойстик»...».

    echo=None — печатать в stdout, если не выключен LIPID2_STATUS.
    Колбэки вызываются СНАЧАЛА: GUI важнее, а stdout не может бросить
    исключение так, чтобы мы потеряли подписчика.
    """
    text = str(message).strip()
    if not text:
        return
    with _LOCK:
        sids = list(_SUBSCRIBERS.items())
    for sid, cb in sids:
        try:
            cb(text)
        except Exception as exc:                       # слушатель НЕ роняет мозг
            note = f"подписчик #{sid} отключён: {type(exc).__name__}: {exc}"
            with _LOCK:
                _SUBSCRIBERS.pop(sid, None)
                _DROPPED.append(note)
            sys.stderr.write(f"{PREFIX} ⚠️ {note}\n")
    if echo if echo is not None else _stdout_enabled():
        print(f"{PREFIX} {text}")


def note(message: str) -> None:
    """Событие без изменения текущего статуса (например, итог сеанса)."""
    status(message)


# ═══════════════════════════════════════════════════════════════
#  ОБРАТНЫЙ КАНАЛ: мозг задаёт вопрос и ждёт ответа
# ═══════════════════════════════════════════════════════════════
def ask_handler(handler: Optional[Callable[[str], Optional[str]]]) -> None:
    """
    Установить обработчик вопросов (обычно GUI). None — снять.

    Обработчик вызывается НЕ из GUI-потока (он вызывается из рабочего потока,
    задавшего вопрос), поэтому обязан сам переложить вопрос в Tk через
    `after()` и вернуть ответ, когда пользователь его введёт.
    """
    global _QUESTION_HANDLER
    with _LOCK:
        _QUESTION_HANDLER = handler


def can_ask() -> bool:
    with _LOCK:
        return _QUESTION_HANDLER is not None


def request(prompt: str) -> Optional[str]:
    """
    Спросить у слушателя и дождаться ответа.

    Возвращает текст ответа или None — если спрашивать некого или обработчик
    упал. None — это ЧЕСТНОЕ «знания нет», а не ошибка: вызывающий обязан
    отреагировать как на пустой канал (канал 3 исчерпан).

    Таймаута здесь НЕТ сознательно: ждать ответ человек может минуту, а
    прерывать рабочий поток раньше — значит молча потерятьknowledge, за которое
    система уже потратила два канала. Нужен лимит — ставь его ВНАУЖИ вызова.
    """
    with _LOCK:
        handler = _QUESTION_HANDLER
    text = str(prompt).strip()
    if not text:
        return None
    if handler is None:
        status("⚠️ вопросов принимать некому: обратный канал не подключён "
               "(GUI не вызвал ask_handler) — знание от пользователя недоступно")
        return None
    status(f"спрашиваю у тебя: {text}")
    try:
        answer = handler(text)
    except Exception as exc:
        status(f"⛔ обратный канал упал: {type(exc).__name__}: {exc}")
        return None
    if answer is None or not str(answer).strip():
        status("ответа не получен")
        return None
    out = str(answer).strip()
    status(f"получен ответ: {out[:120]}")
    return out
