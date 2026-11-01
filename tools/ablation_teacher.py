"""
АБЛЯЦИЯ: помогает ли СЛОТ ДЛЯ ОТНОШЕНИЯ в запросе к локальному Учителю.

ВОПРОС ОДИН: если в запросе появится поле, куда модель кладёт «чем концепт
является по отношению к кому», перестанет ли пропадать «жена сына»?

ОДНА ПЕРЕМЕННАЯ. Отличаются РОВНО ТРИ связанных куска одного и того же
изменения (запрос к модели):
  1. в system_prompt добавлен инвариант 4 про definition/relates_to;
  2. в ПРИМЕРЕ структуры выхода появились эти два поля;
  3. description поля content в схеме инструмента перечисляет схему ответа.
ВСЁ ОСТАЛЬНОЕ ИДЕНТИЧНО БОЮ: те же max_steps=6, temperature=0.1,
reasoning_effort=none, tool_choice НЕ выставлен, max_tokens заводской.
Прошлые мои пробы mixesали правки (tool_choice=required + max_tokens=600 +
один шаг) и дали 2.6 с — та цифра к production-пути отношения НЕ имеет и в
этом файле не используется.

ЧТО СЧИТАЕТСЯ ПОБЕДОЙ (по цели хозяина):
  * в записи «невестка» появилось слово «сын» и связка «жена сына»;
  * иерархия НЕ ДЕГРАДИРОВАЛА — замеры Control на «кошка» и «флюгер»
    обязаны остаться классификационными (правка не имеет права сломать
    «кошка IS_A кошачьи», это заперто другими тестами);
  * цена прогона записана ЧИСЛОМ.

ЧТО ПРАВИЛЬНО И ПРИ ПРОИГРЫШЕ. Если «сын» не появился — правка промпта
бесполезна, и это надо писать в docstring, как сделал
test_teacher_prompt_contract («правка НЕ подняла качество иерархии»).
Правка в пустоту запрещена.

ЖИВАЯ МОДЕЛЬ (localhost:8080). В мир и в живой каталог НЕ пишет: свой mktemp,
живой lipid_world.json сверяется по хэшу.

Запуск:
  LIPID2_STATUS=0 .venv/bin/python tools/ablation_teacher.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REAL = os.path.join(ROOT, "data", "memory", "lipid_world.json")


def digest(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


real_before = digest(REAL)
os.environ["LIPID2_STATUS"] = "0"

from teachers.teacher1 import AgentTeacherV2      # noqa: E402

# ── ЕДИНСТВЕННАЯ ПРАВКА ЗАПРОСА (три куска одного изменения) ──────────
INVARIANT_4 = (
    "4. ГЛАВНОЕ ЗНАНИЕ — СВЯЗЬ, А НЕ ВЕРХНИЙ КЛАСС (`definition`, "
    "`relates_to`):\n"
    "   - `definition` — ОДНО предложение о том, чем концепт ЯВЛЯЕТСЯ по "
    "отношению к другому (слова родства, должности, части: «жена сына», "
    "«мать мужа», «капитан корабля»).\n"
    "   - `relates_to` — ИМЕНА тех сущностей, на которые эта связь указывает.\n"
    "   - Если такая связь есть, верхние классы НЕ главнее её: «люди», "
    "«существа», «объекты» не отвечают на вопрос «кем приходится».\n\n"
)
EXAMPLE_FIELDS = (
    '    "definition": "домашнее хищное животное",\n'
    '    "relates_to": ["хищник", "дом"],\n'
)
CONTENT_DESC = (
    'JSON вида {"base_entity":{"concept":"...","definition":"ОДНО предложение'
    ', чем концепт является по отношению к кому/чему","relates_to":["имена'
    ' сущностей, на которые указывает эта связь"],"hierarchy":["ближайший'
    ' класс","...","верхний корень"],"properties":["признак"],"actions":[{'
    '"name":"действие","action_profile":[K,M,S]}]},"derived_entity":{...}}'
)

WORDS = ["невестка", "свекровь", "жена", "кошка", "флюгер"]
REPEATS = int(os.environ.get("REPEATS", "1"))


class Patched(AgentTeacherV2):
    """Учитель с подменённым ТРАНСПОРТОМ: правит исходящий payload.

    Правка исходника функции сразу сделала бы невозможным сравнение «до/после»
    на одной версии кода, а разброс модели между прогонами больше любого
    эффекта правки (это написано в test_teacher_prompt_contract). Подменой
    транспорта обе пробы идут в одном процессе и на одном коде.
    """

    variant = "base"

    def _patch(self, payload):
        if self.variant != "rel":
            return payload
        msgs = payload.get("messages") or []
        if msgs and msgs[0].get("role") == "system":
            sys_txt = msgs[0]["content"]
            if "ПРИМЕР СТРУКТУРЫ ВЫХОДА" in sys_txt:
                sys_txt = sys_txt.replace("ПРИМЕР СТРУКТУРЫ ВЫХОДА",
                                          INVARIANT_4 + "ПРИМЕР СТРУКТУРЫ ВЫХОДА")
            sys_txt = sys_txt.replace(
                '    "properties": ["пушистая", "гибкая", "хищная"],\n',
                '    "properties": ["пушистая", "гибкая", "хищная"],\n'
                + EXAMPLE_FIELDS)
            msgs[0]["content"] = sys_txt
        for t in payload.get("tools") or []:
            props = (t.get("function", {}).get("parameters", {})
                     .get("properties", {}))
            if "content" in props:
                props["content"]["description"] = CONTENT_DESC
        return payload


def run(word, variant):
    d = tempfile.mkdtemp(prefix=f"abl_{variant}_")
    real = urllib.request.urlopen

    def patched(req, timeout=None):
        try:
            p = json.loads(req.data.decode())
            t.variant = variant
            req.data = json.dumps(t._patch(p)).encode()
        except Exception:
            pass
        return real(req, timeout=timeout)

    t = Patched(host=os.environ.get("LIPID2_LLM_HOST", "http://localhost:8080"),
                memory_dir=d)
    urllib.request.urlopen = patched
    t0 = time.time()
    try:
        t.run_agent(base_concept=word, derived_concept=word)
    finally:
        urllib.request.urlopen = real
    dt = time.time() - t0
    path = os.path.join(d, "lipid_world.json")
    ent = {}
    if os.path.exists(path):
        ent = json.load(open(path)).get("entities", {}).get(word, {})
    shutil.rmtree(d, ignore_errors=True)
    return dt, ent, (t.last_failure or "")


def summarize(word, dt, ent, err):
    blob = json.dumps(ent, ensure_ascii=False).lower()
    print(f"    {dt:6.1f} с | definition={str(ent.get('definition'))[:60]!r}")
    print(f"            relates_to={ent.get('relates_to')}")
    print(f"            hierarchy={ent.get('hierarchy')}")
    print(f"            properties={ent.get('properties')}")
    if err:
        print(f"            last_failure: {str(err)[:90]}")
    return blob


print("=" * 78)
print("АБЛЯЦИЯ: запрос без слота отношения (base) против запроса со слотом (rel)")
print(f"  модель: {os.environ.get('LIPID2_LLM_HOST','http://localhost:8080')}"
      f"  повторов на слово: {REPEATS}")
print("=" * 78)

results = {}
for variant in ("base", "rel"):
    print(f"\n--- вариант: {variant} ---")
    for word in WORDS:
        for i in range(REPEATS):
            dt, ent, err = run(word, variant)
            print(f"  «{word}» прогон {i+1}:")
            blob = summarize(word, dt, ent, err)
            results.setdefault((variant, word), []).append(
                {"dt": dt, "has_syn": "сын" in blob,
                 "has_zhena": "жен" in blob,
                 "hier_len": len(ent.get("hierarchy") or []),
                 "has_def": bool(ent.get("definition")),
                 "rel": ent.get("relates_to")})

print("\n" + "=" * 78)
print("СВОДКА: есть ли в записи слово «сын» (цель хозяина)")
print("=" * 78)
fail = []
for word in ("невестка", "свекровь", "жена"):
    b = results.get(("base", word), [])
    r = results.get(("rel", word), [])
    bs = sum(x["has_syn"] for x in b)
    rs = sum(x["has_syn"] for x in r)
    bd = sum(x["has_def"] for x in b)
    rd = sum(x["has_def"] for x in r)
    bt = sum(x["dt"] for x in b) / max(1, len(b))
    rt = sum(x["dt"] for x in r) / max(1, len(r))
    print(f"  {word:10} base: «сын» {bs}/{len(b)}, definition {bd}/{len(b)}, "
          f"{bt:.1f} с   |   rel: «сын» {rs}/{len(r)}, definition {rd}/{len(r)}, "
          f"{rt:.1f} с")

print("\n  КОНТРОЛЬ ОБРАТНОЙ СТОРОНЫ: классификация не имеет права деградировать")
for word in ("кошка", "флюгер"):
    b = results.get(("base", word), [])
    r = results.get(("rel", word), [])
    bh = sum(x["hier_len"] for x in b) / max(1, len(b))
    rh = sum(x["hier_len"] for x in r) / max(1, len(r))
    print(f"  {word:10} длина иерархии: base {bh:.1f} -> rel {rh:.1f}")
    if r and rh < bh - 1:
        fail.append(f"правка СЛОМАЛА классификацию у «{word}»: иерархия "
                    f"укоротилась {bh:.1f} -> {rh:.1f}")

if not any(results.get(("rel", w), [{}])[0].get("has_syn")
           for w in ("невестка", "свекровь") if results.get(("rel", w))):
    fail.append("ПРАВКА НЕ ПОМОГЛА: ни в одном прогоне варианта rel слово «сын» "
                "в запись не попало — правка промпта бесполезна, чинить надо "
                "не запрос")

real_after = digest(REAL)
print(f"\n  живой мир: {real_before} -> {real_after} "
      f"{'НЕ ТРОНУТ' if real_before == real_after else 'ИЗМЕНЁН!!!'}")
if real_before != real_after:
    fail.append("ИЗОЛЯЦИЯ: абляция изменила живой lipid_world.json")
if fail:
    print("\nКРАСНЫЙ:")
    for f in fail:
        print("  ✗ " + f)
    raise SystemExit(1)
print("\nЗЕЛЁНЫЙ: слот отношения даёт знание, классификация цела")
