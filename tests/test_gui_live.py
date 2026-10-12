"""
Живое окно под xvfb. Долго (~15 с: импорт torch неизбежен, он в цепочке
core.sdr -> core.encoder), но это единственная проверка, что окно вообще
открывается: компиляция и FakeBrain-тесты её не заменяют.

Что реально проверяется (см. ui/headless.py):
  * вкладки меняют и панель, и поле ввода (иначе Enter в видимом поле
    обрабатывает скрытое);
  * ПКМ-копирование из журнала (Tk 8.6 не знает readonly для Text, а disabled
    запрещает выделение — проверено на этой машине);
  * в журнал нельзя печатать, но можно выделять и копировать;
  * вопрос мозгу доходит до человека, ответ возвращается, закрытие окна
    отпускает рабочий поток;
  * выключенный канал отказывает ЯВНО и это видно в журнале.

Если xvfb-run не установлен — тест пропускается, но НЕ «зеленеет молча»:
причина пропуска печатается в stderr.
"""
import os
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADLESS = os.path.join(ROOT, "ui", "headless.py")


@unittest.skipIf(shutil.which("xvfb-run") is None,
                 "нет xvfb-run: GUI не проверить (не путать с «проверено»)")
class TestGuiUnderXvfb(unittest.TestCase):

    def test_window_builds_and_behaves(self):
        env = dict(os.environ,
                   LIPID2_STATUS="0",
                   LIPID_HEADLESS_FAST="1",   # не грузить модель ради баннера
                   OMP_NUM_THREADS="2")
        proc = subprocess.run(
            ["xvfb-run", "-a", sys.executable, HEADLESS],
            cwd=ROOT, env=env, timeout=300,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = proc.stdout.decode("utf-8", "replace")
        # причина отказа нужна полная: «returncode 1» ничего не объясняет
        self.assertEqual(proc.returncode, 0,
                         f"headless-проверка окна упала:\n{out[-4000:]}")
        self.assertIn("GUI:", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
