# -*- coding: utf-8 -*-
"""定时任务的补跑与状态记录测试。

线上担心点：容器在计划时间之后才启动（重启、重新部署），当天就不会再跑。
规则：计划时间已过、当天还没成功执行、且距离计划时间不超过 6 小时 -> 启动时补跑一次。
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import scheduler as scheduler_module


class ShouldCatchUpTest(unittest.TestCase):
    def test_before_scheduled_time_does_not_catch_up(self) -> None:
        now = datetime(2026, 10, 30, 9, 0)

        self.assertFalse(scheduler_module.should_catch_up(now, "10:00", None))

    def test_after_schedule_without_run_today_catches_up(self) -> None:
        now = datetime(2026, 10, 30, 10, 30)

        self.assertTrue(scheduler_module.should_catch_up(now, "10:00", "2026-10-29"))

    def test_already_ran_today_does_not_catch_up(self) -> None:
        now = datetime(2026, 10, 30, 10, 30)

        self.assertFalse(scheduler_module.should_catch_up(now, "10:00", "2026-10-30"))

    def test_outside_window_does_not_catch_up(self) -> None:
        now = datetime(2026, 10, 30, 20, 0)

        self.assertFalse(scheduler_module.should_catch_up(now, "10:00", "2026-10-29"))

    def test_invalid_schedule_time_does_not_catch_up(self) -> None:
        now = datetime(2026, 10, 30, 10, 30)

        self.assertFalse(scheduler_module.should_catch_up(now, "25:99", None))


class RunStateFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_path = os.path.join(self.temp_dir.name, "state.json")
        self._patcher = patch.object(scheduler_module, "SCHEDULE_STATE_PATH", self.state_path)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self.temp_dir.cleanup()

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(scheduler_module.load_last_run_date())

    def test_round_trip(self) -> None:
        scheduler_module.mark_run_completed("2026-10-30")

        self.assertEqual(scheduler_module.load_last_run_date(), "2026-10-30")

    def test_corrupted_file_returns_none(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as handle:
            handle.write("{not-json")

        self.assertIsNone(scheduler_module.load_last_run_date())


class RunWithScheduleCatchUpTest(unittest.TestCase):
    def test_catch_up_runs_immediately_when_today_was_missed(self) -> None:
        recorded = {}

        class _FakeScheduler:
            def __init__(self, schedule_time, schedule_time_provider=None):
                self.schedule = type("S", (), {"every": lambda self: self})()

            def set_daily_task(self, task, run_immediately=True):
                recorded["run_immediately"] = run_immediately
                recorded["task"] = task

            def run(self):
                recorded["ran"] = True

        task_calls = []
        with patch.object(scheduler_module, "Scheduler", _FakeScheduler), \
             patch.object(scheduler_module, "load_last_run_date", return_value="2026-10-29"), \
             patch.object(scheduler_module, "mark_run_completed") as mark, \
             patch.object(scheduler_module, "should_catch_up", return_value=True):
            scheduler_module.run_with_schedule(
                task=lambda: task_calls.append("run"),
                schedule_time="10:00",
                run_immediately=False,
            )
            # 真实 Scheduler 会在这里执行一次任务；补跑状态记录由包装函数负责
            recorded["task"]()
            mark.assert_called_once()

        self.assertTrue(recorded["run_immediately"])
        self.assertTrue(recorded["ran"])
        self.assertEqual(task_calls, ["run"])

    def test_no_catch_up_when_today_already_ran(self) -> None:
        recorded = {}

        class _FakeScheduler:
            def __init__(self, schedule_time, schedule_time_provider=None):
                self.schedule = type("S", (), {"every": lambda self: self})()

            def set_daily_task(self, task, run_immediately=True):
                recorded["run_immediately"] = run_immediately

            def run(self):
                recorded["ran"] = True

        with patch.object(scheduler_module, "Scheduler", _FakeScheduler), \
             patch.object(scheduler_module, "load_last_run_date", return_value="2026-10-30"), \
             patch.object(scheduler_module, "should_catch_up", return_value=False):
            scheduler_module.run_with_schedule(
                task=lambda: None,
                schedule_time="10:00",
                run_immediately=False,
            )

        self.assertFalse(recorded["run_immediately"])


if __name__ == "__main__":
    unittest.main()
