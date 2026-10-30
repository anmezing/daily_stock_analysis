# -*- coding: utf-8 -*-
"""启动模式判定的回归测试。

覆盖两个曾经把服务器 `analyzer` 容器坑住的语义：

1. ``WEBUI_ENABLED=true`` 不应该把一次性分析命令变成"只起 Web 服务"；
2. ``RUN_IMMEDIATELY=false`` 只用于"起 Web 且常驻"的进程，进程本身
   只做分析时必须照常执行，否则命令会什么都不做地退出。
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

import main
from src.config import Config


class _DummyConfig(SimpleNamespace):
    def validate(self):
        return []


class MainCliModeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_path = Path(self.temp_dir.name) / ".env"
        self.env_path.write_text("STOCK_LIST=AAPL\n", encoding="utf-8")
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.env_patch = patch.dict(os.environ, {"ENV_FILE": str(self.env_path)}, clear=False)
        self.env_patch.start()
        Config.reset_instance()
        root_logger = logging.getLogger()
        self._original_root_handlers = list(root_logger.handlers)
        self._original_root_level = root_logger.level

    def tearDown(self) -> None:
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if handler not in self._original_root_handlers:
                root_logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        root_logger.setLevel(self._original_root_level)
        os.chdir(self.original_cwd)
        Config.reset_instance()
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def _make_args(self, **overrides):
        defaults = {
            "debug": False,
            "stocks": None,
            "webui": False,
            "webui_only": False,
            "serve": False,
            "serve_only": False,
            "host": "0.0.0.0",
            "port": 8000,
            "backtest": False,
            "market_review": False,
            "schedule": False,
            "no_run_immediately": False,
            "no_notify": True,
            "no_market_review": False,
            "dry_run": False,
            "workers": 1,
            "force_run": False,
            "single_notify": False,
            "no_context_snapshot": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_config(self, **overrides):
        defaults = {
            "log_dir": self.temp_dir.name,
            "webui_enabled": False,
            "dingtalk_stream_enabled": False,
            "feishu_stream_enabled": False,
            "schedule_enabled": False,
            "schedule_time": "18:00",
            "schedule_run_immediately": True,
            "run_immediately": True,
        }
        defaults.update(overrides)
        return _DummyConfig(**defaults)

    def test_resolve_explicit_serve_wins(self) -> None:
        args = self._make_args(serve_only=True)
        config = self._make_config(webui_enabled=False)

        self.assertEqual(main._resolve_web_server_mode(config, args), (True, "explicit"))

    def test_resolve_disabled_webui(self) -> None:
        args = self._make_args()
        config = self._make_config(webui_enabled=False)

        self.assertEqual(main._resolve_web_server_mode(config, args), (False, "disabled"))

    def test_resolve_schedule_mode_blocks_web(self) -> None:
        args = self._make_args(schedule=True)
        config = self._make_config(webui_enabled=True)

        self.assertEqual(main._resolve_web_server_mode(config, args), (False, "schedule"))

    def test_resolve_one_shot_params_block_web(self) -> None:
        args = self._make_args(stocks="AAPL")
        config = self._make_config(webui_enabled=True)

        self.assertEqual(main._resolve_web_server_mode(config, args), (False, "one_shot"))

    def test_resolve_bare_run_keeps_legacy_webui_behaviour(self) -> None:
        args = self._make_args()
        config = self._make_config(webui_enabled=True)

        self.assertEqual(main._resolve_web_server_mode(config, args), (True, "webui_enabled"))

    def test_bare_run_without_web_runs_analysis_even_if_run_immediately_false(self) -> None:
        args = self._make_args()
        config = self._make_config(webui_enabled=False, run_immediately=False)

        with patch("main.parse_arguments", return_value=args), \
             patch("main.get_config", return_value=config), \
             patch("main.setup_logging"), \
             patch("main.run_full_analysis") as run_full_analysis:
            exit_code = main.main()

        self.assertEqual(exit_code, 0)
        run_full_analysis.assert_called_once_with(config, args, None)

    def test_implicit_web_still_honours_run_immediately_false(self) -> None:
        args = self._make_args()
        config = self._make_config(webui_enabled=True, run_immediately=False)

        with patch("main.parse_arguments", return_value=args), \
             patch("main.get_config", return_value=config), \
             patch("main.setup_logging"), \
             patch("main.prepare_webui_frontend_assets", return_value=True), \
             patch("main.start_api_server"), \
             patch("main.start_bot_stream_clients"), \
             patch("main.time.sleep", side_effect=KeyboardInterrupt), \
             patch("main.run_full_analysis") as run_full_analysis:
            exit_code = main.main()

        self.assertEqual(exit_code, 0)
        run_full_analysis.assert_not_called()


if __name__ == "__main__":
    unittest.main()
