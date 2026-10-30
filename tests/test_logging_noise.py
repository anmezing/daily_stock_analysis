# -*- coding: utf-8 -*-
"""第三方日志噪音过滤的回归测试。"""

import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logging_config import ThirdPartyNoiseFilter, setup_logging


class ThirdPartyNoiseFilterTest(unittest.TestCase):
    def _record(self, name: str, level: int) -> logging.LogRecord:
        return logging.LogRecord(name=name, level=level, pathname="x.py", lineno=1, msg="m", args=(), exc_info=None)

    def test_litellm_info_is_dropped(self) -> None:
        noise_filter = ThirdPartyNoiseFilter()

        self.assertFalse(noise_filter.filter(self._record("LiteLLM Router", logging.INFO)))
        self.assertFalse(noise_filter.filter(self._record("litellm.utils", logging.DEBUG)))

    def test_litellm_error_is_kept(self) -> None:
        noise_filter = ThirdPartyNoiseFilter()

        self.assertTrue(noise_filter.filter(self._record("LiteLLM Router", logging.ERROR)))

    def test_application_logs_are_kept(self) -> None:
        noise_filter = ThirdPartyNoiseFilter()

        self.assertTrue(noise_filter.filter(self._record("src.core.pipeline", logging.INFO)))
        self.assertTrue(noise_filter.filter(self._record("main", logging.DEBUG)))

    def test_yfinance_and_httpx_info_dropped(self) -> None:
        noise_filter = ThirdPartyNoiseFilter()

        self.assertFalse(noise_filter.filter(self._record("yfinance", logging.INFO)))
        self.assertFalse(noise_filter.filter(self._record("httpx", logging.INFO)))


class SetupLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        self.records = []

        class _Collector(logging.Handler):
            def emit(inner_self, record):  # noqa: N805 - logging.Handler API
                self.records.append(record)

        self.collector = _Collector(level=logging.DEBUG)

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        logging.getLogger().handlers.clear()
        self.temp_dir.cleanup()

    def test_setup_logging_quiets_litellm_loggers(self) -> None:
        os.chdir(self.temp_dir.name)
        setup_logging(log_prefix="t", log_dir=os.path.join(self.temp_dir.name, "logs"))

        self.assertGreaterEqual(logging.getLogger("LiteLLM Router").level, logging.ERROR)
        self.assertGreaterEqual(logging.getLogger("litellm").level, logging.ERROR)


if __name__ == "__main__":
    unittest.main()
