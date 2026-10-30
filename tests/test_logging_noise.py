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


class ThirdPartyWarningFilterTest(unittest.TestCase):
    def test_inplace_warning_is_ignored(self) -> None:
        from src.logging_config import _should_ignore_warning

        self.assertTrue(
            _should_ignore_warning("The inplace keyword in DataFrame.sort_index is deprecated")
        )
        self.assertTrue(_should_ignore_warning("See ...#deprecation-inplace for more details. PDEP-8"))

    def test_unrelated_warning_is_not_ignored(self) -> None:
        from src.logging_config import _should_ignore_warning

        self.assertFalse(_should_ignore_warning("Another warning"))

    def test_setup_logging_installs_warning_hook(self) -> None:
        """运行期 simplefilter("always") 会压过 filterwarnings，因此必须接管 showwarning。"""
        import warnings

        from src.logging_config import setup_logging

        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                setup_logging(log_prefix="w", log_dir=os.path.join(temp_dir, "logs"))
                self.assertTrue(getattr(warnings.showwarning, "__name__", "") == "_filtered_showwarning")
            finally:
                os.chdir(cwd)
                logging.getLogger().handlers.clear()


if __name__ == "__main__":
    unittest.main()
