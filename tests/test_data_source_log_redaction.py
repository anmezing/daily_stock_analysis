# -*- coding: utf-8 -*-
"""数据源失败日志的脱敏与截断测试。

线上日志里出现过完整请求 URL：

    [数据源失败 1/3] [TwelveDataFetcher] MU: reason=[TwelveDataFetcher] 网络请求失败: 429 ...
    for url: https://api.twelvedata.com/time_series?...&apikey=<真实 KEY>

requests 的 HTTPError 会把 URL 一起带进异常消息，直接写日志等于把密钥落盘。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.base import _redact_url_secrets, summarize_exception


class RedactSecretsTest(unittest.TestCase):
    def test_apikey_query_param_is_masked(self) -> None:
        text = "HTTPError for url: https://api.twelvedata.com/time_series?symbol=MU&apikey=abc123"

        self.assertEqual(
            _redact_url_secrets(text),
            "HTTPError for url: https://api.twelvedata.com/time_series?symbol=MU&apikey=***",
        )

    def test_common_secret_keys_are_masked(self) -> None:
        for param in ("token", "api_key", "access_token", "secret", "key"):
            text = f"https://example.com/x?{param}=s3cr3t&symbol=AAPL"

            self.assertNotIn("s3cr3t", _redact_url_secrets(text), param)

    def test_non_secret_params_are_untouched(self) -> None:
        text = "https://example.com/x?symbol=AAPL&interval=1day"

        self.assertEqual(_redact_url_secrets(text), text)


class SummarizeExceptionTest(unittest.TestCase):
    def test_secrets_are_removed_from_the_summary(self) -> None:
        exc = Exception(
            "[Finnhub] 请求失败: 403 Client Error for url: "
            "https://finnhub.io/api/v1/stock/candle?symbol=MU&token=abcdef123456"
        )

        _, reason = summarize_exception(exc)

        self.assertNotIn("abcdef123456", reason)
        self.assertIn("token=***", reason)

    def test_long_messages_are_truncated(self) -> None:
        exc = Exception("x" * 500)

        _, reason = summarize_exception(exc)

        self.assertLessEqual(len(reason), 243)
        self.assertTrue(reason.endswith("..."))

    def test_whitespace_is_collapsed(self) -> None:
        exc = Exception("line one\n\n   line two")

        _, reason = summarize_exception(exc)

        self.assertEqual(reason, "line one line two")


if __name__ == "__main__":
    unittest.main()
