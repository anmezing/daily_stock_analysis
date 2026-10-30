# -*- coding: utf-8 -*-
"""数据源链路可用性过滤的回归测试。

线上表象：TwelveData 进入 429 冷却、Finnhub 日线被判付费专属之后，每只股票仍然
"尝试 → 失败 → 切换"三连，日志里全是 ERROR/WARNING，看上去像程序坏了。

约定：链路组装阶段就跳过当前不可用的源，并让"可预期跳过"只记 DEBUG。
"""

import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()


def _manager():
    from data_provider.base import DataFetcherManager

    with patch("src.config.get_config") as mock_config:
        mock_config.return_value = MagicMock(
            finnhub_api_key="sk-test",
            twelvedata_api_key="td-test",
            tushare_token=None,
            longbridge_app_key=None,
            longbridge_app_secret=None,
            longbridge_access_token=None,
            tickflow_api_key=None,
            alphavantage_api_key=None,
        )
        return DataFetcherManager()


class SourceChainAvailabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = _manager()
        self.twelvedata = next(f for f in self.manager._fetchers if f.name == "TwelveDataFetcher")
        self.finnhub = next(f for f in self.manager._fetchers if f.name == "FinnhubFetcher")

    def _daily_chain(self, code="AAPL"):
        return [f.name for f in self.manager._get_us_hk_priority_chain(code)]

    def _realtime_chain(self, code="AAPL"):
        return [f.name for f in self.manager._get_us_hk_priority_chain(code, daily=False)]

    def _cool_down_all_keys(self) -> None:
        fetcher = self.twelvedata
        fetcher._key_cooldown = {key: 10 ** 12 for key in fetcher._api_keys}

    def test_rate_limited_source_leaves_both_chains(self) -> None:
        self.assertIn("TwelveDataFetcher", self._daily_chain())
        self.assertIn("TwelveDataFetcher", self._realtime_chain())

        self._cool_down_all_keys()  # 所有 Key 都在冷却

        self.assertNotIn("TwelveDataFetcher", self._daily_chain())
        self.assertNotIn("TwelveDataFetcher", self._realtime_chain())

    def test_candle_forbidden_only_removes_the_daily_chain_entry(self) -> None:
        self.finnhub._candle_forbidden = True

        # 日线接口不可用 -> 日线链路跳过；实时报价仍然可用
        self.assertNotIn("FinnhubFetcher", self._daily_chain())
        self.assertIn("FinnhubFetcher", self._realtime_chain())

    def test_yfinance_stays_as_the_last_fallback(self) -> None:
        self._cool_down_all_keys()
        self.finnhub._candle_forbidden = True

        self.assertEqual(self._daily_chain(), ["YfinanceFetcher"])


class ExpectedSkipLoggingTest(unittest.TestCase):
    def test_rate_limit_error_is_flagged_as_expected_skip(self) -> None:
        from data_provider.base import DataFetchError
        from data_provider.twelvedata_fetcher import TwelveDataRateLimited

        manager = _manager()
        fetcher = next(f for f in manager._fetchers if f.name == "TwelveDataFetcher")

        with patch.object(fetcher, "_fetch_raw_data", side_effect=TwelveDataRateLimited("冷却中")):
            with self.assertLogs("data_provider.base", level=logging.DEBUG) as captured:
                with self.assertRaises(DataFetchError) as ctx:
                    fetcher.get_daily_data("AAPL", days=5)

        self.assertTrue(getattr(ctx.exception, "expected_skip", False))
        levels = {record.levelname for record in captured.records}
        self.assertNotIn("ERROR", levels)


if __name__ == "__main__":
    unittest.main()
