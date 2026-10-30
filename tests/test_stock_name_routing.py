# -*- coding: utf-8 -*-
"""股票名称解析路由测试。

TwelveData 免费档只有 8 credits/分钟，名称解析原先单独维护了一份"TwelveData 开头"
的硬编码表，14 只美股各取一次名称就把额度耗掉一半。现在名称解析跟随实时行情链路
（美股 Finnhub 优先），既省额度又和实际报价源一致。
"""

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


class StockNameRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = _manager()
        self.calls = []

    def _stub(self, name: str, value):
        fetcher = next((f for f in self.manager._fetchers if f.name == name), None)
        if fetcher is None:
            return None

        def _fn(_code):
            self.calls.append(name)
            if isinstance(value, Exception):
                raise value
            if callable(value):
                return value(name)
            return value

        fetcher.get_stock_name = _fn
        return fetcher

    def test_us_name_comes_from_finnhub_first(self) -> None:
        self._stub("FinnhubFetcher", "Meta Platforms Inc")
        self._stub("TwelveDataFetcher", "Meta Platforms Inc. Class A")
        self._stub("YfinanceFetcher", "Meta Platforms, Inc.")

        name = self.manager.get_stock_name("META", allow_realtime=False)

        self.assertEqual(name, "Meta Platforms Inc")
        self.assertEqual(self.calls[0], "FinnhubFetcher")
        self.assertNotIn("TwelveDataFetcher", self.calls)

    def test_cooled_down_twelvedata_is_not_asked_for_names(self) -> None:
        twelvedata = self._stub("TwelveDataFetcher", "should not be used")
        self._stub("FinnhubFetcher", RuntimeError("boom"))
        self._stub("YfinanceFetcher", "Meta Platforms, Inc.")
        twelvedata._key_cooldown = {key: 10 ** 12 for key in twelvedata._api_keys}

        name = self.manager.get_stock_name("META", allow_realtime=False)

        self.assertEqual(name, "Meta Platforms, Inc.")
        self.assertNotIn("TwelveDataFetcher", self.calls)

    def test_hk_chain_keeps_akshare_as_a_last_resort(self) -> None:
        self._stub("TwelveDataFetcher", RuntimeError("throttled"))
        self._stub("YfinanceFetcher", RuntimeError("no data"))
        self._stub("AkshareFetcher", "腾讯控股")

        name = self.manager.get_stock_name("hk00700", allow_realtime=False)

        self.assertEqual(name, "腾讯控股")
        self.assertEqual(self.calls[-1], "AkshareFetcher")

    def test_name_is_cached_after_first_hit(self) -> None:
        self._stub("FinnhubFetcher", "Meta Platforms Inc")

        first = self.manager.get_stock_name("META", allow_realtime=False)
        second = self.manager.get_stock_name("META", allow_realtime=False)

        self.assertEqual(first, second)
        self.assertEqual(self.calls.count("FinnhubFetcher"), 1)


if __name__ == "__main__":
    unittest.main()
