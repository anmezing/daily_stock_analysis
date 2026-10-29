# -*- coding: utf-8 -*-
"""YfinanceFetcher 实时行情估值字段（PE/PB）离线测试。"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()
try:
    import json_repair  # noqa: F401
except ImportError:
    if "json_repair" not in sys.modules:
        sys.modules["json_repair"] = MagicMock()

from data_provider.yfinance_fetcher import YfinanceFetcher, _positive_float_or_none


class TestPositiveFloatOrNone(unittest.TestCase):
    def test_keeps_positive_numbers(self):
        self.assertAlmostEqual(_positive_float_or_none(39.67), 39.67)
        self.assertAlmostEqual(_positive_float_or_none("45.08"), 45.08)

    def test_rejects_non_positive_and_garbage(self):
        for value in (0, 0.0, -12.5, None, "", "n/a", float("nan"), float("inf")):
            self.assertIsNone(_positive_float_or_none(value), msg=f"value={value!r}")


class TestRealtimeValuation(unittest.TestCase):
    """美股实时行情应带回 PE/PB，而不是固定 None。"""

    def _make_fast_info(self):
        fast_info = MagicMock()
        fast_info.lastPrice = 150.0
        fast_info.previousClose = 148.0
        fast_info.open = 149.0
        fast_info.dayHigh = 151.0
        fast_info.dayLow = 148.0
        fast_info.lastVolume = 5000000
        fast_info.marketCap = 3000000000000
        return fast_info

    @patch("yfinance.Ticker")
    def test_us_quote_populates_pe_and_pb(self, mock_ticker_cls):
        ticker = MagicMock()
        ticker.fast_info = self._make_fast_info()
        ticker.info = {
            "longName": "Apple Inc.",
            "trailingPE": 39.67,
            "priceToBook": 45.08,
        }
        mock_ticker_cls.return_value = ticker

        quote = YfinanceFetcher().get_realtime_quote("AAPL")

        self.assertIsNotNone(quote)
        self.assertAlmostEqual(quote.pe_ratio, 39.67)
        self.assertAlmostEqual(quote.pb_ratio, 45.08)

    @patch("yfinance.Ticker")
    def test_us_quote_without_valuation_keeps_none(self, mock_ticker_cls):
        ticker = MagicMock()
        ticker.fast_info = self._make_fast_info()
        ticker.info = {"longName": "Apple Inc."}
        mock_ticker_cls.return_value = ticker

        quote = YfinanceFetcher().get_realtime_quote("AAPL")

        self.assertIsNotNone(quote)
        self.assertIsNone(quote.pe_ratio)
        self.assertIsNone(quote.pb_ratio)

    @patch("yfinance.Ticker")
    def test_info_failure_does_not_break_quote(self, mock_ticker_cls):
        ticker = MagicMock()
        ticker.fast_info = self._make_fast_info()
        type(ticker).info = property(lambda self: (_ for _ in ()).throw(RuntimeError("info boom")))
        mock_ticker_cls.return_value = ticker

        quote = YfinanceFetcher().get_realtime_quote("AAPL")

        self.assertIsNotNone(quote)
        self.assertIsNone(quote.pe_ratio)
        self.assertIsNone(quote.pb_ratio)
        self.assertAlmostEqual(quote.price, 150.0)


if __name__ == "__main__":
    unittest.main()
