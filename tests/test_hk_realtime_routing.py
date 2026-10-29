# -*- coding: utf-8 -*-
"""
Regression tests for Hong Kong realtime quote routing.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from data_provider.base import DataFetcherManager
from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote


class _DummyFetcher:
    def __init__(self, name: str, priority: int, result=None, *, api_ready: bool = False, supports_code=None):
        self.name = name
        self.priority = priority
        self.result = result
        self._api_ready = api_ready
        self.calls = []
        if supports_code is not None:
            self.supports_code = supports_code

    def get_realtime_quote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    def is_api_ready(self) -> bool:
        return self._api_ready


class _DummyQuote:
    @staticmethod
    def has_basic_data():
        return True


def _make_quote(
    *,
    code: str,
    source: RealtimeSource,
    price: float = 100.0,
    volume_ratio=None,
    turnover_rate=None,
    amplitude=None,
    pe_ratio=None,
    pb_ratio=None,
    total_mv=None,
    circ_mv=None,
):
    return UnifiedRealtimeQuote(
        code=code,
        source=source,
        price=price,
        volume_ratio=volume_ratio,
        turnover_rate=turnover_rate,
        amplitude=amplitude,
        pe_ratio=pe_ratio,
        pb_ratio=pb_ratio,
        total_mv=total_mv,
        circ_mv=circ_mv,
    )


class TestHKRealtimeRouting(unittest.TestCase):
    """Ensure HK realtime lookup does not fan out into A-share sources."""

    @patch("src.config.get_config")
    def test_manager_routes_hk_suffix_to_hk_specific_sources_only(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance,akshare_em,tushare",
        )

        yfinance = _DummyFetcher("YfinanceFetcher", 4, result=_DummyQuote())
        efinance = _DummyFetcher("EfinanceFetcher", 0, result={"should": "not be called"})
        akshare = _DummyFetcher("AkshareFetcher", 1, result=None)
        tushare = _DummyFetcher("TushareFetcher", 2, result={"should": "not be called"})

        manager = DataFetcherManager(fetchers=[efinance, akshare, tushare, yfinance])
        quote = manager.get_realtime_quote("1810.HK")

        self.assertIsNotNone(quote)
        self.assertEqual(yfinance.calls, [(("HK01810",), {})])
        self.assertEqual(akshare.calls, [])
        self.assertEqual(efinance.calls, [])
        self.assertEqual(tushare.calls, [])

    @patch("src.config.get_config")
    def test_manager_prefers_configured_api_fetcher_before_yfinance_for_hk(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance,akshare_em,tushare",
        )

        twelvedata = _DummyFetcher(
            "TwelveDataFetcher",
            2,
            result=_make_quote(
                code="HK01810",
                source=RealtimeSource.TWELVEDATA,
                price=88.0,
                volume_ratio=1.1,
                turnover_rate=0.8,
                amplitude=2.3,
                pe_ratio=18.0,
                pb_ratio=3.2,
                total_mv=400_000_000.0,
                circ_mv=350_000_000.0,
            ),
            api_ready=True,
        )
        longbridge = _DummyFetcher("LongbridgeFetcher", 5, result=_DummyQuote(), api_ready=True)
        yfinance = _DummyFetcher("YfinanceFetcher", 0, result=_DummyQuote())
        akshare = _DummyFetcher("AkshareFetcher", 1, result=_DummyQuote())

        manager = DataFetcherManager(fetchers=[yfinance, akshare, longbridge, twelvedata])
        quote = manager.get_realtime_quote("1810.HK")

        self.assertIsNotNone(quote)
        self.assertEqual(twelvedata.calls, [(("HK01810",), {})])
        self.assertEqual(longbridge.calls, [])
        self.assertEqual(yfinance.calls, [])
        self.assertEqual(akshare.calls, [])

    @patch("src.config.get_config")
    def test_manager_keeps_us_index_on_yfinance_first(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance,akshare_em,tushare",
        )

        yfinance = _DummyFetcher(
            "YfinanceFetcher",
            0,
            result=_make_quote(
                code="SPX",
                source=RealtimeSource.FALLBACK,
                price=5000.0,
                volume_ratio=1.0,
                turnover_rate=0.5,
                amplitude=1.1,
                pe_ratio=21.0,
                pb_ratio=4.1,
                total_mv=2_000_000_000.0,
                circ_mv=1_900_000_000.0,
            ),
        )
        twelvedata = _DummyFetcher("TwelveDataFetcher", 2, result=_DummyQuote(), api_ready=True)
        longbridge = _DummyFetcher("LongbridgeFetcher", 5, result=_DummyQuote(), api_ready=True)

        manager = DataFetcherManager(fetchers=[twelvedata, longbridge, yfinance])
        quote = manager.get_realtime_quote("SPX")

        self.assertIsNotNone(quote)
        self.assertEqual(yfinance.calls, [(("SPX",), {})])
        self.assertEqual(twelvedata.calls, [])
        self.assertEqual(longbridge.calls, [])

    @patch("src.config.get_config")
    def test_manager_supplements_us_quote_fields_from_later_fetcher(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance,akshare_em,tushare",
        )

        twelvedata = _DummyFetcher(
            "TwelveDataFetcher",
            2,
            result=_make_quote(
                code="AAPL",
                source=RealtimeSource.TWELVEDATA,
                price=201.23,
            ),
            api_ready=True,
        )
        longbridge = _DummyFetcher(
            "LongbridgeFetcher",
            5,
            result=_make_quote(
                code="AAPL",
                source=RealtimeSource.LONGBRIDGE,
                price=201.23,
                volume_ratio=0.75,
                turnover_rate=1.62,
                amplitude=1.3,
                pe_ratio=28.0,
                pb_ratio=12.5,
                total_mv=1_000_000_000.0,
                circ_mv=900_000_000.0,
            ),
            api_ready=True,
        )
        yfinance = _DummyFetcher("YfinanceFetcher", 0, result=_DummyQuote())

        manager = DataFetcherManager(fetchers=[yfinance, longbridge, twelvedata])
        quote = manager.get_realtime_quote("AAPL")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.source, RealtimeSource.TWELVEDATA)
        self.assertEqual(quote.volume_ratio, 0.75)
        self.assertEqual(quote.turnover_rate, 1.62)
        self.assertEqual(quote.pe_ratio, 28.0)
        self.assertEqual(quote.circ_mv, 900_000_000.0)
        self.assertEqual(twelvedata.calls, [(("AAPL",), {})])
        self.assertEqual(longbridge.calls, [(("AAPL",), {})])
        self.assertEqual(yfinance.calls, [])

    @patch("src.config.get_config")
    def test_manager_keeps_supplementing_after_partial_fill(self, mock_get_config):
        """第二个源只补到一部分字段时，仍允许再问第三个源。"""
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance,akshare_em,tushare",
        )

        twelvedata = _DummyFetcher(
            "TwelveDataFetcher",
            2,
            result=_make_quote(code="AAPL", source=RealtimeSource.TWELVEDATA, price=201.23),
            api_ready=True,
        )
        finnhub = _DummyFetcher(
            "FinnhubFetcher",
            2,
            result=_make_quote(code="AAPL", source=RealtimeSource.FALLBACK, price=201.23, pe_ratio=28.0),
            api_ready=True,
        )
        alphavantage = _DummyFetcher(
            "AlphaVantageFetcher",
            3,
            result=_make_quote(
                code="AAPL",
                source=RealtimeSource.FALLBACK,
                price=201.23,
                volume_ratio=0.75,
                turnover_rate=1.62,
            ),
            api_ready=True,
        )
        yfinance = _DummyFetcher("YfinanceFetcher", 0, result=_DummyQuote())

        manager = DataFetcherManager(fetchers=[twelvedata, finnhub, alphavantage, yfinance])
        quote = manager.get_realtime_quote("AAPL")

        self.assertIsNotNone(quote)
        # 美股实时链首位是 Finnhub（免费档提供实时报价），TwelveData 变为补充源
        self.assertEqual(quote.source, RealtimeSource.FALLBACK)
        self.assertEqual(quote.pe_ratio, 28.0)
        self.assertEqual(quote.volume_ratio, 0.75)
        self.assertEqual(quote.turnover_rate, 1.62)
        self.assertEqual(len(finnhub.calls), 1)
        self.assertEqual(len(twelvedata.calls), 1)
        self.assertEqual(len(alphavantage.calls), 1)
        self.assertEqual(yfinance.calls, [])




class TestUSRealtimeOrdering(unittest.TestCase):
    @patch("src.config.get_config")
    def test_us_realtime_prefers_finnhub_but_daily_order_is_unchanged(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance,akshare_em,tushare",
        )

        twelve = _DummyFetcher("TwelveDataFetcher", 2, result=_DummyQuote(), api_ready=True)
        finnhub = _DummyFetcher("FinnhubFetcher", 2, result=_DummyQuote(), api_ready=True)
        yfinance = _DummyFetcher("YfinanceFetcher", 4, result=_DummyQuote())

        manager = DataFetcherManager(fetchers=[yfinance, twelve, finnhub])

        daily_chain = [f.name for f in manager._get_us_hk_priority_chain("AAPL")]
        realtime_chain = [
            f.name
            for f in manager._prefer_realtime_source(
                manager._get_us_hk_priority_chain("AAPL"), "FinnhubFetcher"
            )
        ]

        # 日线仍然按优先级：TwelveData 的批量历史接口更合适
        self.assertEqual(daily_chain[0], "TwelveDataFetcher")
        # 实时行情把 Finnhub 提到最前（免费档提供美股实时报价）
        self.assertEqual(realtime_chain[0], "FinnhubFetcher")
        self.assertEqual(realtime_chain[-1], "YfinanceFetcher")

    @patch("src.config.get_config")
    def test_hk_realtime_chain_is_untouched(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(
            enable_realtime_quote=True,
            realtime_source_priority="tencent,akshare_sina,efinance,akshare_em,tushare",
        )

        twelve = _DummyFetcher("TwelveDataFetcher", 2, result=_DummyQuote(), api_ready=True)
        finnhub = _DummyFetcher(
            "FinnhubFetcher",
            2,
            result=_DummyQuote(),
            api_ready=True,
            supports_code=lambda code: code.upper() == "AAPL",
        )
        yfinance = _DummyFetcher("YfinanceFetcher", 4, result=_DummyQuote())

        manager = DataFetcherManager(fetchers=[yfinance, twelve, finnhub])
        chain = [
            f.name
            for f in manager._prefer_realtime_source(
                manager._get_us_hk_priority_chain("hk00700"), "FinnhubFetcher"
            )
        ]

        self.assertEqual(chain, ["TwelveDataFetcher", "YfinanceFetcher"])


if __name__ == "__main__":
    unittest.main()
