# -*- coding: utf-8 -*-
"""
Regression tests for HK stock realtime fallback when stock_hk_spot_em fails.
"""

import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pandas as pd

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()
try:
    import json_repair  # noqa: F401
except ImportError:
    if "json_repair" not in sys.modules:
        sys.modules["json_repair"] = MagicMock()

from data_provider import akshare_fetcher as akshare_fetcher_module
from data_provider.akshare_fetcher import AkshareFetcher


class _DummyCircuitBreaker:
    def __init__(self):
        self.failures = []
        self.successes = []

    def is_available(self, source: str) -> bool:
        return True

    def record_success(self, source: str) -> None:
        self.successes.append(source)

    def record_failure(self, source: str, error=None) -> None:
        self.failures.append((source, error))


def _make_spot_em_df():
    return pd.DataFrame(
        [{
            "代码": "00700",
            "名称": "腾讯控股",
            "最新价": 370.0,
            "涨跌幅": 1.5,
            "涨跌额": 5.5,
            "成交量": 10000,
            "成交额": 3700000.0,
            "量比": 1.2,
            "换手率": 0.3,
            "振幅": 2.0,
            "市盈率": 20.0,
            "市净率": 3.5,
            "总市值": 3.5e12,
            "流通市值": 3.5e12,
            "52周最高": 400.0,
            "52周最低": 280.0,
        }]
    )


def _make_spot_df():
    return pd.DataFrame(
        [{
            "代码": "00700",
            "名称": "腾讯控股",
            "最新价": 368.0,
            "涨跌额": 3.5,
            "涨跌幅": 0.96,
            "买入": 367.8,
            "卖出": 368.2,
            "昨收": 364.5,
            "今开": 365.0,
            "最高": 370.0,
            "最低": 364.0,
            "成交量": 9800,
            "成交额": 3606400.0,
        }]
    )


class TestHKRealtimeFallback(unittest.TestCase):
    def setUp(self):
        self.fetcher = AkshareFetcher()
        self.fetcher._enforce_rate_limit = lambda: None
        self.fetcher._set_random_user_agent = lambda: None
        akshare_fetcher_module._hk_realtime_cache.update(
            {
                "data": None,
                "timestamp": 0,
                "ttl": 1200,
                "failure_ttl": 30,
                "last_result": None,
            }
        )

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_em_success_returns_quote_with_name(self, mock_cb):
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.return_value = _make_spot_em_df()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            quote = self.fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.name, "腾讯控股")
        self.assertAlmostEqual(quote.price, 370.0)

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_em_failure_falls_back_to_spot(self, mock_cb):
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.side_effect = Exception("接口异常：数据源不可用")
        ak_mock.stock_hk_spot.return_value = _make_spot_df()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            quote = self.fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.name, "腾讯控股")
        self.assertAlmostEqual(quote.price, 368.0)
        ak_mock.stock_hk_spot.assert_called_once()

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_em_returns_empty_df_falls_back_to_spot(self, mock_cb):
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.return_value = pd.DataFrame(columns=["代码", "名称", "最新价"])
        ak_mock.stock_hk_spot.return_value = _make_spot_df()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            quote = self.fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.name, "腾讯控股")

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_both_fail_returns_none(self, mock_cb):
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.side_effect = Exception("东方财富接口超时")
        ak_mock.stock_hk_spot.side_effect = Exception("新浪接口超时")

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            quote = self.fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNone(quote)

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_circuit_breaker_open_returns_none(self, mock_cb):
        cb = _DummyCircuitBreaker()
        cb.is_available = lambda source: False
        mock_cb.return_value = cb
        ak_mock = MagicMock()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            quote = self.fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNone(quote)
        ak_mock.stock_hk_spot_em.assert_not_called()

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_repeated_em_lookup_uses_hot_cache_without_network_or_wait(self, mock_cb):
        """TTL 内重复查询同一港股不应再次拉全市场，也不应再执行限速等待。"""
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.return_value = _make_spot_em_df()
        self.fetcher._set_random_user_agent = MagicMock()
        self.fetcher._enforce_rate_limit = MagicMock()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            first = self.fetcher._get_hk_realtime_quote("HK00700")
            second = self.fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        ak_mock.stock_hk_spot_em.assert_called_once()
        self.fetcher._set_random_user_agent.assert_called_once()
        self.fetcher._enforce_rate_limit.assert_called_once()

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_cold_cache_is_shared_across_different_hk_codes(self, mock_cb):
        """不同港股代码在同一 TTL 内共用一个全市场快照。"""
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.return_value = _make_spot_em_df()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            self.fetcher._get_hk_realtime_quote("HK00700")
            self.fetcher._get_hk_realtime_quote("HK09988")

        ak_mock.stock_hk_spot_em.assert_called_once()

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_parallel_cold_lookups_share_one_em_request(self, mock_cb):
        """组合快照并发冷查询时，全市场主接口只能由一个线程刷新。"""
        mock_cb.return_value = _DummyCircuitBreaker()
        codes = ["00700", "09988", "03690", "01810"]
        em_df = pd.concat(
            [_make_spot_em_df().assign(代码=code) for code in codes],
            ignore_index=True,
        )
        ak_mock = MagicMock()

        def delayed_em_response():
            time.sleep(0.05)
            return em_df

        ak_mock.stock_hk_spot_em.side_effect = delayed_em_response
        start = threading.Barrier(len(codes))

        def fetch(code):
            start.wait(timeout=5)
            return self.fetcher._get_hk_realtime_quote(code)

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            with ThreadPoolExecutor(max_workers=len(codes)) as pool:
                quotes = list(pool.map(fetch, codes))

        self.assertTrue(all(q is not None for q in quotes))
        ak_mock.stock_hk_spot_em.assert_called_once()

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_failure_is_short_cached(self, mock_cb):
        """冷拉取失败后短时间内复用失败结果，避免连续打接口。"""
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.side_effect = Exception("接口异常")
        ak_mock.stock_hk_spot.return_value = _make_spot_df()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            self.fetcher._get_hk_realtime_quote("HK00700")
            self.fetcher._get_hk_realtime_quote("HK00700")

        ak_mock.stock_hk_spot_em.assert_called_once()

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_stock_name_lookup_reuses_realtime_snapshot(self, mock_cb):
        """名称查询应复用实时行情已拉取的全市场快照，不再单独拉一次。"""
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.return_value = _make_spot_em_df()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            self.assertIsNotNone(self.fetcher._get_hk_realtime_quote("HK00700"))
            name = self.fetcher.get_stock_name("HK00700")

        self.assertEqual(name, "腾讯控股")
        ak_mock.stock_hk_spot_em.assert_called_once()

    @patch("data_provider.akshare_fetcher.get_realtime_circuit_breaker")
    def test_empty_snapshot_is_not_cached_as_success(self, mock_cb):
        """空市场快照不能写入成功缓存。"""
        mock_cb.return_value = _DummyCircuitBreaker()
        ak_mock = MagicMock()
        ak_mock.stock_hk_spot_em.return_value = pd.DataFrame(columns=["代码", "名称", "最新价"])
        ak_mock.stock_hk_spot.return_value = _make_spot_df()

        with patch.dict(sys.modules, {"akshare": ak_mock}):
            quote = self.fetcher._get_hk_realtime_quote("HK00700")

        self.assertIsNotNone(quote)
        self.assertIsNone(akshare_fetcher_module._hk_realtime_cache["data"])
        self.assertEqual(akshare_fetcher_module._hk_realtime_cache["last_result"], "failure")


if __name__ == "__main__":
    unittest.main()
