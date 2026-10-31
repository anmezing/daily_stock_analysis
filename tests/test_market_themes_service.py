# -*- coding: utf-8 -*-
"""Tests for the market-theme / sector-rotation snapshot."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.market_themes_service import (
    HK_BENCHMARKS,
    HK_THEMES,
    US_BENCHMARKS,
    US_THEMES,
    MarketThemesService,
)


def _series(slope: float, days: int = 90, start: float = 100.0):
    values = []
    price = start
    for _ in range(days):
        values.append(price)
        price *= 1.0 + slope
    return values


class MarketThemesServiceTest(unittest.TestCase):
    def test_us_snapshot_ranks_leaders_and_marks_risk_on(self) -> None:
        slopes = {symbol: 0.0 for symbol in list(US_THEMES) + list(US_BENCHMARKS)}
        slopes.update(
            {
                "XLK": 0.006,
                "XLF": 0.004,
                "XLI": 0.003,
                "XLE": -0.004,
                "XLU": -0.003,
                "XLB": -0.002,
                "SPY": 0.002,
                "QQQ": 0.0025,
            }
        )
        calls = []

        def fetcher(symbols):
            calls.append(list(symbols))
            return {symbol: _series(slopes.get(symbol, 0.0)) for symbol in symbols}

        service = MarketThemesService(fetcher=fetcher, ttl_seconds=60)
        data = service.get_snapshot("us")

        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["risk_on"])
        self.assertEqual(data["leaders"][0]["symbol"], "XLK")
        self.assertIn("XLE", [row["symbol"] for row in data["laggards"]])
        self.assertIn("领涨", data["summary"])
        self.assertEqual(len(calls), 1)

        # Cache hit: no second provider call.
        service.get_snapshot("us")
        self.assertEqual(len(calls), 1)

    def test_hk_snapshot_uses_theme_etfs(self) -> None:
        slopes = {symbol: 0.0 for symbol in list(HK_THEMES) + list(HK_BENCHMARKS)}
        slopes.update({"3033.HK": 0.008, "2828.HK": 0.001, "2822.HK": -0.002, "2800.HK": 0.001})

        service = MarketThemesService(
            fetcher=lambda symbols: {
                symbol: _series(slopes.get(symbol, 0.0)) for symbol in symbols
            },
            ttl_seconds=60,
        )
        data = service.get_snapshot("hk")

        self.assertEqual(data["market"], "hk")
        self.assertEqual(data["leaders"][0]["symbol"], "3033.HK")
        self.assertIn("恒生", data["summary"])

    def test_empty_provider_returns_not_supported(self) -> None:
        service = MarketThemesService(
            fetcher=lambda symbols: {},
            fallback_fetcher=lambda symbols: {},
            ttl_seconds=0,
        )
        data = service.get_snapshot("us")
        self.assertEqual(data["status"], "not_supported")

    def test_cn_is_not_supported(self) -> None:
        service = MarketThemesService(fetcher=lambda symbols: {}, ttl_seconds=0)
        data = service.get_snapshot("cn")
        self.assertEqual(data["status"], "not_supported")


if __name__ == "__main__":
    unittest.main()
