# -*- coding: utf-8 -*-
"""Prompt blocks for market themes, relative strength and volume ratio."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analyzer import GeminiAnalyzer


def _context(**overrides):
    context = {
        "code": "AAPL",
        "stock_name": "Apple",
        "date": "2026-10-30",
        "analysis_date": "2026-10-30",
        "market": "us",
        "market_context": {"market": "us"},
        "is_intraday": True,
        "today": {
            "date": "2026-10-30",
            "close": 344.63,
            "open": 340.0,
            "high": 346.0,
            "low": 339.0,
            "pct_chg": 1.2,
            "volume": 10_000_000,
            "ma5": 345.68,
            "ma10": 345.08,
            "ma20": 342.08,
            "ma50": 333.04,
            "ma200": 300.0,
        },
        "history": [
            {"date": f"d{index}", "open": 340, "high": 350, "low": 338, "close": 344, "pct_chg": 0.2}
            for index in range(30)
        ],
        "trend_analysis": {
            "trend_status": "震荡",
            "ma_alignment": "中性",
            "trend_strength": 55,
            "bias_ma5": 0.5,
            "volume_status": "平量",
            "volume_ratio_5d": 0.82,
            "volume_trend": "平量",
            "buy_signal": "观望",
            "signal_score": 50,
        },
        "realtime": {"price": 344.63, "turnover_rate": 0.1},
        "market_themes": {
            "status": "ok",
            "risk_on": False,
            "benchmarks": [
                {
                    "symbol": "SPY",
                    "name": "标普500",
                    "return_1d": -0.5,
                    "return_5d": -1.2,
                    "return_20d": -2.0,
                    "above_ma50": False,
                    "above_ma200": True,
                }
            ],
            "leaders": [
                {"symbol": "XLV", "name": "医疗", "return_1d": 0.4, "return_5d": 3.1}
            ],
            "laggards": [
                {"symbol": "XLK", "name": "科技", "return_1d": -1.0, "return_5d": -2.0}
            ],
        },
        "relative_strength": {
            "benchmark": "SPY",
            "stock_return_20d": 5.2,
            "benchmark_return_20d": -2.0,
            "excess_20d": 7.2,
            "stock_return_60d": 10.0,
            "benchmark_return_60d": 1.0,
            "excess_60d": 9.0,
            "label": "强于基准（领涨）",
        },
    }
    context.update(overrides)
    return context


class PromptMarketBlocksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)

    def test_prompt_includes_themes_and_relative_strength(self) -> None:
        prompt = self.analyzer._format_prompt(_context(), "Apple", None)

        self.assertIn("当前市场热点与板块轮动", prompt)
        self.assertIn("医疗 XLV", prompt)
        self.assertIn("相对强弱（个股 vs SPY）", prompt)
        self.assertIn("强于基准（领涨）", prompt)

    def test_intraday_volume_ratio_is_not_shown_as_a_number(self) -> None:
        prompt = self.analyzer._format_prompt(_context(), "Apple", None)

        self.assertIn("盘中累计量，待收盘确认", prompt)
        self.assertIn("5日量比（日K）", prompt)
        self.assertIn("盘中待收盘确认", prompt)

    def test_missing_themes_does_not_break_prompt(self) -> None:
        prompt = self.analyzer._format_prompt(
            _context(market_themes={}, relative_strength={}),
            "Apple",
            None,
        )
        self.assertNotIn("当前市场热点与板块轮动", prompt)
        self.assertNotIn("相对强弱（个股 vs SPY）", prompt)


if __name__ == "__main__":
    unittest.main()
