# -*- coding: utf-8 -*-
"""Tests for the hot-sector stock screener."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.stock_screener_service import (
    Candidate,
    StockScreenerService,
    score_metrics,
)


def _series(start: float = 100.0, slope: float = 0.003, days: int = 120, volume: float = 2_000_000):
    closes = []
    highs = []
    lows = []
    volumes = []
    price = start
    for index in range(days):
        closes.append(price)
        highs.append(price * 1.01)
        lows.append(price * 0.99)
        volumes.append(volume)
        price *= 1.0 + slope
    return {"close": closes, "high": highs, "low": lows, "volume": volumes}


class _Themes:
    def __init__(self, status: str = "ok"):
        self.status = status

    def get_snapshot(self, market: str):
        if self.status != "ok":
            return {"status": self.status, "market": market}
        return {
            "status": "ok",
            "market": market,
            "risk_on": True,
            "leaders": [
                {"symbol": "XLK", "name": "科技", "return_5d": 3.0},
                {"symbol": "XLF", "name": "金融", "return_5d": 2.0},
                {"symbol": "XLV", "name": "医疗", "return_5d": 1.0},
            ],
        }


def _candidates(etf: str, name: str, rank: int):
    mapping = {
        "XLK": [("AAA", "Alpha Tech"), ("BBB", "Beta Tech")],
        "XLF": [("CCC", "Gamma Finance")],
        "XLV": [("DDD", "Delta Health")],
    }
    return [Candidate(symbol, label, etf, name, rank) for symbol, label in mapping.get(etf, [])]


def _history(symbols):
    data = {}
    for symbol in symbols:
        if symbol in ("XLK", "XLF", "XLV"):
            data[symbol] = _series(slope=0.0005, volume=5_000_000)
        elif symbol == "DDD":
            data[symbol] = _series(slope=-0.001, volume=2_000_000)
        else:
            data[symbol] = _series(slope=0.004, volume=8_000_000)
    return data


class ScoreMetricsTest(unittest.TestCase):
    def test_pullback_setup_gets_top_bucket(self) -> None:
        result = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": 2.5,
                "rs_30d": 4.0,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(result["state"], "pullback")
        self.assertGreater(result["score"], 80)

    def test_extended_and_weak_are_penalised(self) -> None:
        extended = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": 5.0,
                "rs_30d": 8.0,
                "dist_ma20_pct": 12.0,
                "volume_ratio_5_20": 1.2,
                "high_20d_proximity": 1.0,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        weak = score_metrics(
            {
                "close": 100,
                "ma50": 110,
                "rs_15d": -2.0,
                "rs_30d": -4.0,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(extended["state"], "extended")
        self.assertEqual(weak["state"], "weak")
        self.assertGreater(extended["score"], weak["score"])

    def test_illiquid_stock_is_dropped(self) -> None:
        result = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": 2.0,
                "rs_30d": 3.0,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 1_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(result["score"], 0.0)

    def test_missing_sector_benchmark_is_rejected(self) -> None:
        result = score_metrics(
            {
                "close": 100,
                "ma50": 90,
                "rs_15d": None,
                "rs_30d": None,
                "dist_ma20_pct": 1.0,
                "volume_ratio_5_20": 0.8,
                "high_20d_proximity": 0.95,
                "dollar_volume_20d": 120_000_000,
                "atr_pct": 3.0,
            },
            sector_rank=1,
        )
        self.assertEqual(result["score"], 0.0)
        self.assertIn("缺少板块基准", result["reason"])

    def test_risk_off_reduces_score(self) -> None:
        metrics = {
            "close": 100,
            "ma50": 90,
            "rs_15d": 2.5,
            "rs_30d": 4.0,
            "dist_ma20_pct": 1.0,
            "volume_ratio_5_20": 0.8,
            "high_20d_proximity": 0.95,
            "dollar_volume_20d": 120_000_000,
            "atr_pct": 3.0,
        }
        risk_on = score_metrics(metrics, sector_rank=1, risk_on=True)
        risk_off = score_metrics(metrics, sector_rank=1, risk_on=False)
        self.assertLess(risk_off["score"], risk_on["score"])
        self.assertIn("risk-off", risk_off["reason"])


class StockScreenerTest(unittest.TestCase):
    def _service(self, themes_status: str = "ok") -> StockScreenerService:
        return StockScreenerService(
            themes_service=_Themes(themes_status),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {},
            cache_ttl_seconds=0,
        )

    def test_screen_returns_ranked_picks_with_plan(self) -> None:
        result = self._service().screen(top_n=3)

        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(len(result["picks"]), 3)
        self.assertTrue(result["picks"])
        self.assertIn("今日主线", result["summary"])
        first = result["picks"][0]
        self.assertGreater(first["score"], 0)
        self.assertIn("buy_trigger", first["plan"])
        self.assertIsNotNone(first["plan"]["stop_loss"])

    def test_hk_is_not_supported(self) -> None:
        result = self._service().screen(market="hk")
        self.assertEqual(result["status"], "not_supported")

    def test_missing_themes_is_not_supported(self) -> None:
        result = self._service(themes_status="not_supported").screen()
        self.assertEqual(result["status"], "not_supported")

    def test_render_markdown_contains_table_and_plan(self) -> None:
        markdown = StockScreenerService.render_markdown(self._service().screen(top_n=2))
        self.assertIn("热门板块选股", markdown)
        self.assertIn("| 板块 |", markdown)
        self.assertIn("交易计划", markdown)

    def test_earnings_window_filter_skips_name(self) -> None:
        service = StockScreenerService(
            themes_service=_Themes(),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {
                symbol: {"days_to_earnings": 2, "quality_ok": True}
                for symbol in symbols
            },
            cache_ttl_seconds=0,
        )
        result = service.screen(top_n=3)
        self.assertEqual(result["picks"], [])
        self.assertGreater(result["filters"]["earnings_skipped"], 0)

    def test_low_quality_name_is_skipped(self) -> None:
        service = StockScreenerService(
            themes_service=_Themes(),
            candidate_provider=_candidates,
            history_provider=_history,
            fundamental_provider=lambda symbols: {
                symbol: {"quality_ok": False, "quality_notes": ["净利率为负"]}
                for symbol in symbols
            },
            cache_ttl_seconds=0,
        )
        result = service.screen(top_n=3)
        self.assertEqual(result["picks"], [])
        self.assertGreater(result["filters"]["quality_skipped"], 0)

    def test_correlated_names_are_deduplicated(self) -> None:
        result = self._service().screen(top_n=10)
        symbols = [item["symbol"] for item in result["picks"]]
        # AAA and BBB use the same synthetic series -> only one should survive.
        self.assertFalse({"AAA", "BBB"}.issubset(set(symbols)))
        self.assertGreater(result["filters"]["correlation_skipped"], 0)

    def test_walk_forward_validation_runs(self) -> None:
        service = self._service()
        candidates = [
            Candidate("AAA", "Alpha Tech", "XLK", "科技", 1),
            Candidate("CCC", "Gamma Finance", "XLF", "金融", 2),
        ]
        history = {
            "SPY": _series(slope=0.0005),
            "XLK": _series(slope=0.0005),
            "XLF": _series(slope=0.0005),
            "XLV": _series(slope=0.0005),
            "AAA": _series(slope=0.0015),
            "CCC": _series(slope=0.0010),
        }
        result = service._walk_forward_validation(
            candidates=candidates,
            history=history,
            top_n=2,
            risk_on=True,
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["samples"], 0)
        self.assertIn("hit_rate_10d", result)


if __name__ == "__main__":
    unittest.main()
